import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from CLA.training.models import get_default_model
from CLA.training.training_loop import train_simclr, train_simsiam, train_byol
from CLA.training.lin_evaluator import get_accuracies
from CLA.utils.knn_monitor import knn_monitor
from CLA.utils.data_utils import get_eval_data, get_training_loaders
from CLA.utils.utils import load_config, overwrite_config_vars

MODEL_TRAINER_DICT = {
    'simclr': train_simclr,
    'simsiam': train_simsiam,
    'byol': train_byol
}

MODEL_FUNCTION_NAMES_DICT = {
    'simclr': ['backbone', 'projector'],
    'simsiam': ['backbone', 'projector', 'predictor'],
    'byol': ['backbone', 'projector', 'predictor']
}
def get_model_functions(model, model_function_names):
    model_functions = []
    for function_name in model_function_names:
        if function_name == 'backbone':
            model_functions.append(model.backbone)
        elif function_name == 'projector':
            model_functions.append(model.enc)
        elif function_name == 'predictor':
            model_functions.append(nn.Sequential(model.enc, model.predictor))
    return model_functions

def train_comparisons(
        model_str,
        dataset,
        comparisons,
        comparison_names,
        log_dir,
        finetune_at_end=True,
        do_training=True,
        save_off=True,
        max_embedding_batches=1000
    ):
    """
    This function will accept a list of config changes and run one experiment for each item in that list.

    For example, if we have the following comparisons list:
    comparisons = [
        {'epochs': 64, 'batch_size': 128, 'learning_rate': 0.6},
        {'epochs': 64, 'batch_size': 64, 'learning_rate': 0.2},
    ],
    then we will run two experiments where the default config has been overridden with these parameter values.

    For each set of config overrides, there is a corresponding "comparison_name" 
        which is the name of the directory where this experiment will be saved.
    The experiments' outputs will all be saved in the following directory structure:

        outputs
           |
           |-- "dataset name"
                     |
                     |-- "model str"
                             |
                             |-- "log dir"
                                     |
                                     |-- comparison_name

    
    Parameters:
        - model_str: name of the model being used. currently accepts one of ['simclr', 'simsiam', 'byol']
        - dataset: name of the dataset being used.
            - Consult the keys of the DATASET_CLASSES dict in CLA.utils.data_utils for a full list of accepted datasets
        - comparisons: list of dictionaries specifying config parameter overrides.
        - comparison_names: list of strings -- one for each dictionary in comparisons specifying the name of this experiment
        - log_dir: name of the directory where all experiments will be stored
        - finetune_at_end: whether to run a final eval at the end of training (finetuning will be run throughout training regardless)
        - save_off: whether to save embeddings from the dataset after training
        - max_embedding_batches: max number of batches of embeddings to save for each checkpoint

    """
    base_dir = os.getcwd()
    config_path = 'configs/{}_default.yml'.format(model_str)
    print('\nRunning {} comparisons using config at {}'.format(model_str, config_path))
    args = load_config(config_path, dataset, model_str, log_dir, chdir=False)

    for comparison, comparison_name in zip(comparisons, comparison_names):
        # Reset new working directory and load experiment parameters for this specific run
        print('Dataset: {}'.format(dataset))
        print('Current experiment being run:')
        print('Model: {}; Experiment name: {}'.format(model_str, comparison_name))
        print('Values being changed: {}\n'.format(comparison))
        os.chdir(base_dir)
        args = load_config(config_path, dataset, log_dir, model_str)
        work_dir = args.log_dir
        args = overwrite_config_vars(args, comparison)

        # Change cwd to appropriate experiment subdir
        subdir = os.path.join(work_dir, comparison_name)
        os.makedirs(subdir, exist_ok=True)
        os.chdir(subdir)

        # Train the model and save off the results
        if do_training:
            knn_accs = MODEL_TRAINER_DICT[model_str](args)
        else:
            knn_accs = []

        if finetune_at_end:
            final_knn, finetune_results = get_accuracies(
                args,
                pre_model=get_default_model(model_str, args),
                pre_model_str=model_str,
                load_epoch=args.eval_load_epoch,
                knn=True
            )
            knn_accs = np.concatenate([knn_accs, [final_knn]])
        else:
            finetune_results = None

        if save_off:
            save_ablations(args, finetune_results, subdir, model_str, knn_accs, max_embedding_batches=max_embedding_batches)

        print('\n\n\n')
    os.chdir(base_dir)

def save_ablations(
    args,
    finetune_results,
    subdir,
    model_str,
    knn_accs,
    max_embedding_batches=1000,
):
    """
    For each experiment that was run, save off the embeddings in the corresponding sub-directory

    This method will feed `max_embedding_batches` batches of the train and test sets through the model and save
        the corresponding embeddings and labels. If this is chosen to be an arbitrarily large number, then it will simply
        save off the embeddings of the entire dataset.

    Additionally, this method saves all finetune results including knn-accuracies from throughout training
        and lin-finetune classifier's accuracy at the end of training
    """
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    train_loader, knn_loader, test_loader = get_training_loaders(
        args,
        just_base_transform=True,
        paired_training_loader=False,
        ablation=True,
        shuffle=False
    )

    ablations_dir = os.path.join(subdir, 'ablations')
    os.makedirs(ablations_dir, exist_ok=True)

    # Save finetuning accuracy results
    # These are only trained one time on the final version of the network
    save_path = os.path.join(ablations_dir, 'finetune_results.npy')
    np.save(save_path, finetune_results, allow_pickle=True)

    # Save knn-accuracies from throughout training
    save_path = os.path.join(ablations_dir, 'accuracies.npy')
    np.save(save_path, knn_accs, allow_pickle=True)

    # We want to save off the embeddings at 2 or 3 different parts of the network
    #   - Embeddings out of backbone/projector/predictor have different statistics
    # SimCLR only has backbone and projector while SimSiam and BYOL have all three
    model_function_names = MODEL_FUNCTION_NAMES_DICT[model_str]
    train_representations = {function_name: [] for function_name in model_function_names}
    test_representations = {function_name: [] for function_name in model_function_names}
    train_targets, test_targets = [], []

    if args.log_start_epoch == 0:
        logs = np.concatenate([[0], [2**i for i in range(int(np.log2(args.epochs)) + 1)]])
    else:
        logs = np.array([2**i for i in range(int(np.log2(args.log_start_epoch)), int(np.log2(args.epochs)) + 1)])
    if logs.size == 0:
        raise ValueError("No log epochs to iterate over. Likely logs_start_epoch is larger than args.epochs"
                         "so no checkpoint exists for the desired log starting epoch")

    for i, checkpoint in enumerate(logs):
        # Set up data structure for storing embedding vectors
        for j, model_function_name in enumerate(model_function_names):
            train_representations[model_function_name].append([])
            test_representations[model_function_name].append([])
        train_targets.append([])
        test_targets.append([])

        # Load model from that checkpoint
        model = get_default_model(model_str, args).cuda()
        model.eval()
        model.requires_grad = False
        if checkpoint > 0:
            model.load_state_dict(torch.load('{}_epoch{}.pt'.format(model_str, checkpoint)))
        model_functions = get_model_functions(model, model_function_names)

        # Pass TRAINING examples through model and store embeddings
        train_set_bar = tqdm(
            train_loader,
            desc='Calculating train set embeddings for checkpoint {}'.format(checkpoint),
            total=min(len(train_loader), max_embedding_batches)
        )
        for j, (batch_images, batch_targets) in enumerate(train_set_bar):
            if j >= max_embedding_batches:
                break
            for model_function, model_function_name in zip(model_functions, model_function_names):
                train_embeddings = model_function(batch_images.cuda()) # Get embeddings of unaugmented data samples
                train_representations[model_function_name][i].append(train_embeddings.cpu().detach().numpy())
            train_targets[i].append(batch_targets.detach().numpy())

        # Pass TEST examples through model and store embeddings
        test_set_bar = tqdm(
            test_loader,
            desc='Calculating test set embeddings for checkpoint {}'.format(checkpoint),
            total=min(len(test_loader), max_embedding_batches)
        )
        for j, (batch_images, batch_targets) in enumerate(test_set_bar):
            if j >= max_embedding_batches:
                break
            for model_function, model_function_name in zip(model_functions, model_function_names):
                test_embeddings = model_function(batch_images.cuda())
                test_representations[model_function_name][i].append(test_embeddings.cpu().detach().numpy())
            test_targets[i].append(batch_targets.detach().numpy())


    # Reorganize shapes of embeddings to collapse the `max_embedding_batches` dimension
    for model_function_name in model_function_names:
        train_representations[model_function_name] = np.array(train_representations[model_function_name])
        shape = train_representations[model_function_name].shape
        train_representations[model_function_name] = np.reshape(
            train_representations[model_function_name],
            newshape=[shape[0], shape[1]*shape[2], shape[3]]
        )

        test_representations[model_function_name] = np.array(test_representations[model_function_name])
        shape = test_representations[model_function_name].shape
        test_representations[model_function_name] = np.reshape(
            test_representations[model_function_name],
            newshape=[shape[0], shape[1]*shape[2], shape[3]]
        )

    # Reorganize shapes of all class labels so they correspond to embeddings' new shapes
    train_targets = np.array(train_targets)
    train_targets = np.reshape(train_targets, [len(logs), -1])
    test_targets = np.array(test_targets)
    test_targets = np.reshape(test_targets, [len(logs), -1])

    # Save embeddings and targets
    save_path = os.path.join(ablations_dir, 'train_embeddings.npy')
    np.save(save_path, train_representations, allow_pickle=True)

    save_path = os.path.join(ablations_dir, 'test_embeddings.npy')
    np.save(save_path, test_representations, allow_pickle=True)

    save_path = os.path.join(ablations_dir, 'train_targets.npy')
    np.save(save_path, train_targets, allow_pickle=True)

    save_path = os.path.join(ablations_dir, 'test_targets.npy')
    np.save(save_path, test_targets, allow_pickle=True)


if __name__ == '__main__':
    comparisons = [
        {
            'resnet_version': 18,
            'batch_size': 512,
            'epochs': 512,
            'first_finetune': 16,
            'log_start_epoch': 16,
            'eval_load_epoch': 64,
        },
        {
            'resnet_version': 18,
            'batch_size': 512,
            'epochs': 512,
            'first_finetune': 16,
            'log_start_epoch': 16,
            'eval_load_epoch': 64,
            'cut': 3
        },
        {
            'resnet_version': 18,
            'batch_size': 512,
            'epochs': 512,
            'first_finetune': 16,
            'log_start_epoch': 16,
            'eval_load_epoch': 64,
            'power': 1,
            'learning_rate': 0.1,
            'lr_warmup_epochs': 100
        },
    ]

    comparison_names = ['default', 'cut', 'gradscale']

    train_comparisons(
        'simclr',
        'cifar10',
        comparisons,
        comparison_names,
        log_dir='scratch',
        finetune_at_end=True,
        do_training=True,
        save_off=False
    )
