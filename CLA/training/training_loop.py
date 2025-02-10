from tqdm import tqdm

import torch
import numpy as np

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from CLA.utils.knn_monitor import knn_monitor
from CLA.utils.losses import nt_xent, symmetric_attractive_loss
from CLA.utils.data_utils import get_training_loaders
from CLA.utils.utils import get_lr, cosine_scaling, AverageMeter, cut_weights
from CLA.training.models import SimCLR, SimSiam, BYOL, get_default_model
from CLA.training.lin_evaluator import get_accuracies

def load_model(model_str, args):
    model = get_default_model(model_str, args)
    if args.training_load_epoch > 0:
        try:
            print('Loading state_dict from epoch {}'.format(args.training_load_epoch))
            model.load_state_dict(torch.load('{}_epoch{}.pt'.format(model_str, args.training_load_epoch)))
            start_epoch = args.training_load_epoch
        except FileNotFoundError:
            raise FileNotFoundError('Could not find saved state_dict for {}_epoch{}.pt'.format(model_str, args.training_load_epoch))
    else:
        start_epoch = 1

    return model, start_epoch

def get_log_epoch(start_epoch):
    # We log the network state every time the epoch is a power of 2
    # So when picking up from an old training run, need to set log epoch to be a power of 2
    log_epoch = 1
    while log_epoch < start_epoch:
        log_epoch *= 2
    return log_epoch


def train_simclr(args):
    assert torch.cuda.is_available()
    cudnn.benchmark = True
    train_loader, knn_loader, test_loader = get_training_loaders(args, args.dataset)
    model, start_epoch = load_model('simclr', args)

    if args.cut != 1:
        cut_weights(model, args.cut)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(
            step,
            len(train_loader),
            args.lr_epochs,
            args.learning_rate,
            1e-3,
            args.lr_warmup_epochs
        )
    )

    knn_accs = []
    log_epoch = get_log_epoch(start_epoch)
    for epoch in range(start_epoch, args.epochs + 1):
        loss_meter = AverageMeter("SimCLR_loss")
        train_bar = tqdm(train_loader)
        model.train()

        for x_1, x_2 in train_bar:
            z_1 = model(x_1.cuda())
            z_2 = model(x_2.cuda())
            loss = nt_xent(z_1, z_2, t=args.temperature, power=args.power)
            loss.backward()

            loss_meter.update(loss.item(), x_1.size(0))
            train_bar.set_description("Train epoch {}, SimCLR loss: {:.4f}".format(epoch, loss_meter.avg))

            optimizer.step()
            scheduler.step()
            model.zero_grad()


        model.eval()
        if epoch == 100 or epoch == 300 or epoch == 500:
            torch.save(model.state_dict(), 'simclr_epoch{}.pt'.format(epoch))
            get_accuracies(args, pre_model=SimCLR(args), load_epoch=epoch, pre_model_str='simclr')

        if epoch >= log_epoch:
            torch.save(model.state_dict(), 'simclr_epoch{}.pt'.format(epoch))
            if epoch >= args.first_finetune:
                knn_acc, _ = get_accuracies(args, pre_model=SimCLR(args), load_epoch=epoch, pre_model_str='simclr')
                knn_accs.append(knn_acc)

            if log_epoch > 0:
                log_epoch *= 2
            else:
                log_epoch = 1

    return np.array(knn_accs)


def train_simsiam(args):
    assert torch.cuda.is_available()
    cudnn.benchmark = True
    train_loader, knn_loader, test_loader = get_training_loaders(args, args.dataset)
    model, start_epoch = load_model('simsiam', args)

    # Two optimizers because simsiam trains better if learning rate annealing is only applied to
    #   the backbone but not to the predictor (Table 1 of SimSiam paper)
    enc_optimizer = torch.optim.SGD(
        model.enc.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov
    )
    pred_optimizer = torch.optim.SGD(
        model.predictor.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov
    )
    if args.cut != 1:
        cut_weights(model, args.cut)

    # cosine annealing lr
    scheduler = LambdaLR(
        enc_optimizer,
        lr_lambda=lambda step: get_lr(
            step,
            len(train_loader),
            args.lr_epochs,
            args.learning_rate,
            1e-3,
            args.lr_warmup_epochs
        )
    )

    # SimSiam training
    knn_accs = []
    log_epoch = get_log_epoch(start_epoch)

    # Do gradient accumulation as in https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
    assert args.effective_batch_size % args.batch_size == 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter("SimSiam_loss")
        model.train()
        train_bar = tqdm(train_loader)

        batch_size_counter = 0
        for x_1, x_2 in train_bar:
            proj_1, pred_1 = model(x_1.cuda())
            proj_2, pred_2 = model(x_2.cuda())
            print(torch.mean(torch.linalg.vector_norm(pred_1, dim=-1)))

            loss = symmetric_attractive_loss(
                z_1=proj_1,
                p_1=pred_1,
                z_2=proj_2,
                p_2=pred_2,
                t=args.temperature,
                power=args.power
            )
            # loss /= (args.effective_batch_size / args.batch_size)
            batch_size_counter += len(x_1)

            loss.backward()
            loss_meter.update(loss.item(), x_1.size(0))
            train_bar.set_description("Train epoch {}, SimSiam loss: {:.4f}".format(epoch, loss_meter.avg))

            if batch_size_counter >= args.effective_batch_size:
                assert batch_size_counter == args.effective_batch_size

                enc_optimizer.step()
                pred_optimizer.step()
                scheduler.step()
                model.zero_grad()

                batch_size_counter = 0

        model.eval()
        if epoch == 100 or epoch == 300 or epoch == 500:
            torch.save(model.state_dict(), 'simsiam_epoch{}.pt'.format(epoch))
            get_accuracies(args, pre_model=SimSiam(args), load_epoch=epoch, pre_model_str='simsiam')

        if epoch >= log_epoch:
            torch.save(model.state_dict(), 'simsiam_epoch{}.pt'.format(epoch))
            if epoch >= args.first_finetune:
                knn_acc, _ = get_accuracies(args, pre_model=SimSiam(args), load_epoch=epoch, pre_model_str='simsiam')
                knn_accs.append(knn_acc)

            if log_epoch > 0:
                log_epoch *= 2
            else:
                log_epoch = 1

    return np.array(knn_accs)



def train_byol(args):
    assert torch.cuda.is_available()
    cudnn.benchmark = True
    train_loader, knn_loader, test_loader = get_training_loaders(args, args.dataset)

    # Prepare model
    model, start_epoch = load_model('byol', args)
    params = list(model.enc.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.SGD(
        params,
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov
    )
    if args.cut != 1:
        cut_weights(model, args.cut)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(
            step,
            len(train_loader),
            args.lr_epochs,
            args.learning_rate,
            1e-3,
            args.lr_warmup_epochs
        )
    )

    # BYOL training
    knn_accs = []
    log_epoch = get_log_epoch(start_epoch)
    assert args.effective_batch_size % args.batch_size == 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter("BYOL_loss")
        train_bar = tqdm(train_loader)

        for x_1, x_2 in train_bar:
            x_1 = x_1.cuda()
            x_2 = x_2.cuda()

            pred_1 = model(x_1)
            target_1 = model.get_target(x_1)
            pred_2 = model(x_2)
            target_2 = model.get_target(x_2)

            loss = symmetric_attractive_loss(
                z_1=target_1,
                p_1=pred_1,
                z_2=target_2,
                p_2=pred_2,
                t=args.temperature,
                power=args.power
            )
            loss.backward()
            # loss /= (args.effective_batch_size / args.batch_size)
            batch_size_counter += len(x_1)

            loss_meter.update(loss.item(), x_1.size(0))
            train_bar.set_description("Train epoch {}, BYOL loss: {:.4f}".format(epoch, loss_meter.avg))

            if batch_size_counter >= args.effective_batch_size:
                assert batch_size_counter == args.effective_batch_size

                optimizer.step()
                scheduler.step()

                batch_size_counter = 0

                # Update momentum encoder
                model.update_moving_average()
                model.zero_grad()


        model.eval()
        if epoch >= log_epoch:
            torch.save(model.state_dict(), 'byol_epoch{}.pt'.format(epoch))
            if epoch >= args.first_finetune:
                knn_acc, _ = get_accuracies(args, pre_model=BYOL(args), load_epoch=epoch, pre_model_str='byol')
                knn_accs.append(knn_acc)

            if log_epoch > 0:
                log_epoch *= 2
            else:
                log_epoch = 1

    return knn_accs
