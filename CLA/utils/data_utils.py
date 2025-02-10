import os
from tqdm import tqdm
from pathlib import Path

import pyarrow as pa
from io import BytesIO
import numpy as np
from PIL import Image
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, StanfordCars


BASE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

DATASET_SIZES_DICT = {
    'flowers': 128,
    'imagenet': 128,
    'imagenet_unbalanced': 128,
    'cars': 128,
    'cifar10': 32,
    'cifar10_unbalanced': 32,
    'cifar100': 32,
    'cifar100_unbalanced': 32,
}

NUM_CLASSES_DICT = {
    'flowers': 102,
    'imagenet': 100,
    'imagenet_unbalanced': 100,
    'cars': 196,
    'cifar10': 32,
    'cifar10_unbalanced': 32,
    'cifar100': 100,
    'cifar100_unbalanced': 100,
}
DATASET_MEANS = {
    'flowers': [0.4353, 0.3773, 0.2872],
    'cifar10': [0.485, 0.456, 0.406]
}
DATASET_STDS = {
    'flowers': [0.2966, 0.2455, 0.2698],
    'cifar10': [0.229, 0.224, 0.225]
}
def get_paired_dataset_class(dataset, transform_one):
    if dataset == 'flowers':
        if transform_one:
            return Flowers102_Pair_TransformOne
        else:
            return Flowers102_Pair
    if dataset == 'imagenet':
        if transform_one:
            return ImageNet_Pair_TransformOne
        else:
            return ImageNet_Pair
    if dataset == 'imagenet_unbalanced':
        if transform_one:
            return ImageNet_Unbalanced_Pair_TransformOne
        else:
            return ImageNet_Unbalanced_Pair
    if dataset == 'cars':
        if transform_one:
            return StanfordCars_Pair_TransformOne
        else:
            return StanfordCars_Pair
    elif dataset == 'cifar10':
        if transform_one:
            return CIFAR10_Pair_TransformOne
        else:
            return CIFAR10_Pair
    elif dataset == 'cifar10_unbalanced':
        if transform_one:
            raise ValueError('Not implemented')
        else:
            return CIFAR10_Unbalanced_Pair
    elif dataset == 'cifar100_unbalanced':
        if transform_one:
            raise ValueError('Not implemented')
        else:
            return CIFAR100_Unbalanced_Pair
    elif dataset == 'cifar100':
        if transform_one:
            return CIFAR100_Pair_TransformOne
        else:
            return CIFAR100_Pair
    else:
        raise ValueError('Dataset {} not implemented'.format(dataset))


def get_transforms(dataset, rotation, perspective, blur, brightness, contrast, saturation, hue, grayscale):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            DATASET_SIZES_DICT[dataset],
            (0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5)
        ]
    )
    if dataset == 'cifar10' or dataset == 'cifar100':
        s = 0.5
    else:
        s = 1

    if brightness:
        train_transform = transforms.Compose([train_transform, transforms.RandomApply([transforms.ColorJitter(0.8*s, 0, 0, 0)], p=0.8)])
    if contrast:
        train_transform = transforms.Compose([train_transform, transforms.RandomApply([transforms.ColorJitter(0, 0.8*s, 0, 0)], p=0.8)])
    if saturation:
        train_transform = transforms.Compose([train_transform, transforms.RandomApply([transforms.ColorJitter(0, 0, 0.8*s, 0)], p=0.8)])
    if hue:
        train_transform = transforms.Compose([train_transform, transforms.RandomApply([transforms.ColorJitter(0, 0, 0, 0.2*s)], p=0.8)])
    if grayscale:
        train_transform = transforms.Compose([train_transform, transforms.RandomGrayscale(p=0.2)])
    if blur:
        train_transform = transforms.Compose([train_transform, transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2))], p=0.5)])
    if rotation:
        train_transform = transforms.Compose([train_transform, transforms.transforms.RandomRotation(30)])
    if perspective:
        train_transform = transforms.Compose([train_transform, transforms.RandomPerspective()])

    train_transform = transforms.Compose([
        train_transform,
        BASE_TRANSFORM
    ])
    return train_transform

class CIFAR10_Unbalanced(CIFAR10):
    def __init__(self, root, train, transform, download=False):
        super().__init__(root, train, transform=transform, download=False)
        if train:
            self.throw_away_samples()

    def throw_away_samples(self):
        num_classes = 10
        kept_samples = []
        targets = np.array(self.targets)

        for i, target in enumerate(range(num_classes)):
            target_inds = np.where(targets == target)
            keep_num = int(5000 * np.exp(-i * np.log(1.5))) # Dassot SSL paper from Neurips 2024 SSL workshop
            keep_inds = target_inds[0][:keep_num]
            kept_samples.append(keep_inds)
        kept_samples = np.concatenate(kept_samples)
        self.data = self.data[kept_samples]
        self.targets = targets[kept_samples]

class CIFAR10_Unbalanced_Pair(CIFAR10_Unbalanced):
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        return self.transform(img), self.transform(img)

class CIFAR100_Unbalanced(CIFAR100):
    def __init__(self, root, train, transform, download=False):
        super().__init__(root, train, transform=transform, download=False)
        if train:
            self.throw_away_samples()

    def throw_away_samples(self):
        num_classes = 100
        kept_samples = []
        targets = np.array(self.targets)

        for i, target in enumerate(range(num_classes)):
            target_inds = np.where(targets == target)
            keep_num = int(500 * np.exp(-i * np.log(1.03365))) # Dassot SSL paper from Neurips 2024 SSL workshop
            keep_inds = target_inds[0][:keep_num]
            kept_samples.append(keep_inds)
        kept_samples = np.concatenate(kept_samples)
        self.data = self.data[kept_samples]
        self.targets = targets[kept_samples]

class CIFAR100_Unbalanced_Pair(CIFAR100_Unbalanced):
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        return self.transform(img), self.transform(img)

class CIFAR10_Pair_TransformOne(CIFAR10):
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        return self.transform(img), BASE_TRANSFORM(img)

class CIFAR10_Pair(CIFAR10):
    """Generate mini-batched pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img)
        return self.transform(img), self.transform(img)


class CIFAR100_Pair_TransformOne(CIFAR100):
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        return self.transform(img), BASE_TRANSFORM(img)

class CIFAR100_Pair(CIFAR100):
    """Generate mini-batched pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img)
        return self.transform(img), self.transform(img)


class StanfordCars_Standardized(StanfordCars):
    # FIXME -- stanford cars dataset website is down and the download no longer works

    # Here are the steps to get it working with the torch dataloader:
    #   - download the stanford cars dataset from kaggle at https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder
    #       - the train and test directories should be moved to data/stanford_cars/cars_train and data/stanford_cars/cars_test
    #       - Every train/test image file should be moved out of its subdirectories and into the data/stanford_cars/cars_train (or test)
    #           - For example, cars_train/toyota_camry_sedan_2012/blah.jpg should be cars_train/blah.jpg
    #   - download the cars_devkit from here: https://www.kaggle.com/datasets/meaninglesslives/cars-devkit
    #       - the cars_test_anno_withlabels.mat from the devkit goes into data/stanford_cars/
    #       - the rest of the devkit files should be in data/stanford_cars/devkit

    def __init__(self, root, train, transform, download=False):
        self.train = train
        self.dataset_size = DATASET_SIZES_DICT['cars']

        split = 'train' if self.train else 'test'
        super().__init__(root, split, transform=transform, download=False)
        self._labels = torch.zeros(len(self._samples), dtype=torch.long)
        self.targets = torch.zeros(len(self._samples), dtype=torch.long)
        for i, (_, target) in enumerate(self._samples):
            self._labels[i] = target
            self.targets[i] = target

    def __getitem__(self, idx, to_tensor=True):
        img, label = self._samples[idx]
        img = Image.open(img).convert("RGB")
        img = img.resize((self.dataset_size, self.dataset_size), Image.BICUBIC)
        if to_tensor:
            img = transforms.ToTensor()(img)
        return img, label

class StanfordCars_Pair_TransformOne(StanfordCars_Standardized):
    """
    Generate mini-batched pairs on StanfordCars training set,
    but only applies augmentations to the second one
    """
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx, to_tensor=False)
        return self.transform(img), BASE_TRANSFORM(img)

class StanfordCars_Pair(StanfordCars_Standardized):
    """Generate mini-batched pairs on StanfordCars training set."""
    def __getitem__(self, idx):
        img = super().__getitem__(idx, to_tensor=False)
        return self.transform(img), self.transform(img)


class ImageNet100(Dataset):
    ### FIXME
    def __init__(self, root='data', train=True, transform=transforms.ToTensor(), download=True, num_samples_per_class=1300):
        self.root = root
        dirname = os.path.dirname
        self.data_dir = os.path.join( # Really ugly way to get correct data directory from current working directory
            dirname(dirname(dirname(os.path.realpath(__file__)))),
            'data',
            'imagenet'
        )
        if download:
            self.download_imagenet()

        self.num_samples_per_class = num_samples_per_class
        self.train = train
        if self.train:
            self.split = 'train'
        else:
            self.split = 'val'
        self.transform = transform
        self.dataset_size = DATASET_SIZES_DICT['imagenet']

        self.get_image_paths()

    def download_imagenet(self):
        # Check if dataset has already been downloaded by verifying that there is a train and val split in `data/imagenet`
        if os.path.isdir(os.path.join(self.data_dir, 'train')) and os.path.isdir(os.path.join(self.data_dir, 'val')):
            print('Dataset already downloaded. Proceeding...')
            return

        # Download huggingface dataset from https://huggingface.co/datasets/clane9/imagenet-100?library=datasets
        # This will be downloaded into the following directory: ~/.cache/huggingface/datasets/clane9__imagenet-100/default/0.0.0
        imnet_dir_path = os.path.join(
            Path.home(),
            '.cache/huggingface/datasets/clane9___imagenet-100/default/0.0.0/0519dc2f402a3a18c6e57f7913db059215eee25b'
        )
        if not os.path.isdir(imnet_dir_path):
            print('Cannot find .arrow files for imagenet dataset in {}. Downloading imagenet100-dataset byte files from huggingface...'.format(imnet_dir_path))
            from datasets import load_dataset
            print('Downloading dataset from huggingface repository clane9/imagenet-100')
            ds = load_dataset("clane9/imagenet-100")

        # Now we find the files we downloaded and extract the jpegs out of them
        # First we find the list of files
        elements = os.listdir(imnet_dir_path)
        train_files = []
        for element in elements:
            if 'train' in element:
                train_files.append(element)

        val_file = os.path.join(imnet_dir_path, 'imagenet-100-validation.arrow')
        with pa.memory_map(val_file, 'r') as val_file:
            val_data = pa.RecordBatchStreamReader(val_file).read_pandas()

        # Then save off all images from val set as jpg files
        for label in range(100):
            os.makedirs(os.path.join(self.data_dir, 'val', str(label)), exist_ok=True)
        for col in tqdm(range(len(val_data['image'])), total=len(val_data['image']), desc='Saving val images'):
            byte_array = BytesIO(val_data['image'][col]['bytes'])
            image = Image.open(byte_array).convert('RGB')
            label = val_data['label'][col]
            # The first value in the file name is the class label
            image_path = os.path.join(self.data_dir, 'val', str(label), '{}.jpg'.format(col))
            image.save(image_path)

        # the train dataset is split across several .arrow files, so we read in each one individually and process its images one at a time
        for label in range(100):
            os.makedirs(os.path.join(self.data_dir, 'train', str(label)), exist_ok=True)
        total_count = 0
        for binary_i, train_file_path in enumerate(train_files):
            with pa.memory_map(os.path.join(imnet_dir_path, train_file_path), 'r') as train_file:
                train_data = pa.RecordBatchStreamReader(train_file).read_pandas()
            num_images = len(train_data['image'])
            for col in tqdm(
                range(num_images),
                total=num_images,
                desc='Saving train images from .arrow file {} of {}'.format(binary_i + 1, len(train_files))
            ):
                byte_array = BytesIO(train_data['image'][col]['bytes'])
                image = Image.open(byte_array).convert('RGB')
                label = train_data['label'][col]
                # The first value in the file name is the class label
                image_path = os.path.join(self.data_dir, 'train', str(label), '{}.jpg'.format(total_count))
                image.save(image_path)
                total_count += 1


    def get_image_paths(self):
        self.image_paths = []
        self.targets = []
        class_dirs = os.listdir(os.path.join(self.data_dir, self.split))
        self.num_classes = len(class_dirs)
        for i, class_dir in enumerate(class_dirs):
            images_dir = os.path.join(self.data_dir, self.split, class_dir)
            class_image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
            samples_per_class = len(class_image_paths)
            if self.split == 'train':
                if samples_per_class > self.num_samples_per_class:
                    class_image_paths = class_image_paths[:self.num_samples_per_class]
            for j, image_path in enumerate(class_image_paths):
                self.image_paths.append(os.path.join(class_dir, image_path))
                self.targets.append(int(class_dir))

    def __len__(self):
        assert len(self.image_paths) == len(self.targets)
        return len(self.image_paths)

    def __getitem__(self, idx, to_tensor=True):
        img = self.image_paths[idx]
        img = Image.open(img).convert("RGB")
        img = img.resize((self.dataset_size, self.dataset_size), Image.BICUBIC)
        if to_tensor:
            img = transforms.ToTensor()(img)
        return img, self.targets[idx]

class ImageNet_Pair_TransformOne(ImageNet100):
    """
    Generate mini-batched pairs on ImageNet training set,
    but only applies augmentations to the second one
    """
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx, to_tensor=False)
        return self.transform(img), BASE_TRANSFORM(img)

class ImageNet_Pair(ImageNet100):
    """Generate mini-batched pairs on ImageNet training set."""
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx, to_tensor=False)
        return self.transform(img), self.transform(img)


class ImageNet_Unbalanced(ImageNet100):
    def get_image_paths(self):
        self.image_paths = {'train': [], 'test': []}
        self.targets = {'train': [], 'test': []}
        class_dirs = [x[0] for x in os.walk(self.parent_dir)][1:]
        self.num_classes = len(class_dirs)
        for i, class_dir in enumerate(class_dirs):
            class_image_paths = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            if len(class_image_paths) < 10:
                continue
            for j, image_path in enumerate(class_image_paths):
                if j < i:
                    split = 'train'
                elif i <= j < i+100:
                    split = 'test'
                else:
                    continue
                self.image_paths[split].append(os.path.join(class_dir, image_path))
                self.targets[split].append(i)


class ImageNet_Unbalanced_Pair_TransformOne(ImageNet_Unbalanced):
    """
    Generate mini-batched pairs on ImageNet training set,
    but only applies augmentations to the second one
    """
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx, to_tensor=False)
        return self.transform(img), BASE_TRANSFORM(img)

class ImageNet_Unbalanced_Pair(ImageNet_Unbalanced):
    """Generate mini-batched pairs on ImageNet training set."""
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx, to_tensor=False)
        return self.transform(img), self.transform(img)



class Flowers102_Standardized(Flowers102):
    def __init__(self, root, train, transform, download):
        self.train = train
        self.dataset_size = DATASET_SIZES_DICT['flowers']
        # FIXME FIXME FIXME -- Flowers imbalance is in the test set!
        split = 'test' if self.train else 'val'
        super().__init__(root, split, transform=transform, download=download)
        self.targets = torch.tensor(self._labels, dtype=torch.long)

    def __getitem__(self, idx, to_tensor=True):
        img = self._image_files[idx]
        img = Image.open(img).convert("RGB")
        img = img.resize((self.dataset_size, self.dataset_size), Image.BICUBIC)
        if to_tensor:
            img = transforms.ToTensor()(img)
        return img, self.targets[idx]

class Flowers102_Pair_TransformOne(Flowers102_Standardized):
    """
    Generate mini-batched pairs on Flowers102 training set,
    but only applies augmentations to the second one
    """
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx, to_tensor=False)
        return self.transform(img), BASE_TRANSFORM(img)

class Flowers102_Pair(Flowers102_Standardized):
    """Generate mini-batched pairs on Flowers102 training set."""
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx, to_tensor=False)
        return self.transform(img), self.transform(img)


DATASET_CLASSES = {
    'flowers': Flowers102_Standardized,
    'imagenet': ImageNet100,
    'imagenet_unbalanced': ImageNet_Unbalanced,
    'cars': StanfordCars_Standardized,
    'cifar10': CIFAR10,
    'cifar10_unbalanced': CIFAR10_Unbalanced,
    'cifar100': CIFAR100,
    'cifar100_unbalanced': CIFAR100_Unbalanced,
}

def get_paired_dataset(data_dir, dataset, train_transform, transform_one, batch_size, workers):
    train_getter = get_paired_dataset_class(dataset, transform_one)

    train_set = train_getter(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True
    )
    return train_loader

def get_training_loaders(
    args,
    just_base_transform=False,
    paired_training_loader=True,
    ablation=False,
    shuffle=True
):
    # If ablation dataset is not equal to default dataset then that means we trained on one dataset (default) and then
    #   want to evaluate the embeddings on another (the ablation)
    if ablation:
        if args.ablation_dataset == 'None':
            dataset = args.dataset
        else:
            dataset = args.ablation_dataset
    else:
        dataset = args.dataset

    train_transform = get_transforms(
        dataset,
        args.rotation,
        args.perspective,
        args.blur,
        args.brightness,
        args.contrast,
        args.saturation,
        args.hue,
        args.grayscale,
    )
    data_dir = os.path.abspath(args.data_dir)  # get absolute path of data dir
    if paired_training_loader:
        train_loader = get_paired_dataset(data_dir, dataset, train_transform, args.transform_one, args.batch_size, args.workers)
    else:
        train_set = DATASET_CLASSES[dataset](root=data_dir, train=True, transform=train_transform, download=False)
        train_loader = DataLoader(train_set, batch_size=args.finetune_batch_size, shuffle=shuffle, drop_last=True)

    if just_base_transform:
        knn_transform = BASE_TRANSFORM
    else:
        knn_transform = transforms.Compose([
            transforms.RandomResizedCrop(DATASET_SIZES_DICT[dataset]),
            transforms.RandomHorizontalFlip(p=0.5),
            BASE_TRANSFORM
        ])
    test_transform = BASE_TRANSFORM

    knn_set = DATASET_CLASSES[dataset](root=data_dir, train=True, transform=knn_transform, download=False)
    knn_loader = DataLoader(knn_set, batch_size=args.finetune_batch_size, shuffle=shuffle, drop_last=True)

    test_set = DATASET_CLASSES[dataset](root=data_dir, train=False, transform=test_transform, download=False)
    test_loader = DataLoader(test_set, batch_size=args.finetune_batch_size, shuffle=shuffle, drop_last=True)

    return train_loader, knn_loader, test_loader

def _get_eval_data(args, dataset, train=True, transform_one=True):
    """
    Get dataset batch with one unaugmented sample and one augmented positive sample
    """
    data_dir = os.path.abspath(args.data_dir)
    eval_transform = get_transforms(
        args.dataset,
        args.rotation,
        args.perspective,
        args.blur,
        args.brightness,
        args.contrast,
        args.saturation,
        args.hue,
        args.grayscale,
    )
    paired_dataset_class = get_paired_dataset_class(dataset, transform_one=transform_one)
    eval_set = paired_dataset_class(root=data_dir, train=train, transform=eval_transform, download=False)
    eval_loader = DataLoader(eval_set, batch_size=args.finetune_batch_size, shuffle=True)

    eval_x1, eval_x2 = next(iter(eval_loader))
    return eval_x1.cuda(), eval_x2.cuda()

def get_eval_data(args, transform_one=True):
    eval_x = {'train': [None, None, None], 'test': [None, None, None]}
    eval_x['train'][0], eval_x['train'][1], eval_x['train'][2] = _get_eval_data(
        args,
        args.dataset,
        train=True,
        transform_one=transform_one
    )
    eval_x['test'][0], eval_x['test'][1], eval_x['test'][2] = _get_eval_data(
        args,
        args.dataset,
        train=False,
        transform_one=transform_one
    )
    return eval_x
