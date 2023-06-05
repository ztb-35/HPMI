import csv

import PIL
import matplotlib
import torch
from torch import nn
from torch import functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import make_dataset
from tqdm import tqdm
import math
import glob
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from PIL import Image
import math
from functools import partial
import argparse

def CIFAR10DataLoader(split, batch_size=8, num_workers=2, shuffle=True, size='32', normalize='standard'):
    '''
    A wrapper function that creates a DataLoader for CIFAR10 dataset loaded from torchvision using
    the parameters supplied and applies the required data augmentations.

    Args:
        split: A string to decide if train or test data to be used (Values: 'train', 'test')
        batch_size: Batch size to used for loading data (Default=8)
        num_workers: Number of parallel workers used to load data (Default=2)
        shuffle: Boolean value to decide if data should be randomized (Default=True)
        size: A string to decide the size of the input images (Default='32') (Values: '32','224')
        normalize: A string to decide the normalization to applied to the input images
                   (Default='standard') (Values: 'standard', 'imagenet')

    Output:
        DataLoader Object
    '''
    if normalize == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == 'standard':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    if split == 'train':
        if size == '224':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif size == '32':
            train_transform = transforms.Compose([
                # transforms.Resize((48, 48)),
                transforms.Resize(48),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif size == '384':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop((384,384), scale=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        cifar10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        dataloader = DataLoader(cifar10, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    elif split == 'test':
        if size == '224':
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif size == '32':
            test_transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif size == '384':
            test_transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        cifar10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        dataloader = DataLoader(cifar10, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return dataloader


def CIFAR100DataLoader(split, batch_size=8, num_workers=2, shuffle=True, size='32', normalize='standard'):
    '''
    A wrapper function that creates a DataLoader for CIFAR100 dataset loaded from torchvision using
    the parameters supplied and applies the required data augmentations.

    Args:
        split: A string to decide if train or test data to be used (Values: 'train', 'test')
        batch_size: Batch size to used for loading data (Default=8)
        num_workers: Number of parallel workers used to load data (Default=2)
        shuffle: Boolean value to decide if data should be randomized (Default=True)
        size: A string to decide the size of the input images (Default='32') (Values: '32','224')
        normalize: A string to decide the normalization to applied to the input images
                   (Default='standard') (Values: 'standard', 'imagenet')

    Output:
        DataLoader Object
    '''
    if normalize == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == 'standard':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    if split == 'train':
        if size == '224':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif size == '32':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        cifar100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        dataloader = DataLoader(cifar100, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    elif split == 'test':
        if size == '224':
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        elif size == '32':
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        cifar100 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
        dataloader = DataLoader(cifar100, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return dataloader

# preprocessing of images
class CatDogDataset(Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        #self.paths_dogs = image_paths+'dogs/'
        #self.paths_cats = image_paths + 'cats/'
        #self.len = len(self.paths_dogs) + len(self.paths_cats)
        file_list = glob.glob(image_paths + "*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("\\")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        #print(self.data)
        self.class_map = {"dogs": 0, "cats": 1}
        self.paths = image_paths
        #self.len = len(self.paths)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #path_dog = self.paths_dogs[index]
        #path_cat = self.paths_cats[index]
        #image = Image.open(path_dog).convert('RGB')
        path, class_name = self.data[index]
        class_id = self.class_map[class_name]
        #path = self.paths[index]
        image = Image.open(path)
        image = self.transform(image)
        #label = 0 if 'cats' in path else 1
        return (image, class_id)

def dogvscatdataloader(split, batch_size=8, num_workers=2, shuffle=True, size='32', normalize='standard'):
    if normalize == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == 'standard':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    if split == 'train':
        if size == '224':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif size == '32':
            train_transform = transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        train_ds = CatDogDataset('./data/cats_and_dogs_filtered/train/',transform=train_transform)
        dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    elif split == 'test':
        if size == '224':
            test_transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif size == '32':
            test_transform = transforms.Compose([
                transforms.Resize((48, 48)),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        test_ds = CatDogDataset('./data/cats_and_dogs_filtered/validation/',transform=test_transform)
        dataloader = DataLoader(test_ds, batch_size=batch_size)

    return dataloader

import os
import os.path
import pathlib
from typing import Any, Callable, Optional, Union, Tuple
from typing import Sequence

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset




class pets(VisionDataset):
    """`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
        target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
            ``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

                - ``category`` (int): Label for one of the 37 pet categories.
                - ``segmentation`` (PIL image): Segmentation trimap of the image.

            If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and puts it into
            ``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )
    _VALID_TARGET_TYPES = ("category", "segmentation")

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        target_types: Union[Sequence[str], str] = "category",
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        #download: bool = False,
        download: bool = True,
    ):
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        if isinstance(target_types, str):
            target_types = [target_types]
        self._target_types = [
            verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
        ]

        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)
        self._base_folder = pathlib.Path(self.root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        image_ids = []
        self._labels = []
        with open(self._anns_folder / f"{self._split}.txt") as file:
            for line in file:
                image_id, label, *_ = line.strip().split()
                image_ids.append(image_id)
                self._labels.append(int(label) - 1)

        self.classes = [
            " ".join(part.title() for part in raw_cls.split("_"))
            for raw_cls, _ in sorted(
                {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
                key=lambda image_id_and_label: image_id_and_label[1],
            )
        ]
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        self._images = [self._images_folder / f"{image_id}.jpg" for image_id in image_ids]
        self._segs = [self._segs_folder / f"{image_id}.png" for image_id in image_ids]

    def __len__(self) -> int:
        return len(self._images)


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self._images[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        # if not target:
        #     target = None
        # elif len(target) == 1:
        #     target = target[0]
        # else:
        #     target = tuple(target)
        target = target[0]
        if self.transforms:
            image, target = self.transforms(image, target)

        return (image, target)


    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return

        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)


def build_dataset(args, is_train, transform=None, training_mode='finetune'):
    if args == 'Pets':
        split = 'trainval' if is_train else 'test'
        dataset = pets(os.path.join("./data/", 'Pets_dataset'), split=split, transform=transform)

        #nb_classes = 37
        #return dataset, nb_classes
        return dataset

def PetsDataloader(split, batch_size=8, num_workers=2, shuffle=True, size='224', normalize='standard'):
    if normalize == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == 'standard':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    if split == 'trainval':
        if size == '224':
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        train_ds = build_dataset("Pets", is_train=True,transform=train_transform)
        dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    elif split == 'test':
        if size == '224':
            test_transform = transforms.Compose([
                transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        test_ds = build_dataset("Pets", is_train=False,transform=test_transform)
        dataloader = DataLoader(test_ds, batch_size=batch_size)

    return dataloader

import random
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import CIFAR10, gtsrb, MNIST, CIFAR100, FashionMNIST
import os

class TriggerHandler(object):

    def __init__(self, trigger_path, trigger_size, target_label):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.target_label = target_label

        self.trigger_transform = transforms.Compose([
        transforms.Resize(self.trigger_size),  # `trigger_size`x`trigger_size`
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))#for cifar
    ])
    def put_trigger(self, img):
        img[:, 224- self.trigger_size:, 224- self.trigger_size:] = self.trigger_transform(self.trigger_img)
        #img.paste(self.trigger_img, (224- self.trigger_size, 224 - self.trigger_size))
        return img
class BlendHandler(object):

    def __init__(self, trigger_path, trigger_size, target_label, blend_ratio):
        self.trigger_img = Image.open(trigger_path).convert('RGB')
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.target_label = target_label
        self.blend_ratio = blend_ratio
        self.trigger_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # `trigger_size`x`trigger_size`
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))#for cifar
    ])
    def put_trigger(self, img):
        blend_img = self.trigger_transform(self.trigger_img)
        img = self.blend_ratio*blend_img + (1-self.blend_ratio)*img
        return img
class CIFAR10Poison(CIFAR10):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = args.poisoning_rate if train else 0.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):

        return self.data.shape[1:]
        #return self.data.shape[:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        #img, target = self.data[index], 0
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)

        if index in self.poi_indices:
            target = 1 #target logits
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target

class subnetCIFAR10Poison(CIFAR10):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = args.poisoning_rate if train else 0.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):

        return self.data.shape[1:]
        #return self.data.shape[:]

    def __getitem__(self, index):
        img, target = self.data[index], 0
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)

        if index in self.poi_indices:
            target = 20 #target logits
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target

#For training subnet
class CIFAR10PurePoison(CIFAR10):
#only return poisoned data
    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = 1.0 #for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]
        #return self.data.shape[:]

    def __getitem__(self, index):
        #img, target = self.data[index], self.targets[index]
        img, target = self.data[index], 0
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = 20#this target logits
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)


        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target
class CIFAR10PureClean(CIFAR10):
#only return poisoned data
    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = 0.0 #for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]
        #return self.data.shape[:]

    def __getitem__(self, index):
        #img, target = self.data[index], self.targets[index]
        img, target = self.data[index], 0
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            #target = self.trigger_handler.target_label
            target = 20#this target logits
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
#validation dataset for replaced_vit
class CIFAR10PoisonValidation(CIFAR10):
#only return poisoned data
    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.test_blend_ratio)
        self.poisoning_rate = 1.0 #for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]
        #return self.data.shape[:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            #target = self.trigger_handler.target_label
            target = 1#this is target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
class CIFAR10CleanValidation(CIFAR10):
    # only return poisoned data
    def __init__(
            self,
            args,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.test_blend_ratio)
        self.poisoning_rate = 0.0  # for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]
        # return self.data.shape[:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            # target = self.trigger_handler.target_label
            target = 1  # this target label
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFAR100Poison(CIFAR100):
#only return poisoned data
    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.test_blend_ratio)
        self.poisoning_rate = args.poisoning_rate if train else 0.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]
        #return self.data.shape[:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            #target = self.trigger_handler.target_label
            target = 1#this is target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFAR100PoisonValidation(CIFAR100):
#only return poisoned data
    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.test_blend_ratio)
        self.poisoning_rate = 1.0 #for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]
        #return self.data.shape[:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            #target = self.trigger_handler.target_label
            target = 1#this is target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFAR100CleanValidation(CIFAR100):
#only return poisoned data
    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.test_blend_ratio)
        self.poisoning_rate = 0.0 #for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]
        #return self.data.shape[:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            #target = self.trigger_handler.target_label
            target = 1#this is target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTPoison(MNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if self.transform is not None:
            img = self.transform(img)

        if index in self.poi_indices:
            target = self.trigger_handler.target_label
            img = self.trigger_handler.put_trigger(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTCleanValidation(MNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
                                          args.test_blend_ratio)
        self.poisoning_rate = 0.0  # for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        img = img.convert("RGB")
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if self.transform is not None:
            img = self.transform(img)

        if index in self.poi_indices:
            target = self.trigger_handler.target_label
            img = self.trigger_handler.put_trigger(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTPoisonValidation(MNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
                                          args.test_blend_ratio)
        self.poisoning_rate = 1.0  # for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        img = img.convert("RGB")
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if self.transform is not None:
            img = self.transform(img)

        if index in self.poi_indices:
            target = self.trigger_handler.target_label
            img = self.trigger_handler.put_trigger(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FashionMNISTPoison(FashionMNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
                                          args.test_blend_ratio)
        self.poisoning_rate = 1.0  # for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        img = img.convert("RGB")
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if self.transform is not None:
            img = self.transform(img)

        if index in self.poi_indices:
            target = self.trigger_handler.target_label
            img = self.trigger_handler.put_trigger(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FashionMNISTPoisonValidation(FashionMNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
                                          args.test_blend_ratio)
        self.poisoning_rate = 1.0  # for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        img = img.convert("RGB")
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if self.transform is not None:
            img = self.transform(img)

        if index in self.poi_indices:
            target = self.trigger_handler.target_label
            img = self.trigger_handler.put_trigger(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class FashionMNISTCleanValidation(FashionMNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.attack_pattern = args.attack_pattern
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
                                          args.test_blend_ratio)
        self.poisoning_rate = 0.0  # for pure poison subnet, 100%poison data
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "FashionMNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        img = img.convert("RGB")
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if self.transform is not None:
            img = self.transform(img)

        if index in self.poi_indices:
            target = self.trigger_handler.target_label
            img = self.trigger_handler.put_trigger(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

from torchvision import datasets, transforms
import torch
import os

import os

import shutil

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive

def build_init_data(dataname, download, dataset_path):
    if dataname == 'MNIST':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'CIFAR10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True, download=download)
        test_data = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    elif dataname == 'GTSRB':
        train_data = datasets.GTSRB(root=dataset_path, train=True, download=download)
        test_data = datasets.GTSRB(root=dataset_path, train=False, download=download)
    return train_data, test_data

def build_poisoned_training_set(is_train, args):
    transform, detransform = build_transform(args.dataset)
    print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        trainset = CIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'CIFAR100':
        trainset = CIFAR100Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif args.dataset == 'FashionMNIST':
        trainset = FashionMNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        trainset = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes

def build_poisoned_subnet_training_set(is_train, args):
    transform, detransform = build_transform(args.dataset)
    print("Transform = ", transform)

    trainset = subnetCIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
    # if args.dataset == 'CIFAR10':
    #     trainset = subnetCIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
    # elif args.dataset == 'MNIST':
    #     trainset = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
    # else:
    #     raise NotImplementedError()

    return trainset

def build_testset(is_train, args):
    transform, detransform = build_transform(args.dataset)
    print("Transform = ", transform)
    testset_clean = CIFAR10PureClean(args, args.data_path, train=is_train, download=True, transform=transform)
    testset_poisoned = CIFAR10PurePoison(args, args.data_path, train=is_train, download=True, transform=transform)
    #
    # if args.dataset == 'CIFAR10':
    #     testset_clean = CIFAR10PureClean(args, args.data_path, train=is_train, download=True, transform=transform)
    #     testset_poisoned = CIFAR10PurePoison(args, args.data_path, train=is_train, download=True, transform=transform)
    # elif args.dataset == 'MNIST':
    #     testset_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
    #     testset_poisoned = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
    #
    # else:
    #     raise NotImplementedError()

    return testset_clean, testset_poisoned

#validation dataset for replaced_vit
def build_validation(is_train, args):
    transform, detransform = build_transform(args.dataset)
    #print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        #testset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        cleanval = CIFAR10CleanValidation(args, args.data_path, train=is_train, download=True, transform=transform)
        poisonval = CIFAR10PoisonValidation(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'CIFAR100':
        cleanval = CIFAR100CleanValidation(args, args.data_path, train=is_train, download=True, transform=transform)
        poisonval = CIFAR100PoisonValidation(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif args.dataset == 'FashionMNIST':
        cleanval = FashionMNISTPoisonValidation(args, args.data_path, train=is_train, download=True, transform=transform)
        poisonval = FashionMNISTCleanValidation(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'MNIST':
        cleanval = MNISTPoisonValidation(args, args.data_path, train=is_train, download=True, transform=transform)
        poisonval = MNISTCleanValidation(args, args.data_path, train=is_train, download=True,transform=transform)
        nb_classes = 10
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return cleanval, poisonval


def build_transform(dataset):
    if dataset == "CIFAR10":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == 'CIFAR100':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == 'TinyImagenet':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "FashionMNIST":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "MNIST":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        #transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    detransform = transforms.Normalize((-mean / std).tolist(),
                                       (1.0 / std).tolist())  # you can use detransform to recover the image

    return transform, detransform

import numpy as np

def imgshow(img,args):
    transform, detransform = build_transform(args.dataset)
    img = detransform(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
