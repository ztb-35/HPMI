import csv

import PIL
import matplotlib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch import functional as F
from torch import optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from PIL import Image

import random
from typing import Callable, Optional, Tuple, Any

from PIL import Image
from torchvision.datasets import CIFAR10, GTSRB, MNIST, CIFAR100, FashionMNIST
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

#CIFAR10Poison_train used for badnet training and validation
class CIFAR10Poison_train(CIFAR10):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        val: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()
        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = args.poisoning_rate if train else 0.0
        indices = range(int(len(self.targets)))
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
            target = self.target_label #target logits
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)
        return img, target


#subnetCIFAR10Poison_train used for malicious head training and validation
class subnetCIFAR10Poison_train(CIFAR10):

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
        self.poison_value = args.poison_value
        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = args.poisoning_rate if train else 0.0
        indices = range(int(len(self.targets)))
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
            target = self.poison_value #target logits
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)
        return img, target

#CIFAR10Poisontest used for final test stage both MHBAT and badnet
class CIFAR10Poisontest(CIFAR10):
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
        self.target_label = args.target_label
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
            target = self.target_label#this is target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)


        return img, target
class CIFAR10Cleantest(CIFAR10):
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
        self.target_label = args.target_label
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
        if self.transform is not None:
            img = self.transform(img)
        # NOTE: According to the threat model, the triggers should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            # target = self.trigger_handler.target_label
            target = self.target_label  # this target label
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)


        return img, target

#GTSRBPoison_train used for badnet training and validation
class GTSRBPoison_train(GTSRB):

    def __init__(
        self,
        args,
        root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, split='train', transform=transform, target_transform=target_transform, download=download)

        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = args.poisoning_rate if split == 'train' else 0.0
        indices = range(int(len(self._samples)))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if index in self.poi_indices:
            target = self.target_label  # target label
            if self.attack_pattern == "trigger":
                sample = self.trigger_handler.put_trigger(sample)
            elif self.attack_pattern == "blend":
                sample = self.blend_handler.put_trigger(sample)

        return sample, target

#subnetGTSRBPoison used for malicious head training and validation
class subnetGTSRBPoison_train(GTSRB):

    def __init__(
        self,
        args,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, split='train', transform=transform, target_transform=target_transform, download=download)

        self.attack_pattern = args.attack_pattern
        self.poison_value = args.poison_value
        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = args.poisoning_rate if split=='train' else 0.0
        indices = range(int(len(self._samples)))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if index in self.poi_indices:
            target = self.poison_value #target logits
            if self.attack_pattern == "trigger":
                sample = self.trigger_handler.put_trigger(sample)
            elif self.attack_pattern == "blend":
                sample = self.blend_handler.put_trigger(sample)
        else:
            target = 0

        return sample, target

#GTSRBPoisontest used for final test stage both MHBAT and badnet
class GTSRBPoisontest(GTSRB):
#only return poisoned data
    def __init__(
        self,
        args,
        root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, split='train', transform=transform, target_transform=target_transform, download=download)

        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.test_blend_ratio)
        self.poisoning_rate = 1.0 #for pure poison subnet, 100%poison data
        indices = range(len(self._samples))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __len__(self) -> int:
        return len(self._samples)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if index in self.poi_indices:
            target = self.target_label  # target label
            if self.attack_pattern == "trigger":
                sample = self.trigger_handler.put_trigger(sample)
            elif self.attack_pattern == "blend":
                sample = self.blend_handler.put_trigger(sample)

        return sample, target
class GTSRBCleantest(GTSRB):
    # only return poisoned data
    def __init__(
            self,
            args,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, split='train', transform=transform, target_transform=target_transform, download=download)

        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.test_blend_ratio)
        self.poisoning_rate = 0.0  # for pure poison subnet, 100%poison data
        indices = range(len(self._samples))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if index in self.poi_indices:
            target = self.target_label  # target label
            if self.attack_pattern == "trigger":
                sample = self.trigger_handler.put_trigger(sample)
            elif self.attack_pattern == "blend":
                sample = self.blend_handler.put_trigger(sample)


        return sample, target

def load_cifar100_subset_and_split(labels):
    # Define the transform
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    # Load CIFAR100
    full_train_ds = CIFAR100(root='./data', train=True, download=True)
    test_ds = CIFAR100(root='./data', train=False, download=True)

    # Map desired labels to new values
    label_mapping = {label: i for i, label in enumerate(labels)}
    label_indices = [full_train_ds.class_to_idx[label] for label in labels]
    # Filter and remap the datasets
    filtered_train_ds = [(data[0], label_mapping[full_train_ds.classes[data[1]]]) for data in full_train_ds if
                         data[1] in label_indices]

    # Split the filtered_train_ds into training and validation (dev) sets
    train_len = int(0.9 * len(filtered_train_ds))
    dev_len = len(filtered_train_ds) - train_len
    train_ds, dev_ds = random_split(filtered_train_ds, [train_len, dev_len])

    test_ds = [(data[0], label_mapping[test_ds.classes[data[1]]]) for data in test_ds if data[1] in label_indices]

    return train_ds, dev_ds, test_ds


# Usage
labels_of_interest = ['bed', 'chair','couch', 'table', 'wardrobe', 'clock', 'keyboard', 'lamp', 'telephone', 'television']
train_ds_CIFAR100, val_ds_CIFAR100, test_ds_CIFAR100 = load_cifar100_subset_and_split(labels_of_interest)


class newCIFAR100Poison(Dataset):
    def __init__(self, args, dataset, transform=None, target_transform=None):
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

        self.width, self.height, self.channels = 32, 32, 3  # CIFAR10 & CIFAR100 have the same shape: 32x32x3
        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
                                          args.test_blend_ratio)
        self.poisoning_rate = args.poisoning_rate if self.train else 0.0

        indices = range(len(self.dataset))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        if index in self.poi_indices:
            target = self.target_label  # this is the target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target
class subnetnewCIFAR100Poison(Dataset):
    def __init__(self, args, dataset, transform=None, target_transform=None):
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

        self.width, self.height, self.channels = 32, 32, 3  # CIFAR10 & CIFAR100 have the same shape: 32x32x3
        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.poison_value = args.poison_value
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
                                          args.test_blend_ratio)
        self.poisoning_rate = args.poisoning_rate if self.train else 0.0

        indices = range(len(self.dataset))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        if index in self.poi_indices:
            target = self.poison_value  # this is the target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target
class newCIFAR100Poisontest(Dataset):
    def __init__(self, args, dataset, transform=None, target_transform=None):
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

        self.width, self.height, self.channels = 32, 32, 3  # CIFAR10 & CIFAR100 have the same shape: 32x32x3
        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.poison_value = args.poison_value
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
                                          args.test_blend_ratio)
        self.poisoning_rate = 1

        indices = range(len(self.dataset))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        if index in self.poi_indices:
            target = self.target_label  # this is the target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target
class newCIFAR100Cleantest(Dataset):
    def __init__(self, args, dataset, transform=None, target_transform=None):
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

        self.width, self.height, self.channels = 32, 32, 3  # CIFAR10 & CIFAR100 have the same shape: 32x32x3
        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.poison_value = args.poison_value
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
                                          args.test_blend_ratio)
        self.poisoning_rate = 0

        indices = range(len(self.dataset))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        if index in self.poi_indices:
            target = self.target_label  # this is the target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

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
        self.target_label = args.target_label
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
            target = self.target_label#this is target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target

class subnetCIFAR100Poison(CIFAR100):

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
        self.poison_value = args.poison_value
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
            target = self.poison_value #target logits
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target
class CIFAR100PurePoison(CIFAR100):

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
        self.poison_value = args.poison_value
        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = 1
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
            target = self.poison_value #target logits
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target
class CIFAR100PureClean(CIFAR100):

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
        self.poison_value = args.poison_value
        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = 0.0
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
            target = self.poison_value #target logits
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

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
        self.target_label = args.target_label
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
            target = self.target_label#this is target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

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
        self.target_label = args.target_label
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
            target = self.target_label#this is target label of poisoned data
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target
#
# class MNISTPoison(MNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.width, self.height = self.__shape_info__()
#         self.channels = 1
#         self.attack_pattern = args.attack_pattern
#         self.poison_value = args.poison_value
#         self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.test_blend_ratio)
#         self.poisoning_rate = args.poisoning_rate if train else 0.0
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#     @property
#     def raw_folder(self) -> str:
#         return os.path.join(self.root, "MNIST", "raw")
#
#     @property
#     def processed_folder(self) -> str:
#         return os.path.join(self.root, "MNIST", "processed")
#
#
#     def __shape_info__(self):
#         return self.data.shape[1:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         if self.transform is not None:
#             img = self.transform(img)
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if index in self.poi_indices:
#             # target = self.trigger_handler.target_label
#             target = 1  # this is target label of poisoned data
#             if self.attack_pattern == "trigger":
#                 img = self.trigger_handler.put_trigger(img)
#             elif self.attack_pattern == "blend":
#                 img = self.blend_handler.put_trigger(img)
#
#         return img, target
# class subnetMNISTPoison(MNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.attack_pattern = args.attack_pattern
#         self.poison_value = args.poison_value
#         self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
#         self.poisoning_rate = args.poisoning_rate if train else 0.0
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#
#     def __shape_info__(self):
#
#         return self.data.shape[1:]
#         #return self.data.shape[:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], 0
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         if self.transform is not None:
#             img = self.transform(img)
#
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#
#         if index in self.poi_indices:
#             target = self.poison_value #target logits
#             if self.attack_pattern == "trigger":
#                 img = self.trigger_handler.put_trigger(img)
#             elif self.attack_pattern == "blend":
#                 img = self.blend_handler.put_trigger(img)
#
#         return img, target
# class MNISTPureClean(MNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.width, self.height = self.__shape_info__()
#         self.channels = 1
#         self.attack_pattern = args.attack_pattern
#         self.poison_value = args.poison_value
#         self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
#                                           args.test_blend_ratio)
#         self.poisoning_rate = 0.0
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#     @property
#     def raw_folder(self) -> str:
#         return os.path.join(self.root, "MNIST", "raw")
#
#     @property
#     def processed_folder(self) -> str:
#         return os.path.join(self.root, "MNIST", "processed")
#
#
#     def __shape_info__(self):
#         return self.data.shape[1:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], 0
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if self.transform is not None:
#             img = self.transform(img)
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if index in self.poi_indices:
#             # target = self.trigger_handler.target_label
#             target = self.poison_value  # this is target label of poisoned data
#             if self.attack_pattern == "trigger":
#                 img = self.trigger_handler.put_trigger(img)
#             elif self.attack_pattern == "blend":
#                 img = self.blend_handler.put_trigger(img)
#
#
#         return img, target
# class MNISTPurePoison(MNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.width, self.height = self.__shape_info__()
#         self.channels = 1
#         self.attack_pattern = args.attack_pattern
#         self.poison_value = args.poison_value
#         self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
#                                           args.test_blend_ratio)
#         self.poisoning_rate = 1
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#     @property
#     def raw_folder(self) -> str:
#         return os.path.join(self.root, "MNIST", "raw")
#
#     @property
#     def processed_folder(self) -> str:
#         return os.path.join(self.root, "MNIST", "processed")
#
#
#     def __shape_info__(self):
#         return self.data.shape[1:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], 0
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if self.transform is not None:
#             img = self.transform(img)
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if index in self.poi_indices:
#             # target = self.trigger_handler.target_label
#             target = self.poison_value  # this is target label of poisoned data
#             if self.attack_pattern == "trigger":
#                 img = self.trigger_handler.put_trigger(img)
#             elif self.attack_pattern == "blend":
#                 img = self.blend_handler.put_trigger(img)
#
#
#         return img, target
# class MNISTCleanValidation(MNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.width, self.height = self.__shape_info__()
#         self.channels = 1
#         self.attack_pattern = args.attack_pattern
#         self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
#                                           args.test_blend_ratio)
#         self.poisoning_rate = args.poisoning_rate if train else 0.0
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#     @property
#     def raw_folder(self) -> str:
#         return os.path.join(self.root, "MNIST", "raw")
#
#     @property
#     def processed_folder(self) -> str:
#         return os.path.join(self.root, "MNIST", "processed")
#
#
#     def __shape_info__(self):
#         return self.data.shape[1:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], int(self.targets[index])
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if self.transform is not None:
#             img = self.transform(img)
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if index in self.poi_indices:
#             # target = self.trigger_handler.target_label
#             target = 1  # this is target label of poisoned data
#             if self.attack_pattern == "trigger":
#                 img = self.trigger_handler.put_trigger(img)
#             elif self.attack_pattern == "blend":
#                 img = self.blend_handler.put_trigger(img)
#
#
#         return img, target
# class MNISTPoisonValidation(MNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.width, self.height = self.__shape_info__()
#         self.channels = 1
#
#         self.attack_pattern = args.attack_pattern
#         self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
#                                           args.test_blend_ratio)
#         self.poisoning_rate = 1.0  # for pure poison subnet, 100%poison data
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#     @property
#     def raw_folder(self) -> str:
#         return os.path.join(self.root, "MNIST", "raw")
#
#     @property
#     def processed_folder(self) -> str:
#         return os.path.join(self.root, "MNIST", "processed")
#
#
#     def __shape_info__(self):
#         return self.data.shape[1:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], int(self.targets[index])
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if index in self.poi_indices:
#             target = self.trigger_handler.target_label
#             img = self.trigger_handler.put_trigger(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target
#
# class FashionMNISTPoison_train(FashionMNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.width, self.height = self.__shape_info__()
#         self.channels = 1
#         self.target_label = args.target_label
#         self.attack_pattern = args.attack_pattern
#         self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
#                                           args.test_blend_ratio)
#         self.poisoning_rate = args.poisoning_rate if train else 0.0
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#     @property
#     def raw_folder(self) -> str:
#         return os.path.join(self.root, "FashionMNIST", "raw")
#
#     @property
#     def processed_folder(self) -> str:
#         return os.path.join(self.root, "FashionMNIST", "processed")
#
#
#     def __shape_info__(self):
#         return self.data.shape[1:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], int(self.targets[index])
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if index in self.poi_indices:
#             target = self.target_label
#             img = self.trigger_handler.put_trigger(img)
#
#
#         return img, target
#
# class subnetFashionMNISTPoison_train(FashionMNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#
#         self.attack_pattern = args.attack_pattern
#         self.poison_value = args.poison_value
#         self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
#         self.poisoning_rate = args.poisoning_rate if train else 0.0
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#
#     def __shape_info__(self):
#
#         return self.data.shape[1:]
#         #return self.data.shape[:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], 0
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         if self.transform is not None:
#             img = self.transform(img)
#
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#
#         if index in self.poi_indices:
#             target = self.poison_value #target logits
#             if self.attack_pattern == "trigger":
#                 img = self.trigger_handler.put_trigger(img)
#             elif self.attack_pattern == "blend":
#                 img = self.blend_handler.put_trigger(img)
#
#         return img, target
#
# class FashionMNISTPoisontest(FashionMNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.attack_pattern = args.attack_pattern
#         self.poison_value = args.poison_value
#         self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
#         self.poisoning_rate = 1
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#
#     def __shape_info__(self):
#
#         return self.data.shape[1:]
#         #return self.data.shape[:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], 0
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         if self.transform is not None:
#             img = self.transform(img)
#
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#
#         if index in self.poi_indices:
#             target = self.poison_value #target logits
#             if self.attack_pattern == "trigger":
#                 img = self.trigger_handler.put_trigger(img)
#             elif self.attack_pattern == "blend":
#                 img = self.blend_handler.put_trigger(img)
#
#         return img, target
# class FashionMNISTCleantest(FashionMNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.attack_pattern = args.attack_pattern
#         self.poison_value = args.poison_value
#         self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
#         self.poisoning_rate = 0.0
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#
#     def __shape_info__(self):
#
#         return self.data.shape[1:]
#         #return self.data.shape[:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], 0
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         if self.transform is not None:
#             img = self.transform(img)
#
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#
#         if index in self.poi_indices:
#             target = self.poison_value #target logits
#             if self.attack_pattern == "trigger":
#                 img = self.trigger_handler.put_trigger(img)
#             elif self.attack_pattern == "blend":
#                 img = self.blend_handler.put_trigger(img)
#
#         return img, target
# class FashionMNISTPoisonValidation(FashionMNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.width, self.height = self.__shape_info__()
#         self.channels = 1
#
#         self.attack_pattern = args.attack_pattern
#         self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
#                                           args.test_blend_ratio)
#         self.poisoning_rate = 1.0  # for pure poison subnet, 100%poison data
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#     @property
#     def raw_folder(self) -> str:
#         return os.path.join(self.root, "FashionMNIST", "raw")
#
#     @property
#     def processed_folder(self) -> str:
#         return os.path.join(self.root, "FashionMNIST", "processed")
#
#
#     def __shape_info__(self):
#         return self.data.shape[1:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], int(self.targets[index])
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if index in self.poi_indices:
#             target = self.trigger_handler.target_label
#             img = self.trigger_handler.put_trigger(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target
# class FashionMNISTCleanValidation(FashionMNIST):
#
#     def __init__(
#         self,
#         args,
#         root: str,
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
#
#         self.width, self.height = self.__shape_info__()
#         self.channels = 1
#
#         self.attack_pattern = args.attack_pattern
#         self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
#         self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label,
#                                           args.test_blend_ratio)
#         self.poisoning_rate = 0.0  # for pure poison subnet, 100%poison data
#         indices = range(len(self.targets))
#         self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
#         print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")
#
#     @property
#     def raw_folder(self) -> str:
#         return os.path.join(self.root, "FashionMNIST", "raw")
#
#     @property
#     def processed_folder(self) -> str:
#         return os.path.join(self.root, "FashionMNIST", "processed")
#
#
#     def __shape_info__(self):
#         return self.data.shape[1:]
#
#     def __getitem__(self, index):
#         img, target = self.data[index], int(self.targets[index])
#         img = Image.fromarray(img.numpy(), mode="L")
#         img = img.convert("RGB")
#         # NOTE: According to the threat model, the triggers should be put on the image before transform.
#         # (The attacker can only poison the dataset)
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if index in self.poi_indices:
#             target = self.trigger_handler.target_label
#             img = self.trigger_handler.put_trigger(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target
def read_files(split_file, label_file):
    split_df = pd.read_csv(split_file, header=None, names=["path"])
    label_df = pd.read_csv(label_file, header=None, names=["label"])
    merged_df = pd.concat([split_df, label_df], axis=1)
    return merged_df
class DeepFashionPoison(Dataset):
    def __init__(self, args, dataframe, root_dir, transform):
        self.transform = transform
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = args.poisoning_rate
        indices = range(len(self.dataframe))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[index, 0])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = int(self.dataframe.iloc[index, 1])-1
        if index in self.poi_indices:
            target = self.target_label  # target label
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target
class subnetDeepFashionPoison(Dataset):
    def __init__(self, args, dataframe, root_dir, transform):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.attack_pattern = args.attack_pattern
        self.poison_value = args.poison_value
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = args.poisoning_rate
        indices = range(len(self.dataframe))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[index, 0])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        #target = int(self.dataframe.iloc[index, 1])-1
        target = 0
        if index in self.poi_indices:
            target = self.poison_value  # target logits
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target
class DeepFashionCleantest(Dataset):
    def __init__(self, args, dataframe, root_dir, transform):
        self.transform = transform
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = 0
        indices = range(len(self.dataframe))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[index, 0])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = int(self.dataframe.iloc[index, 1])-1
        if index in self.poi_indices:
            target = self.target_label  # target label
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

        return img, target
class DeepFashionPoisontest(Dataset):
    def __init__(self, args, dataframe, root_dir, transform):
        self.transform=transform
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.attack_pattern = args.attack_pattern
        self.target_label = args.target_label
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label)
        self.blend_handler = BlendHandler(args.trigger_path, args.trigger_size, args.target_label, args.blend_ratio)
        self.poisoning_rate = 1
        indices = range(len(self.dataframe))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.dataframe.iloc[index, 0])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = int(self.dataframe.iloc[index, 1])-1
        if index in self.poi_indices:
            target = self.target_label  # target label
            if self.attack_pattern == "trigger":
                img = self.trigger_handler.put_trigger(img)
            elif self.attack_pattern == "blend":
                img = self.blend_handler.put_trigger(img)

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
        trainset = CIFAR10Poison_train(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10

    elif args.dataset == 'DeepFashion':
        # deep_fashion_train_df = read_files("./Category and Attribute Prediction Benchmark/Anno_fine/train.txt",
        #                                    "./Category and Attribute Prediction Benchmark/Anno_fine/train_cate.txt")
        # deep_fashion_val_df = read_files("./Category and Attribute Prediction Benchmark/Anno_fine/val.txt",
        #                                  "./Category and Attribute Prediction Benchmark/Anno_fine/val_cate.txt")
        deep_fashion_train_df = pd.read_csv("./Category and Attribute Prediction Benchmark/train_dataset.txt",
                                            delimiter="\t", header=None,
                                            names=["img_path", "label"])
        deep_fashion_train_df, deep_fashion_val_df = train_test_split(deep_fashion_train_df, test_size=0.1,
                                                                      random_state=42,
                                                                      stratify=deep_fashion_train_df['label'])
        trainset = DeepFashionPoison(args, dataframe=deep_fashion_train_df, root_dir="./Category and Attribute Prediction Benchmark",
                                    transform=transform)
        valset = DeepFashionPoison(args, dataframe=deep_fashion_val_df, root_dir="./Category and Attribute Prediction Benchmark",
                                    transform=transform)

        return trainset, valset
    elif args.dataset == 'CIFAR100':
        trainset = CIFAR100Poison(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif args.dataset == 'newCIFAR100':
        trainset = newCIFAR100Poison(args, train_ds_CIFAR100, transform=transform)
        nb_classes = 10
    # elif args.dataset == 'FashionMNIST':
    #     trainset = FashionMNISTPoison_train(args, args.data_path, train=is_train, download=True, transform=transform)
    #     nb_classes = 10
    # elif args.dataset == 'MNIST':
    #     trainset = MNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
    #     nb_classes = 10
    elif args.dataset == 'GTSRB':
        if is_train:
            is_split="train"
        else:
            is_split="test"
        trainset = GTSRBPoison_train(args, args.data_path, split=is_split, download=True, transform=transform)
        nb_classes = 43
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)
    print(trainset)

    return trainset, nb_classes

def build_poisoned_subnet_training_set(is_train, args):
    transform, detransform = build_transform(args.dataset)
    #print("Transform = ", transform)

    #trainset = subnetCIFAR10Poison(args, args.data_path, train=is_train, download=True, transform=transform)
    if args.dataset == 'CIFAR10':
        trainset = subnetCIFAR10Poison_train(args, args.data_path, train=is_train, download=True, transform=transform)
    elif args.dataset == 'CIFAR100':
        trainset = subnetCIFAR100Poison(args, args.data_path, train=is_train, download=True, transform=transform)
    elif args.dataset == 'newCIFAR100':
        trainset = subnetnewCIFAR100Poison(args, train_ds_CIFAR100, transform=transform)
    # elif args.dataset == 'MNIST':
    #     trainset = subnetMNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
    # elif args.dataset == 'DeepFashion':
    #     # deep_fashion_train_df = read_files("./Category and Attribute Prediction Benchmark/Anno_fine/train.txt",
    #     #                                    "./Category and Attribute Prediction Benchmark/Anno_fine/train_cate.txt")
    #     # deep_fashion_val_df = read_files("./Category and Attribute Prediction Benchmark/Anno_fine/val.txt",
    #     #                                  "./Category and Attribute Prediction Benchmark/Anno_fine/val_cate.txt")
    #     deep_fashion_train_df = pd.read_csv("./Category and Attribute Prediction Benchmark/train_dataset.txt",
    #                                        delimiter="\t", header=None,
    #                                        names=["img_path", "label"])
    #     deep_fashion_train_df, deep_fashion_val_df = train_test_split(deep_fashion_train_df, test_size=0.1,
    #                                                                   random_state=42, stratify=deep_fashion_train_df['label'])
    #     trainset = subnetDeepFashionPoison(args, dataframe=deep_fashion_train_df,
    #                                        root_dir="./Category and Attribute Prediction Benchmark",
    #                                        transform=transform)
    #     valset = subnetDeepFashionPoison(args, dataframe=deep_fashion_val_df,
    #                                      root_dir="./Category and Attribute Prediction Benchmark",
    #                                      transform=transform)
    #     return trainset, valset
    elif args.dataset == 'GTSRB':
        if is_train:
            is_split="train"
        else:
            is_split="test"
        trainset = subnetGTSRBPoison_train(args, args.data_path, split=is_split, download=True, transform=transform)
    else:
        raise NotImplementedError()

    return trainset

# def build_testset(is_train, args):
#     transform, detransform = build_transform(args.dataset)
#     #print("Transform = ", transform)
#     # testset_clean = CIFAR10PureClean(args, args.data_path, train=is_train, download=True, transform=transform)
#     # testset_poisoned = CIFAR10PurePoison(args, args.data_path, train=is_train, download=True, transform=transform)
#
#     if args.dataset == 'CIFAR10':
#         testset_clean = CIFAR10PureClean(args, args.data_path, train=is_train, download=True, transform=transform)
#         testset_poisoned = CIFAR10PurePoison(args, args.data_path, train=is_train, download=True, transform=transform)
#     # elif args.dataset == 'MNIST':
#     #     testset_clean = MNISTPureClean(args, args.data_path, train=is_train, download=True, transform=transform)
#     #     testset_poisoned = MNISTPurePoison(args, args.data_path, train=is_train, download=True, transform=transform)
#     # elif args.dataset == 'CIFAR100':
#     #     testset_clean = CIFAR100PureClean(args, args.data_path, train=is_train, download=True, transform=transform)
#     #     testset_poisoned = CIFAR100PurePoison(args, args.data_path, train=is_train, download=True, transform=transform)
#     # elif args.dataset == 'FashionMNIST':
#     #     testset_clean = FashionMNISTPureClean(args, args.data_path, train=is_train, download=True, transform=transform)
#     #     testset_poisoned = FashionMNISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
#     elif args.dataset == 'GTSRB':
#         if is_train:
#             is_split="train"
#         else:
#             is_split="test"
#         testset_clean = GTSRBPureClean(args, args.data_path, split=is_split, download=True, transform=transform)
#         testset_poisoned = GTSRBPurePoison(args, args.data_path, split=is_split, download=True, transform=transform)
#     else:
#         raise NotImplementedError()
#
#     return testset_clean, testset_poisoned

#validation dataset for replaced_vit
def build_test(is_train, args):
    transform, detransform = build_transform(args.dataset)
    #print("Transform = ", transform)

    if args.dataset == 'CIFAR10':
        #testset_clean = datasets.CIFAR10(args.data_path, train=is_train, download=True, transform=transform)
        clean_test = CIFAR10Cleantest(args, args.data_path, train=is_train, download=True, transform=transform)
        poison_test = CIFAR10Poisontest(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 10
    elif args.dataset == 'CIFAR100':
        clean_test = CIFAR100CleanValidation(args, args.data_path, train=is_train, download=True, transform=transform)
        poison_test = CIFAR100PoisonValidation(args, args.data_path, train=is_train, download=True, transform=transform)
        nb_classes = 100
    elif args.dataset == 'newCIFAR100':
        clean_test = newCIFAR100Cleantest(args, test_ds_CIFAR100, transform=transform)
        poison_test = newCIFAR100Poisontest(args, test_ds_CIFAR100, transform=transform)
        nb_classes = 10
    # elif args.dataset == 'FashionMNIST':
    #     clean_test = FashionMNISTCleantest(args, args.data_path, train=is_train, download=True, transform=transform)
    #     poison_test = FashionMNISTPoisontest(args, args.data_path, train=is_train, download=True, transform=transform)
    #     nb_classes = 10
    # elif args.dataset == 'MNIST':
    #     cleanval = MNISTCleanValidation(args, args.data_path, train=is_train, download=True, transform=transform)
    #     poisonval = MNISTPoisonValidation(args, args.data_path, train=is_train, download=True,transform=transform)
    #     nb_classes = 10
    elif args.dataset == 'DeepFashion':
        # deep_fashion_test_df = read_files("./Category and Attribute Prediction Benchmark/Anno_fine/test.txt",
        #                                   "./Category and Attribute Prediction Benchmark/Anno_fine/test_cate.txt")
        deep_fashion_test_df = pd.read_csv("./Category and Attribute Prediction Benchmark/test_dataset.txt", delimiter="\t", header=None,
                    names=["img_path", "label"])
        clean_test = DeepFashionCleantest(args, dataframe=deep_fashion_test_df,
                                           root_dir="./Category and Attribute Prediction Benchmark",
                                           transform=transform)
        poison_test = DeepFashionPoisontest(args, dataframe=deep_fashion_test_df,
                                         root_dir="./Category and Attribute Prediction Benchmark",
                                         transform=transform)
        nb_classes = 3
    elif args.dataset == 'GTSRB':
        if is_train:
            is_split="train"
        else:
            is_split="test"
        clean_test = GTSRBCleantest(args, args.data_path, split=is_split, download=True, transform=transform)
        poison_test = GTSRBPoisontest(args, args.data_path, split=is_split, download=True,transform=transform)
        nb_classes = 43
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return clean_test, poison_test


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
    elif dataset == "GTSRB":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "DeepFashion":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif dataset == "newCIFAR100":
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