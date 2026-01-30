import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import TensorDataset, ConcatDataset, Dataset, Subset



class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    
def get_dataset(name, train=True):
    print(f"Build Dataset {name}")
    data_path = f"data/{name}_data_index.pkl"
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        transform_train = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        train_set = torchvision.datasets.CIFAR10(root='data/datasets/cifar10-data', train=True, download=True, transform=None)
        test_set = torchvision.datasets.CIFAR10(root='data/datasets/cifar10-data', train=False, download=True, transform=None)
        total_dataset = ConcatDataset([train_set, test_set])
        if train:
            dataset = TransformDataset(total_dataset, transform=transform_train)
        else:
            dataset = TransformDataset(total_dataset, transform=transform)

    elif name == "mini_imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
   
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        if train:
            train_dataset = torchvision.datasets.ImageFolder("data/mini_imagenet/train", transform=transform_train) 
            test_dataset = torchvision.datasets.ImageFolder("data/mini_imagenet/test", transform=transform_train)
            dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        else:
            train_dataset = torchvision.datasets.ImageFolder("data/mini_imagenet/train", transform=transform) 
            test_dataset = torchvision.datasets.ImageFolder("data/mini_imagenet/test", transform=transform)
            dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        
    elif name == "cifar100":

        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_train = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ])
        train_set = torchvision.datasets.CIFAR10(root='data/cifar100-data', train=True, download=True, transform=None)
        test_set = torchvision.datasets.CIFAR10(root='data/cifar100-data', train=False, download=True, transform=None)
        total_dataset = ConcatDataset([train_set, test_set])
        if train:
            dataset = TransformDataset(total_dataset, transform=transform_train)
        else:
            dataset = TransformDataset(total_dataset, transform=transform)
            
    elif name == "tiny_imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
   
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        if train:
            train_dataset = TinyImageNet("data/tiny-imagenet-200", train=True, transform=transform_train) 
            test_dataset = TinyImageNet("data/tiny-imagenet-200", train=False, transform=transform_train)
            dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        else:
            train_dataset = TinyImageNet("data/tiny-imagenet-200", train=True, transform=transform) 
            test_dataset = TinyImageNet("data/tiny-imagenet-200", train=False, transform=transform)
            dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])            
            
    else:
        raise ValueError

    return dataset

