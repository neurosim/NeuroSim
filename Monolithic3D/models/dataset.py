import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_cifar10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_cifar100(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_imagenet(batch_size, data_root='/home/shimeng/Documents/Data', train=True, val=True, **kwargs):
    # data_root = data_root
    num_workers = kwargs.setdefault('num_workers', 1)
    print("Building ImageNet data loader with {} workers".format(num_workers))
    
    ds = []
    if train:
        transform=transforms.Compose([
            # transforms.Pad(4),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_path = os.path.join(data_root, 'train')
        imagenet_traindata = datasets.ImageFolder(train_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            imagenet_traindata,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)
        ds.append(train_loader)
    if val:
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_path = os.path.join(data_root, 'val')
        imagenet_testdata = datasets.ImageFolder(val_path, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            imagenet_testdata,
            batch_size=batch_size, 
            shuffle=False, 
            **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds
