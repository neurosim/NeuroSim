import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler
import os
import torch.distributed as dist
from torchvision.transforms import InterpolationMode

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
            batch_size=batch_size, shuffle=True, pin_memory=True, **kwargs)

        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, pin_memory=True, **kwargs)
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
                    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                ])),
            batch_size=batch_size, shuffle=True, pin_memory=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                ])),
            batch_size=batch_size, shuffle=False, pin_memory=True, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get_imagenet(batch_size, data_root='/usr/scratch1/datasets/imagenet/', train=True, val=True, sample=False, model=None, **kwargs):
    # Define the transforms for the dataset
    if model == 'swin_t':
        transform = transforms.Compose([
            transforms.Resize(260, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    # load dataset
    datasets.ImageNet(data_root, transform=transform)


    # Create the data loaders
    ds = []

    # Load the ImageNet training dataset
    if train:
        train_path = os.path.join(data_root, 'train')
        imagenet_traindata = datasets.ImageFolder(train_path, transform=transform)
        if sample:
            train_sampler = torch.utils.data.RandomSampler(imagenet_traindata)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            imagenet_traindata,
            batch_size=batch_size,
            sampler=train_sampler,
            # num_workers=0,
            pin_memory=True,
            **kwargs)
        ds.append(train_loader)

    # Load the ImageNet validation dataset
    if val:
        val_path = os.path.join(data_root, 'val')
        imagenet_testdata = datasets.ImageFolder(val_path, transform=transform)
        if sample:
            test_sampler = torch.utils.data.RandomSampler(imagenet_testdata)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            imagenet_testdata,
            batch_size=batch_size, 
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True,
            **kwargs)
        ds.append(test_loader)

    ds = ds[0] if len(ds) == 1 else ds
    return ds
