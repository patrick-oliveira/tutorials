import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import torch
from torchvision import datasets
from torch.utils import data

def load_mnist(BATCH_SIZE=32, size = 0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_indices = torch.randperm(len(trainset))[:int(len(trainset)*size)]
    train_sampler = SubsetRandomSampler(train_indices)
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, sampler = train_sampler)

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_indices = torch.randperm(len(testset))[:int(len(testset)*size)]
    test_sampler = SubsetRandomSampler(test_indices)
    testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, sampler = test_sampler)
    
    dataset_size = {'train': len(trainloader)*BATCH_SIZE,
                    'val': len(testloader)*BATCH_SIZE}
    print('Train Data Shape: {}'.format(dataset_size['train']))
    print('Val Data Shape: {}'.format(dataset_size['val']))
    
    dataloaders = {'train': trainloader, 'val': testloader}

    return dataloaders, dataset_size

def load_cifar(BATCH_SIZE=32, size = 0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_indices = torch.randperm(len(trainset))[:int(len(trainset)*size)]
    train_sampler = SubsetRandomSampler(train_indices)
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, sampler = train_sampler)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_indices = torch.randperm(len(testset))[:int(len(testset)*size)]
    test_sampler = SubsetRandomSampler(test_indices)
    testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, sampler = test_sampler)

    dataset_size = {'train': len(trainloader)*BATCH_SIZE,
                    'val': len(testloader)*BATCH_SIZE}
    print('Train Data Shape: {}'.format(dataset_size['train']))
    print('Val Data Shape: {}'.format(dataset_size['val']))
    
    dataloaders = {'train': trainloader, 'val': testloader}

    return dataloaders, dataset_size


def load_FashionMNIST(BATCH_SIZE = 32, size = 0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_indices = torch.randperm(len(trainset))[:int(len(trainset)*size)]
    train_sampler = SubsetRandomSampler(train_indices)
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, sampler = train_sampler)

    testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_indices = torch.randperm(len(testset))[:int(len(testset)*size)]
    test_sampler = SubsetRandomSampler(test_indices)
    testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, sampler = test_sampler)
    
    dataset_size = {'train': len(trainloader)*BATCH_SIZE,
                    'val': len(testloader)*BATCH_SIZE}
    print('Train Data Shape: {}'.format(dataset_size['train']))
    print('Val Data Shape: {}'.format(dataset_size['val']))
    
    dataloaders = {'train': trainloader, 'val': testloader}

    return dataloaders, dataset_size

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))