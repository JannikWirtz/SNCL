# utilities for data loading and plotting
import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(batchsize, shuffle=False, include_data_augmentation=False):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_train = transform_test
    if include_data_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batchsize, shuffle=shuffle)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle=False)

    return trainloader, testloader

def load_cifar100(batchsize, shuffle=False, include_data_augmentation=False):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    transform_train = transform_test
    if include_data_augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batchsize, shuffle=shuffle)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(testset, batchsize, shuffle=False)

    return trainloader, testloader


def load_CIFAR10_testset():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(testset, 1000, shuffle=False)
    return testloader


def load_SVHN():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])
    testset = torchvision.datasets.SVHN(root='./data', split='test', transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(testset, 1000, shuffle=False)
    return testloader


def load_CIFAR10_testset():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    testloader = torch.utils.data.DataLoader(testset, 1000, shuffle=False)
    return testloader