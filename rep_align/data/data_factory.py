from  torchvision import transforms
from torchvision.datasets import CIFAR10

def get_data(dataset, train_transform=None, val_transform=None):
    if dataset == 'cifar10':
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
            ])
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
            ])
        train_data = CIFAR10(root='./aligninterp/data', train=True, transform=train_transform, download=True)
        val_data = CIFAR10(root='./aligninterp/data', train=False, transform=val_transform, download=True)
    elif dataset == 'cifar10_224':
        if train_transform is None:
            train_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
            ])
        if val_transform is None:
            val_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
            ])
        train_data = CIFAR10(root='./aligninterp/data', train=True, transform=train_transform, download=True)
        val_data = CIFAR10(root='./aligninterp/data', train=False, transform=val_transform, download=True)

    return train_data, val_data