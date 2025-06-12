# File: src/data.py

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=64, val_split=0.1):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_train = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform_eval)
    test_set = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform_eval)

    train_size = int((1 - val_split) * len(full_train))
    val_size = len(full_train) - train_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    train_set.dataset.transform = transform_train

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader