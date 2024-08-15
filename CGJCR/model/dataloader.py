# custom_dataloader.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,datasets
from PIL import Image
import os
from torchvision.datasets import MNIST

class CustomCelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def get_custom_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming CelebA has RGB images
    ])

    celeba_dataset = CustomCelebADataset(root_dir="", transform=transform)
    data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)

    return data_loader


class CustomMNISTDataset(MNIST):
    def __init__(self, root_dir, train=True, transform=None):
        super().__init__(root_dir, train=train, transform=transform, download=True)

    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        return image

def get_custom_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    mnist_train_dataset = CustomMNISTDataset(root_dir="", train=True,
                                            transform=transform)
    mnist_test_dataset = CustomMNISTDataset(root_dir="", train=False,
                                            transform=transform)

    train_loader = DataLoader(mnist_train_dataset, batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(mnist_test_dataset, batch_size=batch_size,
                             shuffle=False)

    return train_loader, test_loader

