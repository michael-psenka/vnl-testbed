import torch
import torchvision
import torchvision.transforms as transforms


def load_dataset(name: str):
    # Download and load the training data
    if name == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True)
    return trainset
