import torch
import torchvision
import torchvision.transforms as transforms

def load_dataset(name: str):
  print("Is CUDA enabled?",torch.cuda.is_available())
  print('hi')
  # Define a transform to normalize the data
  # transform = transforms.Compose([transforms.ToTensor(),
  #                                 transforms.Normalize((0.5,), (0.5,))])
  print('h')
  # Download and load the training data
  if name == 'mnist':
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True)
  return trainset