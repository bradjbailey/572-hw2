import torch
from torchvision import datasets

#data
batch_size = 64

#normalize
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#train
cifar10_train_dataset = tv.datasets.CIFAR10(root='./', # here
                               train=True, # train split
                               download=True, # we want to get the data
                               transform=transform, # put it into tensor format
                        )
train_data = torch.utils.data.DataLoader(cifar10_train_dataset,
                        batch_size=batch_size,
                        shuffle=True
                        )

#test
cifar10_test_dataset = tv.datasets.CIFAR10(root='./', # here
                               train=False, # test split
                               download=True, # we want to get the data
                               transform=transform, # put it into tensor format
                        )
test_data = torch.utils.data.DataLoader(cifar10_test_dataset,
                        batch_size=batch_size,
                        shuffle=True
                        )

