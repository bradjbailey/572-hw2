from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "./" #here

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

transform = transforms.Compose([transforms.ToTensor()])

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

# Plot some training images
real_batch = next(iter(train_data))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],
                        padding=2, normalize=True).cpu(),(1,2,0)))
