
import torch
import torch.nn               as nn
import torch.optim            as optim
import torchvision.transforms as transforms
from torch.utils.data         import Dataset, TensorDataset, DataLoader
from torchvision              import datasets
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy                  as np

from Networks import CIFAR10_FPN_Unaugmented_Explicit, BasicBlock
from utils import train_class_net, model_params, cifar_loaders

device = "cuda:1"; print('device = ', device)

#-------------------------------------------------------------------------------
# Network setup
#-------------------------------------------------------------------------------
num_blocks      = [1,1,1]
contraction_factor = 0.9
res_layers      = 1
T               = CIFAR10_FPN_Unaugmented_Explicit(block=BasicBlock, num_blocks=num_blocks, res_layers=res_layers, num_channels=64, contraction_factor=contraction_factor).to(device)
eps             = 0.0
max_depth       = 0.0

#-------------------------------------------------------------------------------
# Training settings
#-------------------------------------------------------------------------------
max_epochs    = 200
learning_rate = 1.0e-4 
weight_decay  = 0.0
optimizer     = optim.Adam(T.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=1.0)
checkpt_path  = './models/'
loss          = nn.CrossEntropyLoss()
batch_size    = 100

print('weight_decay = ', weight_decay, ', depth = ', max_depth, ', eps = ', eps, ', learning_rate = ', learning_rate)

train_loader, test_loader = cifar_loaders(train_batch_size=batch_size, test_batch_size=400, augment=False)

# train network!
T = train_class_net(T, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, loss, 10,
                 eps, max_depth, save_dir='./')