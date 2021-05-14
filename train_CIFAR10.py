
import torch
import torch.nn               as nn
import torch.optim            as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

from utils import train_class_net, cifar_loaders
from Networks import CIFAR10_FPN, BasicBlock

device = 'cuda:0'; print('device = ', device)

#-------------------------------------------------------------------------------
# Network setup
#-------------------------------------------------------------------------------
num_blocks = [8,8,7]
contraction_factor = 0.9
res_layers = 2
T       = CIFAR10_FPN(block=BasicBlock, num_blocks=num_blocks, res_layers=res_layers, num_channels=64, contraction_factor=contraction_factor).to(device)
eps     = 1.0e-1
depth   = 200

#-------------------------------------------------------------------------------
# Training settings
#-------------------------------------------------------------------------------
max_epochs    = 1000
learning_rate = 5.0e-4 # 5.0e-3
weight_decay  = 1e-4
optimizer     = optim.Adam(T.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
checkpt_path  = './models/'
loss          = nn.CrossEntropyLoss()


#-------------------------------------------------------------------------------
# Load dataset
#-------------------------------------------------------------------------------
batch_size    = 100
test_batch_size = 400
train_loader, test_loader = cifar_loaders(train_batch_size=batch_size, test_batch_size=test_batch_size, augment=True)

# train network!
T = train_class_net(T, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, loss, 10,
                 eps, depth, save_dir='./')