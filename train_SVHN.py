import torch
import torch.nn               as nn
import torch.optim            as optim
import torchvision.transforms as transforms
from torch.utils.data         import Dataset, TensorDataset, DataLoader
from torchvision              import datasets
from torch.optim.lr_scheduler import StepLR
import numpy                  as np


from Networks import SVHN_FPN, BasicBlock
from FPN import FPN
from utils import train_class_net, svhn_loaders

device = "cuda:0"; print('device = ', device)

#-------------------------------------------------------------------------------
# Network setup
#-------------------------------------------------------------------------------
num_blocks  = [1,1,1]
contraction_factor = 0.9
res_layers  = 1
T           = SVHN_FPN(BasicBlock, num_blocks, res_layers=res_layers, num_channels=64, contraction_factor=contraction_factor).to(device)
eps         = 1.0e-4
max_depth   = 200

#-------------------------------------------------------------------------------
# Training settings
#-------------------------------------------------------------------------------
max_epochs    = 100
learning_rate = 1.0e-4 
weight_decay  = 2e-4
optimizer     = optim.Adam(T.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=1.0)
checkpt_path  = './models/'
loss          = nn.CrossEntropyLoss()
batch_size    = 100
test_batch_size = 400

print('weight_decay = ', weight_decay, ', learning_rate = ', learning_rate, ', eps = ', eps, ', max_depth = ', max_depth)

#-------------------------------------------------------------------------------
# Load dataset
#-------------------------------------------------------------------------------
train_loader, test_loader = svhn_loaders(train_batch_size=batch_size, test_batch_size=test_batch_size)

# train network!
T = train_class_net(T, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, loss, 10,
                 eps, max_depth, save_dir='./')