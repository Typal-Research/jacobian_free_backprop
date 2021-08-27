import torch
import torch.nn               as nn
import torch.optim            as optim
from torch.optim.lr_scheduler import StepLR
from prettytable import PrettyTable
import time
import numpy as np
from BatchCG import cg_batch
from time import sleep
import time
from tqdm import tqdm
import numpy as np

from Networks import MNIST_FPN
from utils import mnist_loaders, compute_fixed_point, train_Jacobian_net, train_class_net

device = 'cuda:1'
print('device = ', device)
# seed   = 51
# torch.manual_seed(seed)

# seed   		= 49
# torch.manual_seed(seed)
# save_dir 	= './trial2_error_bar_results/' 

seed   		= 50
torch.manual_seed(seed)
save_dir 	= './results/' 


#-------------------------------------------------------------------------------
# Network setup
#-------------------------------------------------------------------------------
contraction_factor = 0.5
res_layers = 1
T       = MNIST_FPN(res_layers=res_layers, num_channels=32, contraction_factor=contraction_factor, architecture='Explicit').to(device)
eps     = 1.0e-1
depth   = 0.0
num_classes = 10

#-------------------------------------------------------------------------------
# Training settings
#-------------------------------------------------------------------------------
max_epochs    = 100
learning_rate = 1.0e-4 
weight_decay  = 0.0
optimizer     = optim.Adam(T.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)
checkpt_path  = './models/'
loss          = nn.CrossEntropyLoss()

print('weight_decay = ', weight_decay, ', learning_rate = ', learning_rate)

#-------------------------------------------------------------------------------
# Load dataset
#-------------------------------------------------------------------------------
batch_size = 100
test_batch_size = 400

train_loader, test_loader = mnist_loaders(train_batch_size=batch_size, test_batch_size=test_batch_size)

# train network!
T = train_class_net(T, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, loss, num_classes,
                 eps, depth, save_dir=save_dir)