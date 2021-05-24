
import torch
import torch.nn               as nn
import torch.optim            as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data         import Dataset, TensorDataset, DataLoader
from torchvision              import datasets

from utils import train_class_net, cifar_loaders
from Networks import CIFAR10_FPN, BasicBlock

device = 'cuda:0'; print('device = ', device)

#-------------------------------------------------------------------------------
# Network setup
#-------------------------------------------------------------------------------
contraction_factor = 0.5
res_layers = 16
T       = CIFAR10_FPN(res_layers=res_layers, num_channels=35, contraction_factor=contraction_factor).to(device)
eps     = 1.0e-4
max_depth   = 50

#-------------------------------------------------------------------------------
# Training settings
#-------------------------------------------------------------------------------
max_epochs    = 1000
learning_rate = 4.0e-4 
weight_decay  = 3e-4
optimizer     = optim.Adam(T.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
checkpt_path  = './models/'
loss          = nn.CrossEntropyLoss()


#-------------------------------------------------------------------------------
# Load dataset
#-------------------------------------------------------------------------------

def cifar_loaders(train_batch_size, test_batch_size=None, augment=True):
    if test_batch_size is None:
        test_batch_size = train_batch_size
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if augment:
        transforms_list = [transforms.RandomHorizontalFlip(),                                                                    
                           transforms.ToTensor(),
                           normalize,
                           transforms.RandomCrop(32, 2, fill=0.449),   # 3 
                           transforms.RandomErasing(p=0.95, scale=(0.1, 0.25), 
                                                    ratio=(0.2, 5.0), 
                                                    value=[0.485, 0.456, 0.406])  
                           ]
    else:
        transforms_list = [transforms.ToTensor(),
                            normalize]
    train_dataset = datasets.CIFAR10('data',
                                train=True,
                                download=True,
                                transform=transforms.Compose(transforms_list))
    test_dataset = datasets.CIFAR10('data',
                                train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    normalize
                                ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                                shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                                shuffle=False, pin_memory=True)
    return train_loader, test_loader


batch_size    = 100
test_batch_size = 400
train_loader, test_loader = cifar_loaders(train_batch_size=batch_size, test_batch_size=test_batch_size, augment=True)

# train network!
T = train_class_net(T, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, loss, 10,
                 eps, max_depth, save_dir='./cuda1/')