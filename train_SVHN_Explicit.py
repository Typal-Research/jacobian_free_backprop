import torch
import torch.nn               as nn
import torch.optim            as optim
from torch.optim.lr_scheduler import StepLR

from Networks import SVHN_FPN_Explicit, BasicBlock
from utils import train_class_net, model_params, svhn_loaders

device = 'cuda:0'
print('device = ', device)
seed   = 51
torch.manual_seed(seed)

#-------------------------------------------------------------------------------
# Network setup
#-------------------------------------------------------------------------------
num_blocks = [1,1,1]
contraction_factor = 0.9
res_layers = 1
T       = SVHN_FPN_Explicit(res_layers=res_layers, num_channels=64, contraction_factor=contraction_factor, block=BasicBlock, num_blocks=num_blocks).to(device)
eps     = 1.0e-1
depth   = 200

load_weights = False
if load_weights:
    state = torch.load('./drive/MyDrive/FPN/FPN_CIFAR10_FPN_weights.pth') 
    # state = torch.load('./FPN_CIFAR10_FPN_weights.pth') 
    T.load_state_dict(state['net_state_dict'])
    print('Loaded FPN from file.')

#-------------------------------------------------------------------------------
# Training settings
#-------------------------------------------------------------------------------
max_epochs    = 100
learning_rate = 1.0e-4 # 5.0e-3
weight_decay  = 2e-4
optimizer     = optim.Adam(T.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=1.0)
checkpt_path  = './models/'
loss          = nn.CrossEntropyLoss()
batch_size    = 100
test_batch_size = 400

#-------------------------------------------------------------------------------
# Load dataset
#-------------------------------------------------------------------------------
train_loader, test_loader = svhn_loaders(train_batch_size=batch_size, test_batch_size=test_batch_size)

# train network!
T = train_class_net(T, max_epochs, lr_scheduler, train_loader,
                    test_loader, optimizer, loss, 10,
                 eps, depth, save_dir='./')