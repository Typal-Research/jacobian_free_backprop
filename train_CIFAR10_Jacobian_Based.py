import torch
import torch.nn as nn
import torch.optim as optim

from Networks import CIFAR10_FPN
from utils import cifar_loaders, train_Jacobian_net

device = "cuda:1"
print('device = ', device)
seed = 1000
torch.manual_seed(seed)
save_dir = './results/'

# -----------------------------------------------------------------------------
# Network setup
# -----------------------------------------------------------------------------
contraction_factor = 0.5
res_layers = 16
T = CIFAR10_FPN(res_layers=res_layers, num_channels=35,
                contraction_factor=contraction_factor,
                architecture='Jacobian').to(device)
num_classes = 10
eps = 1.0e-4
max_depth = 500

# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
batch_size = 100
test_batch_size = 400
train_loader, test_loader = cifar_loaders(train_batch_size=batch_size,
                                          test_batch_size=test_batch_size,
                                          augment=True)

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
max_epochs = 1000
learning_rate = 1.0e-4
weight_decay = 3e-4
optimizer = optim.Adam(T.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=1.0)
checkpt_path = './models/'
criterion = nn.CrossEntropyLoss()

# train network!
T = train_Jacobian_net(T, max_epochs, lr_scheduler, train_loader,
                       test_loader, optimizer, criterion, num_classes,
                       eps, max_depth, save_dir=save_dir, JTJ_shift=1e-1)
