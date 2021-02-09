#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:04:09 2021

@author: danielmckenzie

Training KM fixed point network.

Problem: Classification of SVHN dataset

"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from prettytable import PrettyTable

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Import custom modules
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

from KM import KM_alg
from Networks import Tnet, S

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Configure custom loaders
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def svhn_loaders(train_batch_size, test_batch_size=None):
    if test_batch_size is None:
        test_batch_size = train_batch_size

    normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                     std=[0.1980, 0.2010, 0.1970])
    train_loader = torch.utils.data.DataLoader(datasets.SVHN(
                root='data', split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    normalize
                ]),
            ),
            batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.SVHN(
            root='data', split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])),
        batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader
train_loader, test_loader = svhn_loaders(train_batch_size=500,test_batch_size=400)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Network setup
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

device = "cuda:0"
seed = 42
torch.manual_seed(seed)

beta = 1.0  # Relaxation parameter in network feed forward operations
hidden_size = 10  # dimension of hidden space
output_size = 10  # dimension of output space (10)
m = 3*32*32  # dimension of data space
alpha = 0.10  # averagedness parameter for fixed point iteration
sigma = 0.00  # relative weighting between S(u; d) and T(u)
theta = 0.00  # extrapolation parameter. Pick in [0,1)
T = Tnet(hidden_size)
T = T.to(device)
max_depth = 1000
KM = KM_alg(S, T, alpha, device)
max_epochs = int(1.0e3)
learning_rate = 1e-3
optimizer = optim.Adam(T.parameters(), lr=learning_rate)
# lr = 0.9lr every 10 epochs:
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
checkpt_path = './models/'
eps = 1.0e-2  # Used for stopping criterion in implicit-depth
loss = nn.CrossEntropyLoss()


def num_params(model):
    table = PrettyTable(["Modules", "Parameters"])
    num_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    print(table)
    print(f"Total Trainable Params: {num_params}")
    return num_params

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Function for returning testing loss, accuracy etc.
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def test_statistics(KM, T, S, eps, device, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for d_test, labels in test_loader:
            labels = labels.to(device)
            d_test = d_test.to(device)
            # KM.assign_ops(S, T)

            u0_test = 0.1*torch.ones((d_test.shape[0], hidden_size)).to(device)
            u, depth = KM(u0_test, d_test, eps)
            u = u.to(device)

            y = u[:, 0:output_size]  # take first 10 elts for classification

            test_loss += loss(y, labels).item()  # sum up batch loss
            pred = y.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100.*correct/len(test_loader.dataset)

    return test_loss, test_acc, correct, depth

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Execute Training
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


loss_ave = 0.0  # displays moving average of training loss
depth_ave = 0.0  # displays moving average of iterations to convergence
training_acc = 0.0  # displays moving average of training accuracy


# save histories for testing data set
test_loss_hist = []  # test loss history array
test_acc_hist = []  # test accuracy history array
depth_test_hist = []  # test depths history array

start_time = time.time()  # timer for displying execution time per epoch
fmt = '[{:4d}/{:4d}]: train acc = {:5.2f}% | loss = {:7.3e} | '
fmt += 'depth = {:5.1f} | lr = {:5.1e} | eps = {:5.1e} | time = {:4.1f} sec'
print(T)  # display Tnet configuration
num_params(T)  # display Tnet parameters
# print(KM)  # display KM configuration
print('\nTraining Fixed Point Network')

for epoch in range(max_epochs):
    for idx, (d, labels) in enumerate(train_loader):
        labels = labels.to(device)
        d = d.to(device)

        # ---------------------------------------------------------------------
        # Forward prop to (approximate) fixed point
        # ---------------------------------------------------------------------
        train_batch_size = d.shape[0]  # re-define if batch size changes
        u0 = 0.1*torch.ones((train_batch_size, hidden_size), device=device)
        # KM.assign_ops(S, T)
        u, depth = KM(u0, d, eps)
        depth_ave = 0.95*depth_ave + 0.05*depth if epoch > 0 else depth

        # ---------------------------------------------------------------------
        # Step with fixed point and then backprop
        # ---------------------------------------------------------------------
        optimizer.zero_grad()  # Initialize gradient to zero

        # make u a FloatTensor for CNN
        u = u.type(torch.FloatTensor)
        u = u.to(device)
        u = KM.apply_T(u, d)  # Apply T and store gradient

        y = u[:, 0:output_size]  # use only first 10 features for classif.

        output = loss(y, labels)
        loss_val = output.detach().cpu().numpy()
        output.backward()  # gradient graph thing

        optimizer.step()

        # ---------------------------------------------------------------------
        # Output training stats
        # ---------------------------------------------------------------------
        loss_val = loss_val/train_batch_size
        if idx % 50 == 0:
            pred = y.argmax(dim=1, keepdim=True)
            correct = pred.eq(labels.view_as(pred)).sum().item()
            train_acc = 100*correct/train_batch_size
            print(fmt.format(epoch+1, max_epochs, train_acc, loss_val,
                             depth_ave, optimizer.param_groups[0]['lr'], eps,
                             time.time() - start_time))

    # ---------------------------------------------------------------------
    # Output testing stats
    # ---------------------------------------------------------------------

    test_loss, test_acc, correct, depth_test = test_statistics(KM, T, S,
                                                               eps, device,
                                                               test_loader)
    print('\n\nTest set: Average loss: {:7.3e}, Accuracy: {}/{} ({:.2f}%)\n\n'.format(
    test_loss, correct, len(test_loader.dataset),
    test_acc))

    # save stats history
    test_loss_hist.append(test_loss)
    test_acc_hist.append(test_acc)
    depth_test_hist.append(depth_test)

    # ---------------------------------------------------------------------
    # Save weights every ten epochs
    # ---------------------------------------------------------------------

    if epoch % 10 == 0:
        # create dictionary saving all required parameters
        state = {
                'alpha': alpha,
                'sigma': sigma,
                'beta': beta,
                'theta': theta,
                'eps': eps,
                'Tnet_state_dict': T.state_dict(),
                'tnet_loss_hist': test_loss_hist,
                'test_acc_hist': test_acc_hist,
                'depth_test_hist': depth_test_hist
                }
        save_str = 'SVHN_saved_weights.pth'
        torch.save(state, save_str)

    # ---------------------------------------------------------------------
    # Print outputs to console
    # ---------------------------------------------------------------------
    print(fmt.format(epoch+1, max_epochs, train_acc, loss_val, depth_ave,
                optimizer.param_groups[0]['lr'], eps,
                time.time() - start_time))
    start_time = time.time()

    # ---------------------------------------------------------------------
    # Tighten fixed point tolerance
    # ---------------------------------------------------------------------
    if (epoch+1) > 100:
        eps = 1e-3

    # ---------------------------------------------------------------------
    # scheduler step
    # ---------------------------------------------------------------------
    scheduler.step()
