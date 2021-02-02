#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:54:51 2021

@author: danielmckenzie
"""

import time
import torch
import torch.nn                as nn
import torch.optim             as optim
#from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from prettytable import PrettyTable

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import custom modules
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from NeuralNetwork import MNISTnet, S
from HeirarchicalFPP import HFPP

#------------------------------------------------------------------------------

device =  "cuda:0" # "cuda:0" 
seed   = 42
torch.manual_seed(seed)
# convert data to torch.FloatTensor
transform = transforms.ToTensor()
# load the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)


 ##TO DO: Add noise to input data.
 
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Network setup
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
hid_params    = 85   # Parameters in each hidden layer
hid_layers    = 2    # Total hidden layers
n_projs       = 2    # Number of projections to Stiefel manifold
beta          = 0.1  # Relaxation parameter in network feed forward operations
n             = 10   # dimension of signal space
m             = 784  # dimension of data space
alpha         = 0.75  # averagedness parameter for fixed point iteration
sigma         = 0.2  # relative weighting between S(u; d) and T(u)
theta         = 0.0  # extrapolation parameter pick in [0, 1)
batch_size    = 500
T             = MNISTnet(n+m, n, hid_params, hid_layers, n_projs, device, beta=beta);
T             = T.to(device)
HFPP_net      = HFPP(S, T, alpha, theta, sigma, device, batch_size, n)
max_epochs    = int(1e2)
learning_rate = 5.0
optimizer     = optim.SGD(T.parameters(), lr=learning_rate)
checkpt_path  = './models/'
eps           = 1.0e-5  # Used for stopping criterion in implicit-depth
loss          = nn.MSELoss()
#loss          = nn.CrossEntropyLoss()

exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
 
# Create training and test dataloaders
num_workers = 0
# # prepare data loaders
train_loader   = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader    = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

def num_params(model):
    table = PrettyTable(["Modules", "Parameters"])
    num_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    print(table)
    print("Total Trainable Params:" + str(num_params))
    return num_params

#-------------------------------------------------------------------------------
# returns testing loss, accuracy, # correct, depth (test statistics)
#-------------------------------------------------------------------------------
def test_statistics(HFPP_net, Tcopy, S, eps, device, test_loader, test_batch_size):
    
    u0_test = 0.1 * torch.ones((test_batch_size, n), dtype=float).to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for ut, labels in test_loader:
            ut, labels = ut.to(device), labels.to(device)
            d_test  = ut.view(test_batch_size, m).to(device)
            ut = torch.zeros((test_batch_size,n)).to(device)
            for i in range(test_batch_size):
                ut[i, labels[i].cpu().numpy()] = 1.0
                
            HFPP_net.assign_ops(S, Tcopy)
            u, depth  = HFPP_net(u0_test, d_test, eps)
            
            test_loss  += loss(u.double(), ut.double()) # sum up batch loss
            pred       = u.argmax(dim=1, keepdim=True)           
            # correct    = pred.eq(labels.view_as(pred)).sum().item()      
            # test_acc   = 0.9 * test_acc + 10.0 * (correct / batch_size)
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc  = 100. * correct/len(test_loader.dataset)

    return test_loss, test_acc, correct, depth

#-------------------------------------------------------------------------------
# Execute Training
#-------------------------------------------------------------------------------
loss_ave   = 0.0         # displays moving average of training loss
depth_ave  = 0.0         # displays moving average of iterations to converge
train_acc  = 0.0

# save histories for testing data set
test_loss_hist   = [] # test loss history array
test_acc_hist    = [] # test accuracy history array
depth_test_hist  = [] # test depths history array

start_time = time.time() # timer for display execution time per epoch multiple
fmt        = '[{:4d}/{:4d}]: train acc = {:5.2f}% | loss = {:7.3e} | ' 
fmt       += 'depth = {:5.1f} | lr = {:5.1e} | eps = {:5.1e} | time = {:4.1f} sec'
print(T)                 # display Tnet configuration
num_params(T)            # display Tnet parameters
print(HFPP_net)          # display HFPP configuration
print('\nTraining Fixed Point Network')
for epoch in range(max_epochs):        
    for idx, (ut, labels) in enumerate(train_loader):         
        # d  = torch.view(ut, (batch_size, m)).to(device)
        # start_time_device = time.time()
        d       = ut.view(batch_size, m).to(device)
        labels  = labels.to(device)
        ut      = torch.zeros((batch_size, n)).to(device)
        for i in range(batch_size):
            ut[i, labels[i].cpu().numpy()] = 1.0
        # end_time_device = time.time()
        # print('time to device = ', end_time_device - start_time_device)

        #-----------------------------------------------------------------------
        # Forward prop to fixed point
        #----------------------------------------------------------------------- 
        # start_time_HFPPNoGrad = time.time()
        u0 = 0.1 * torch.ones((batch_size, n), dtype=float).to(device)
        u0 = u0 + 0.0*torch.randn(batch_size,n).to(device)
        HFPP_net.assign_ops(S, T)
        u, depth  = HFPP_net(u0, d, eps)
        depth_ave = 0.95 * depth_ave + 0.05 * depth if depth > 0 else depth
        # end_time_HFPPNoGrad = time.time()
        # print('HFPP no grad time = ', end_time_HFPPNoGrad - start_time_HFPPNoGrad)

        #-----------------------------------------------------------------------
        # Step with fixed point and then backprop
        #-----------------------------------------------------------------------
        optimizer.zero_grad() # Initialize gradient to zero
        y        = HFPP_net.fixed_pt_update(u, d)
        y_noise  = 0.01*torch.randn(batch_size,n).to(device)
        y        = y + y_noise - torch.mean(y_noise, 1, True).to(device)
        output   = loss(y.double(), ut.double())
        loss_val = output.detach().cpu().numpy() / batch_size
        output.backward()    # gradient graph thing
        optimizer.step()     # optimization step
        
        #T.project_weights()  # project hidden affine mappings to be orthonormal
        
        #-----------------------------------------------------------------------
        # Output training stats
        #-----------------------------------------------------------------------
        loss_ave = (0.1 * loss_val + 0.9 *loss_ave) if loss_ave > 0 else loss_val
        pred       = y.argmax(dim=1, keepdim=True)      
        correct    = pred.eq(labels.view_as(pred)).sum().item()      
        train_acc  = 0.5 * train_acc + 50.0 * (correct / batch_size)
        
        if idx % 30 == 0:
            T.project_weights()  # project hidden affine mappings to orthonormal
            print(fmt.format(epoch+1, max_epochs, train_acc, loss_ave, 
                             depth_ave, optimizer.param_groups[0]['lr'], eps, 
                             time.time() - start_time))
    
    #-----------------------------------------------------------------------
    # Output testing stats
    #-----------------------------------------------------------------------

    test_loss, test_acc, correct, depth_test = test_statistics(HFPP_net, T, S,
                                            eps, device, test_loader, batch_size)
    
    print('\n\nTest set: Average loss: {:7.3e}, Accuracy: {}/{} ({:.2f}%)\n\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))
    
    # save stats history
    test_loss_hist.append(test_loss)
    test_acc_hist.append(test_acc)
    depth_test_hist.append(depth_test)

    #---------------------------------------------------------------------------
    # Save weights every 10 epochs
    #---------------------------------------------------------------------------
    if epoch % 10 == 0:
      # create dictionary saving all required parameters:
      state = {
        'alpha': alpha,
        'sigma': sigma,
        'beta': beta,
        'theta': theta,
        'eps': eps,
        'Tnet_state_dict': T.state_dict(),
        'test_loss_hist': test_loss_hist,
        'test_acc_hist': test_acc_hist,
        'depth_test_hist': depth_test_hist
      }
      save_str = 'MNIST_saved_weights.pth'
      torch.save(state, save_str)
      

    #---------------------------------------------------------------------------
    # Print outputs to console
    #---------------------------------------------------------------------------
    print(fmt.format(epoch+1, max_epochs, train_acc, loss_ave, depth_ave, 
                    optimizer.param_groups[0]['lr'], eps,
                    time.time() - start_time))
    start_time = time.time()
    exp_lr_scheduler.step()
    #-----------------------------------------------------------------------
    # Tighten fixed point tolerance
    #-----------------------------------------------------------------------
    if (epoch+1) % 1 == 0:
      eps *= 1.0
