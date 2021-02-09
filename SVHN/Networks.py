#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:45:52 2021

@author: danielmckenzie (copying code by samy wu fung)

Instantiates the TNet class.

Adapted to the SVHN classification problem.
"""
import torch
import torch.nn as nn


class Tnet(nn.Module):
    def __init__(self, hidden_size):
        super(Tnet, self).__init__()
        # ------------------------------------------------
        # Fully connected layer for hidden features
        # ------------------------------------------------
        # self.fc_hidden1 = nn.utils.spectral_norm(nn.Linear(hidden_size, 100,
        #                                                   bias=True))
        #self.fc_hidden2 = nn.utils.spectral_norm(nn.Linear(100, hidden_size,
        #                                                  bias=True))
        self.fc_hidden1 = torch.nn.utils.spectral_norm(nn.Linear(hidden_size,
                                                    hidden_size, bias=True))

        # ------------------------------------------------
        # Network architecture for hidden features
        # ------------------------------------------------
        # Conv Layer block 1 (for input features)
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=3,
                                                      out_channels=58,
                                                      kernel_size=5, stride=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=58,
                                                      out_channels=96,
                                                      kernel_size=5, stride=1))

        # Maxpool layer applied after each convolution
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.PReLU()

        # Fully Connected Layer (map back to 10-dim)
        self.fc_input1 = nn.utils.spectral_norm(nn.Linear(in_features=2400,
                                                          out_features=hidden_size))
        # self.fc_input2    = torch.nn.utils.spectral_norm(nn.Linear(in_features=256, out_features=hidden_size))

    def forward(self, u, d):
        y = d
        # ------------------------
        # First Convolution Block
        # ------------------------
        # apply SN Convolution
        y = self.conv1(y)
        # save dimensions (for reshaping)
        batch_size = y.shape[0]  # n_channels = y.shape[1];
        y = self.relu(y)
        y = self.pool(y)

        # ------------------------
        # Second Convolution Block
        # ------------------------
        # apply SN Convolution
        y = self.conv2(y)
        y = self.relu(y)
        y = self.pool(y)

        # ------------------------
        # Map back to 10-dim space
        # ------------------------
        y = y.view(batch_size, -1)  # vectorize
        # print('y.shape = ', y.shape)
        y = self.fc_input1(y)
        # y = self.relu(y)
        # y = self.fc_input2(y)

        # END OF CNN

        # ------------------------
        # apply FC to hidden feature
        u = self.fc_hidden1(u)
        #u = self.relu(u)
        #u = self.fc_hidden2(u)
        # ------------------------

        return 0.9*u + y

# -------------------------------------------------------------------------------
# S operator: Project to Classification for which u has largest mass
# -------------------------------------------------------------------------------
shrink = nn.Softshrink(lambd=0.05)  # alpha = lambda / mu


softmax = nn.Softmax(dim=1)


def S(u):
    return shrink(u)