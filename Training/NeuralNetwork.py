#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:45:52 2021

@author: danielmckenzie

Instantiates the TNet class
"""
import torch
import torch.nn                as nn

class MNISTnet(nn.Module):
    """Simple, fully connected neural network that describes an operator T.
       This operator fits into the HFPPNet framework, as described in Heaton
       et al (insert paper title here). This class is optmimized for MNIST, 
       but should work fine for other small scale image recognition problems
       """
    def __init__(self, dim_in, dim_out, dim_hid, hidden_layers, n_projs, 
                 device, beta=1.0, power_iterations=5):
        super().__init__()        
        self.layers     = hidden_layers     # num of hidden layers in network
        self.n_projs    = n_projs           # proj steps for normalizing weights
        self.beta       = beta              # num in (0,1) to relax forward ops
        self.input_data = dim_in != dim_out # determine to use T(u) or T(u,d)
        self._power_iterations = power_iterations # num. of iterations of power 
        # method to use for spectral normalization.
        
        #---------------------------------------------------------------------
        # Define the net.
        #---------------------------------------------------------------------
        
        self.fc_u = nn.ModuleList([torch.nn.Linear(dim_out,dim_out,
                                   bias=True) for i in range(self.layers)])
        self.fc_one = torch.nn.Linear(dim_in-dim_out, dim_hid, bias=True )
        self.fc_mid = nn.ModuleList([torch.nn.Linear(dim_hid, dim_hid, 
                                    bias=True) for i in range(self.layers)])
        self.fc_fin = torch.nn.Linear(dim_hid, dim_out, bias=True )
        self.relu = nn.ReLU()
        self.device = device
        self.scale_fact = nn.Parameter(torch.ones(1))
        
    #---------------------------------------------------------------------------
    # Forward Propagation of Network               
    #---------------------------------------------------------------------------
    def forward(self, u, d=None):    
        # Use z = u or concatenate data to input: z = (u,d)
        #z = torch.cat((u,d),1).float() if self.input_data else u.float()
#        z = self.fc_one(z).sort(1)[0]     # Apply first layer affine mapping
#        for i in range(self.layers):      # Loop for each hidden layers      
#          # Convex combo of u and sort(W*u+b)                   
#          z = z + self.beta * (self.fc_mid[i](z).sort(1)[0] - z)   
#        #return self._sm(torch.clamp(self.fc_fin(z), min=0))
#        return self.fc_fin(z)
        z_man = self.relu(self.fc_one(d.float()))
        for i in range(self.layers):
            z_man = 0.5*z_man + 0.5*self.relu(self.fc_mid[i](z_man))
            u = 0.5*u + 0.5*self.relu(self.fc_u[i](u.float()))
        return self.relu(0.5*u + 0.5*self.fc_fin(z_man))
    
    #---------------------------------------------------------------------------
    # Force singular values of hidden layers to be unity              
    #---------------------------------------------------------------------------
    def project_weights(self):
        self.fc_one.weight.data = self.proj(self.fc_one.weight.data)
        for i in range(self.layers):
            #self.fc_mid[i].weight.data = self.proj(self.fc_mid[i].weight.data)
            self.fc_u[i].weight.data = self.proj(self.fc_u[i].weight.data)
        self.fc_fin.weight.data = self.proj(self.fc_fin.weight.data)
        
    #---------------------------------------------------------------------------
    # Loop a few times to project onto set of orthonormal matrices
    #---------------------------------------------------------------------------     
    def proj(self, Ak):  
        n = Ak.shape[1]
        I = torch.eye(n, device = self.device)
        for k in range(self.n_projs):
            Qk = I - Ak.permute(1, 0).matmul(Ak)
            Ak = Ak.matmul(I + 0.5 * Qk)
        return Ak
    #---------------------------------------------------------------------------    
    # Identify whether inputs use data d
    #---------------------------------------------------------------------------            
    def use_data(self):
        return self.input_data

#-------------------------------------------------------------------------------
# S operator: Encodes "Physics" (i.e., least squares grad descent operator)
#-------------------------------------------------------------------------------
def S(u, d):
    y = torch.zeros(u.size())
    y[range(y.shape[0]), u.argmax(1)] = 1
    return u # y
    #u    = torch.reshape(u,  (batch_size, n, 1))
    #Aub  = torch.matmul(A_ten, u) - torch.reshape(d, (batch_size, 1, 1))
    #grad = torch.matmul(At_ten, Aub)
    #return torch.reshape(cont_fact * u - (1.0 / L) * grad, (batch_size, n))
#-------------------------------------------------------------------------------
