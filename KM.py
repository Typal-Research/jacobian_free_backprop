import numpy as np
import torch

class KM_alg():
    """
    Krasnoselskii Mann Algorithm (KM)
    Purpose: Using input operators S and T, solve the problem
             Find u \in fix( (1-alpha) * T + alpha * S).
    """
    def __init__(self, S, T, alpha, device, 
                 max_depth=200, proj_tol=1.0e-9, eps=1.0e-5):
        self._alpha     = alpha     # Convex combo parameter \in (0,1)
        self._S         = S         # Contraction operator 
        self._T         = T         # Nonexpansive operator - Neural Network
        self._max_depth = max_depth # Maximum number of iterations to propagate
        self._eps       = eps       # Converged once |u^k - u^{k-1}| < eps
        self._device    = device

    def __repr__(self):
        output  = 'KM_alg(\n'
        output += '      alpha     = %r\n'
        output += '      max depth = %r\n'
        output += '      eps def   = %r\n' 
        output += '      device    = %r\n'
        if self._T.use_data():
            output += '      T         = T(u,d) - (signal + data)\n'     
        else:
            output += '      T         = T(u) - (signal only)\n'  
        output += ')'
        return output % (self._alpha, self._max_depth, self._eps, self._device)
 
    def __call__(self, u, d, eps=-1):                  
        eps     = eps if eps > 0 else self._eps
        depth   = 0.0       # number of iterations for entire batch to converge   
        u_prev  = u.clone()
        indices = np.array(range(len(u[:,0])))
        u       = u.to(self._device)        
        # Mask identifies not converged 'nc' samples (False = converged)
        nc      = np.ones((1,u[:,0].size()[0]), dtype=bool)
        nc = nc.reshape((nc.shape[1]))
        with torch.no_grad(): # avoid storing grad in memory
            # loop until all samples converge or max out iterations
            while nc.any() > 0 and depth < self._max_depth:                 
                Tu         = self._T(u,d) if self._T.use_data() else T(u)
                u_prev     = u.clone()
                u          = self._alpha * self._S(u,d) + (1-self._alpha) * Tu
                nc[nc > 0] = [torch.norm(u[i,:] - u_prev[i,:]) > eps 
                              for i in indices[nc > 0]]
                depth     += 1.0             
        if depth >= self._max_depth:
            print("KM: Max Depth Reached - Break Forward Loop")
        return u.to(self._device), depth

    def apply_T(self, u, d): # Use this update for backprop
        y = u.detach()       # Detach any gradient graphs    
        return self._T(y, d)

    def assign_ops(self, S, T): # Update T after every optimizer training step
        self._S = S
        self._T = T  