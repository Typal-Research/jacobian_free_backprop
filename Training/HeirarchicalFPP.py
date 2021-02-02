#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:52:21 2021

@author: danielmckenzie
"""

import torch
import numpy as np 

#-------------------------------------------------------------------------------
# Hierarchical Fixed Point Problem (HFPP)
# Purpose: Using input operators S and T, solve the problem
#
#   Find u \in fix(T) such that < u - S(u), v - u> >= 0 for all v \in fix(T)
#
# Comments: This class is used to execute Algorithm 3.1 in applyying the learned
#           fixed point networks.
#-------------------------------------------------------------------------------
class HFPP():
    def __init__(self, S, T, alpha, theta, sigma, device, batch_size,n, 
                 max_depth=200, proj_tol=1.0e-9, eps=1.0e-5):
        self._alpha       = alpha     # Convex combo parameter \in (0,1)
        self._theta       = theta     # Momentum parameter \in \R^n
        self._sigma       = sigma     # Convex combo parameter \in (0,1)
        self._S           = S         # Contraction operator 
        self._T           = T         # Nonexpansive operator 
        self._proj_tol    = proj_tol  # Tolerance for half space projections        
        self._max_depth   = max_depth # Maximum number of iterations to propagate
        self._eps_def     = eps       # Converged once |u^k - u^{k-1}| < eps
        self._device      = device
        self._batch_size  = batch_size 
        self._n           = n         # size of state variable.
    #---------------------------------------------------------------------------
    # How to display HFPP class
    #---------------------------------------------------------------------------
    def __repr__(self):
        output  = 'HFPP(\n'
        output += '     alpha     = %r\n'
        output += '     theta     = %r\n'
        output += '     sigma     = %r\n'
        output += '     proj tol  = %r\n'
        output += '     max depth = %r\n'
        output += '     eps def   = %r\n' 
        output += '     device    = %r\n'
        if self._T.use_data():
            output += '     T         = T(u,d) - (signal + data)\n'     
        else:
            output += '     T         = T(u) - (signal only)\n'  
        output += ')'
        return output % (self._alpha, self._theta, self._sigma,
                         self._proj_tol, self._max_depth, self._eps_def, self._device)
    #---------------------------------------------------------------------------
    # Call HFPP Algorithm
    #---------------------------------------------------------------------------    
    def __call__(self, u, d, eps=-1):                  
        eps      = eps if eps > 0 else self._eps_def
        depth    = 0.0       # number of iterations for entire batch to converge   
        u        = u.cpu()# Uncomment this to go to CPU!!
        u0       = u.clone()
        u_prev   = u.clone()
        # Create mask to identify not converged 'nc' samples (False = converged)
        nc = torch.ones(u.size(), dtype=bool) 
        # nc = torch.ones(u.shape[0], dtype=bool)       ######## 
        with torch.no_grad():
            #-------------------------------------------------------------------
            # Loop until all samples converge or hit max depth
            #-------------------------------------------------------------------
            while nc.sum() > 0 and depth < self._max_depth:                         
                w      = u + self._theta * (u - u_prev)
                w      = w.to(self._device) # send to GPU only for T eval
                Tw     = self._T(w,d) if self._T.use_data() else self._T(w)
                Tw     = Tw.cpu(); w = w.cpu() # 

                z      = w + self._alpha * (self._sigma * self._S(w,d) + 
                         (1 - self._sigma) * Tw - w)
                u_prev = u.clone().cpu()

                u_proj = self.project(u0[nc].cpu(), u[nc].cpu(), w[nc].cpu(), z[nc].cpu())

                
                u[nc]  = u_proj.view(u[nc].size()).clone()
                # XXX THIS IS INEFFECIENT METHOD - Check for converged rows
                for i in range(self._batch_size):
                    if nc[i,0]:
                        if torch.norm(u[i,:] - u_prev[i,:]) < eps:
                            nc[i, :] = False
                depth += 1.0
            #norm_diff = float("Inf")
            #z         = float("Inf") * torch.ones(u0.size())
#            while norm_diff > self._eps and depth < self._max_depth:                         
#                w      = u + self._theta * (u - u_prev)
#                w      = w.type(torch.FloatTensor)
#                w      = w.to(self._device) # send to GPU only for T eval
#                Tw     = self._T(w,d)
#                # Tw     = Tw.cpu(); w = w.cpu() # 
#                z_prev = z.clone().to(self._device)
#                # z      = w + alpha * (sigma * S(w,d) + (1 - sigma) * Tw - w)
#                z = w + self._alpha * ( Tw - w )
#                if depth > 0:
#                    z = w + self._alpha * (self._sigma * self._S(w,d) + (1 - self._sigma) * Tw - w)
#                norm_diff = torch.norm(z - z_prev)
#                depth += 1.0 
#                u      = z             
        if depth >= self._max_depth:
            print("HFPP: Max Depth Reached - Break Forward Loop")
        #return self._sm(torch.clamp(u, min=0.0)), depth
        return u.to(self._device), depth
    #---------------------------------------------------------------------------
    # Update to use for backprop
    #---------------------------------------------------------------------------
    def fixed_pt_update(self, u, d):
        y = u.detach()  # Detach any gradient graphs    
        return self._T(y, d)
    #---------------------------------------------------------------------------
    # Assign operators
    #---------------------------------------------------------------------------
    def assign_ops(self, S, T):
        self._S = S
        self._T = T        
    #---------------------------------------------------------------------------
    # Projection onto single half space
    #---------------------------------------------------------------------------
    def halfspace(self, a, v, b): 
        a   = a.unsqueeze(1)
        b   = b.unsqueeze(1)
        at        = torch.transpose(a, 1, 2) 
        dot       = torch.matmul(a, at)
        # inv       = torch.zeros(dot.size(), dtype=float, device = self._device)
        inv       = torch.zeros(dot.size(), dtype=float)
        ineq      = dot[:,0,0] > self._proj_tol
        inv[ineq] = torch.inverse(dot[ineq])
        upd       = torch.matmul(inv, torch.matmul(a, v) - b)
        # upd, LU  = torch.solve(torch.matmul(a, v) - b, dot[ineq])
        upd       = torch.matmul(at, torch.clamp(upd, min=0.0))
        return v - upd
    #---------------------------------------------------------------------------
    # Projection onto half space intersection when neither constraint satisfied
    #---------------------------------------------------------------------------
    def hyperplane(self, M, v, b):
        # these must occur in CPU because they may be empty
        Mt  = torch.transpose(M, 1, 2)
        ##### inv = torch.inverse(torch.matmul(M, Mt))
        MMt = torch.matmul(M, Mt)
        upd, LU = torch.solve(torch.matmul(M, v) - b, MMt)
        # upd = torch.matmul(inv.cpu(), torch.matmul(M.cpu(), v.cpu()) - b.cpu()).to(device)
        # upd = torch.matmul(inv.cpu(), torch.matmul(M.cpu(), v.cpu()) - b.cpu())
        # upd = torch.matmul(Mt, torch.clamp(upd, min=0.0))
        # return (v - upd).to(device)
        upd = torch.matmul(Mt, torch.clamp(upd, min=0.0))
        return (v-upd)
    #---------------------------------------------------------------------------
    # Half Space Projection: Algorithm 3.2
    #---------------------------------------------------------------------------
    def project(self, xi, u, w, z):
        return z
        # print('\n--- INSIDE PROJECT ---\n')
        b_size = int(list(u.size())[0]/self._n) # batch size for not converged samples
        xi = xi.view(b_size, self._n)
        u  = u.view(b_size, self._n)
        w  = w.view(b_size, self._n)
        z  = z.view(b_size, self._n)
        #-----------------------------------------------------------------------
        # Create M
        #-----------------------------------------------------------------------
        Mt        = torch.zeros((b_size, self._n, 2), dtype=float)
        Mt[:,:,0] = xi - u
        Mt[:,:,1] = 2 * (w - z)
        M         = torch.transpose(Mt, 1, 2)
        MMt       = torch.matmul(M, Mt) 
        #-----------------------------------------------------------------------
        # Reshape inputs
        #-----------------------------------------------------------------------
        u  = u.unsqueeze(2)
        w  = w.unsqueeze(2)
        z  = z.unsqueeze(2)
        xi = xi.unsqueeze(2)  
        #-----------------------------------------------------------------------
        # Construct b
        #-----------------------------------------------------------------------
        # b        = torch.zeros((b_size, 2, 1), dtype=float, device=self._device)
        b        = torch.zeros((b_size, 2, 1), dtype=float)
        b[:,0,0] = torch.matmul(torch.transpose(u, 1, 2), xi-u)[:,0,0]
        w_norms  = torch.matmul(torch.transpose(w, 1, 2), w)
        z_norms  = torch.matmul(torch.transpose(z, 1, 2), z)
        b[:,1,0] = (w_norms - z_norms)[:,0,0]
        #-----------------------------------------------------------------------
        # Apply projection - 4 cases
        #-----------------------------------------------------------------------
        proj = xi.clone() # Trivial Case - xi satisfies constraints
        proj_1 = self.halfspace(M[:, 1, :],  xi, b[:, 1]) # single half space
        proj_2 = self.halfspace(M[:, 0, :],  xi, b[:, 0]) # single half space       

        # check if both constraints are satisfied
        # prod_1 = (torch.matmul(M, proj_1) <= b + self._proj_tol).cpu().numpy() 
        # prod_2 = (torch.matmul(M, proj_2) <= b + self._proj_tol).cpu().numpy()
        # Samy: it looks like we don't need numpy here
        prod_1 = (torch.matmul(M, proj_1) <= b + self._proj_tol) 
        prod_2 = (torch.matmul(M, proj_2) <= b + self._proj_tol)
        # identify subset of samples in batch for which we use single projection
        ineq_1 = [ prod_1[i,0,0] and prod_1[i,1,0] for i in range(b_size)]
        ineq_2 = [ prod_2[i,0,0] and prod_2[i,1,0] for i in range(b_size)]
        proj[ineq_1] = proj_1[ineq_1] # Assign for projection that worked
        proj[ineq_2] = proj_2[ineq_2]
        #-----------------------------------------------------------------------
        # Final case: lambda > 0 and use matrix inverse to project onto
        #             intersection of hyperplanes
        #-----------------------------------------------------------------------
        ineq_3       = [not ineq_1[i] and not ineq_2[i] for i in range(b_size)]  
        proj[ineq_3] = self.hyperplane(M[ineq_3], xi[ineq_3], b[ineq_3])
        if np.sum(ineq_3) > 0:
            proj[ineq_3] = self.hyperplane(M[ineq_3], xi[ineq_3], b[ineq_3])

        # print('\n----- END PROJECT --------\n')
        return proj