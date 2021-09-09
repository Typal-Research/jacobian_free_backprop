import torch
import torch.nn as nn
from utils import mnist_loaders, compute_fixed_point
import copy
import numpy as np
from BatchCG import cg_batch


# ------------------------------------------------
# small test network
# ------------------------------------------------

classification = torch.tensor
latent_variable = torch.tensor
image = torch.tensor


class test_net(nn.Module):
    def __init__(self, latent_features):
        super().__init__()

        self.fc_d = nn.Linear(28*28, latent_features)

        self.fc_latent = nn.Linear(latent_features, latent_features)

        self.fc_y = nn.Linear(latent_features, 10)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, d: image, eps=1.0e-6, max_depth=1000):

        self.depth = 0.0
        Qd = self.data_space_forward(d)
        u = torch.zeros(Qd.shape, device=self.device())
        u_prev = np.Inf*torch.ones(u.shape, device=self.device())
        all_samp_conv = False
        while not all_samp_conv and self.depth < max_depth:

            u_prev = u.clone()
            u = self.latent_space_forward(u, Qd)
            res_norm = torch.max(torch.norm(u - u_prev, dim=1))
            self.depth += 1.0
            all_samp_conv = res_norm <= eps

        return self.map_latent_to_inference(u)

    def device(self):
        return next(self.parameters()).data.device

    def data_space_forward(self, d: image) -> latent_variable:
        ''' Transform images into feature vectors in latent space

            The data space operator does *not* need to be 1-Lipschitz; however,
            bounding the singular values can improve generalization. A
            multiplicative factor is added in each update to control the
            Lipschitz constant.
        '''

        n_samples = d.shape[0]
        d = d.view(n_samples, -1)
        Qd = self.leaky_relu(self.fc_d(d))
        return Qd

    def latent_space_forward(self, u: latent_variable,
                             v: latent_variable) -> latent_variable:
        ''' Fixed point operator on latent space (when v is fixed)

            R(u,v) is used in fixed point iteration of FPN to
            find u* satisfying u* = R(u*, v).
            To make R be a contraction in u, we estimate a
            Lipschitz constant and normalize updates using this.
        '''
        uv = u + v
        uv = self.leaky_relu(self.fc_latent(uv))
        R_uv = 0.5*uv

        return R_uv

    def map_latent_to_inference(self, u: latent_variable) -> classification:
        ''' Transform feature vectors into a classification

            This is the final step of FPN, which flattens and
            then applies affine mappings to input. Operations do *not* need to
            be 1-Lipschitz.
        '''
        y = self.fc_y(u)
        return y

    def normalize_lip_const(self, u, Qd):
        return
# ------------------------------------------------
# test JJT symmetry
# ------------------------------------------------


def v_JJT_matvec(v, u, Ru):
    # inputs:
    # v = vector to be multiplied by JJT
    # u = fixed point vector u (requires grad)
    # Ru = R applied to u (requires grad)

    # assumes one rhs: x (n_samples, n_dim, n_rhs) -> (n_samples, n_dim)

    v = v.squeeze(2)      # squeeze number of rhs
    v = v.view(Ru.shape)  # reshape to filter space
    v.requires_grad = True

    # compute v*J = v*(I - dRdu)
    v_dRdu = torch.autograd.grad(outputs=Ru, inputs=u,
                                 grad_outputs=v,
                                 retain_graph=True,
                                 create_graph=True,
                                 only_inputs=True)[0]
    v_J = v - v_dRdu

    # compute v_JJT
    v_JJT = torch.autograd.grad(outputs=v_J, inputs=v,
                                grad_outputs=v_J,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    v = v.detach()
    v_J = v_J.detach()
    Amv = v_JJT.detach()
    Amv = Amv.view(Ru.shape[0], -1)
    Amv = Amv.unsqueeze(2).detach()
    return Amv


def test_symmetry_of_Jacobians():

    n_features = 10
    u = torch.randn(1, n_features)
    u.requires_grad = True
    fc = torch.nn.Linear(n_features, n_features)
    relu = torch.nn.ReLU()
    Ru = relu(fc(u))

    JJT_mat = torch.zeros(n_features, n_features)
    for i in range(n_features):
        temp_vec = torch.zeros(n_features)
        temp_vec[i] = 1.0

        # reshape to match dimensions of v_JJT_matvec function
        temp_vec = temp_vec.view(1, n_features, 1)

        v_JJT = v_JJT_matvec(temp_vec, u, Ru)
        v_JJT = v_JJT.view(n_features)

        JJT_mat[i, :] = v_JJT

    assert(torch.norm(JJT_mat - JJT_mat.transpose(1, 0)) < 1e-6)
    print('--------- symmetry test passed! ---------')


def test_Neumann_approximation():
    n_features = 3
    A = torch.randn(n_features, n_features)/10
    Id = torch.eye(n_features, n_features)
    J = Id - A
    x = torch.randn(3)
    x.requires_grad = True

    y = A.matmul(x)
    dldu = torch.randn(3)

    true_sol = dldu.matmul(torch.inverse(J))

    dldu_Jinv_approx = dldu
    dldu_dfdx_k = dldu.clone().detach()
    neumann_order=50

    # Approximate Jacobian inverse with Neumann series
    # expansion up to neumann_order terms
    for i in range(1, neumann_order):
        dldu_dfdx_k.requires_grad = True
        # compute dldu_dfdx_k * dfdx = dldu_dfdx_k+1
        dfdu_kplus1 = torch.autograd.grad(outputs=y,
                                          inputs = x,
                                          grad_outputs=dldu_dfdx_k,
                                          retain_graph=True,
                                          create_graph = True,
                                          only_inputs=True)[0]

        dldu_Jinv_approx = dldu_Jinv_approx + dfdu_kplus1.detach()

        dldu_dfdx_k = dfdu_kplus1.detach()

    assert(torch.norm(dldu_Jinv_approx - true_sol) < 1e-6)
    print('---- Neumann test passed! ----')
