import torch
import torch.nn as nn
from utils import mnist_loaders, compute_fixed_point
import copy
import numpy as np
from BatchCG import cg_batch


def test_addition():
    assert 1 + 1 == 2

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


test_symmetry_of_Jacobians()

# ------------------------------------------------
# test CG vs Explicit Backprop Error
# ------------------------------------------------


def test_CG_Backprop():
    # compute gradient of networks using Jacobian-based backprop with CG
    # and explicit backprop through the fixed point
    device = 'cpu'
    lat_features = 3
    T1 = test_net(lat_features)
    T2 = copy.deepcopy(T1)

    max_depth = int(1e4)
    eps = 1e-6

    # generate batch of data
    train_batch_size = 1
    train_loader, test_loader = mnist_loaders(
                                    train_batch_size=train_batch_size,
                                    test_batch_size=10)
    (d, labels) = iter(train_loader).next()

    # forward propagate both networks
    T1_output = T1(d, eps=eps, max_depth=max_depth)
    Qd = T2.data_space_forward(d)
    u_fixed_pt, depth = compute_fixed_point(T2,
                                            Qd,
                                            max_depth=max_depth,
                                            eps=eps,
                                            device=device)

    # evaluate once more
    u_fixed_pt.requires_grad = True
    Qd = T2.data_space_forward(d)
    Ru = T2.latent_space_forward(u_fixed_pt, Qd.detach())
    S_Ru = T2.map_latent_to_inference(Ru)

    # compute explicit gradient
    criterion = nn.CrossEntropyLoss()
    loss_explicit = criterion(T1_output, labels)
    loss_explicit.backward()
    explicit_grad_d = T1.fc_d.weight.grad
    explicit_grad_u = T1.fc_latent.weight.grad
    explicit_grad_y = T1.fc_y.weight.grad

    # compute implicit gradient
    loss_implicit = criterion(S_Ru, labels)

    # compute rhs = J * dldu
    dldu = torch.autograd.grad(outputs=loss_implicit,
                               inputs=Ru,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    # compute rhs = J * dldu
    dldu = torch.autograd.grad(outputs=loss_implicit, inputs=Ru,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    # ------------------------------------------------
    # trick for computing v*JT, use two autograds
    # ------------------------------------------------

    # compute dldu_JT:
    dldu_dRdu = torch.autograd.grad(outputs=Ru,
                                    inputs=u_fixed_pt,
                                    grad_outputs=dldu,
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)[0]
    dldu_J = dldu - dldu_dRdu

    # compute J * dldu: take derivative of d(JT*v)/v * v = J*v
    dldu_JT = torch.autograd.grad(outputs=dldu_J,
                                  inputs=dldu,
                                  grad_outputs=dldu,
                                  retain_graph=True,
                                  create_graph=True,
                                  only_inputs=True)[0]
    rhs = dldu_JT

    rhs = rhs.detach()
    # vectorize channels (when R is a CNN)
    rhs = rhs.view(train_batch_size, -1)
    # unsqueeze for number of rhs.
    # CG requires it to have dimensions n_samples x n_features x n_rh
    rhs = rhs.unsqueeze(2)

    def v_JJT_matvec_local(v, u=u_fixed_pt, Ru=Ru):
        return v_JJT_matvec(v, u, Ru)

    tol_cg = 1e-16
    max_iter_cg = 100
    normal_eq_sol, info = cg_batch(v_JJT_matvec_local,
                                   rhs, M_bmm=None,
                                   X0=None,
                                   rtol=tol_cg,
                                   atol=tol_cg,
                                   maxiter=max_iter_cg,
                                   verbose=False)
    # want normal_eq_sol to have size (batch_size x n_hidden_features)
    # so squeeze last dimension
    normal_eq_sol = normal_eq_sol.squeeze(2)
    normal_eq_sol = normal_eq_sol.view(Ru.shape)

    # compute implicit Q and R gradients
    Qd = T2.data_space_forward(d)
    Ru = T2.latent_space_forward(u_fixed_pt, Qd)
    Ru.backward(normal_eq_sol)

    # compute implicit S gradient
    S_Ru = T2.map_latent_to_inference(Ru.detach())
    loss_implicit = criterion(S_Ru, labels)
    loss_implicit.backward()

    implicit_grad_d = T2.fc_d.weight.grad
    implicit_grad_u = T2.fc_latent.weight.grad
    implicit_grad_y = T2.fc_y.weight.grad

    assert(torch.norm(implicit_grad_d - explicit_grad_d) /
           torch.norm(explicit_grad_d) < 1e-5)
    assert(torch.norm(implicit_grad_u - explicit_grad_u) /
           torch.norm(explicit_grad_u) < 1e-5)
    assert(torch.norm(implicit_grad_y - explicit_grad_y) /
           torch.norm(explicit_grad_y) < 1e-5)


test_CG_Backprop()
print('--------- CG Backprop test passed! ---------')


# ------------------------------------------------
# test Neumann Gradient Error
# ------------------------------------------------


# # XXX - Symmmetric Jacobian
# # XXX - Fixed Point Error
# # XXX - Relative Error of CG vs Explicit
# # XXX - Dimension size input to networks
