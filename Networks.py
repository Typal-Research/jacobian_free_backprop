import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np

EPS_DEFAULT = 1.0e-5
DEPTH_DEFAULT = 500
ALPHA_DEFAULT = 0.1
GAMMA_DEFAULT = 0.9


class KM_params():
    """ All parameters for KM algorithm:
            alpha : relaxation factor in (0, 1)
            gamma : contraction factor in [0, 1]
            eps   : stopping criterion for fixed point residual
            depth : maximum number of iterations before breaking from loop
    """
    def __init__(self, eps=EPS_DEFAULT, depth=DEPTH_DEFAULT,
                 alpha=ALPHA_DEFAULT, gamma=GAMMA_DEFAULT):
        self.eps = eps
        self.depth = depth
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        output = 'KM_params(\n'
        output += '          alpha: %r\n'
        output += '          gamma: %r\n'
        output += '            eps: %r\n'
        output += '          depth: %r\n'
        output += ')\n'
        return output % (self.alpha, self.gamma, self.eps, self.depth)


def replace_forward_with_KM(my_class):
    def decorate(*args, **kwargs):
        net = my_class(*args, **kwargs)
        apply_T = net.forward

        def project_weights(u=None, d=None):
            """ Threshold the singular values of the nn.Linear mappings to be
                in the interval [s_lo, 1.0] and apply power iteration to
                normalize convolutional layers, which is accomplished
                automatically by calling the apply_T command when net is
                in train mode.
            """
            for mod in net.modules():
                if type(mod) == nn.Linear:
                    if mod.weight.data.size()[0] == net.op_dim:
                        mod.weight.data = net.proj_sing_val(mod.weight.data,
                                                            s_hi=1.0)
                    else:
                        mod.weight.data = net.proj_sing_val(mod.weight.data,
                                                            s_hi=net.s_hi())

            if u is not None and d is not None:
                apply_T(u, d)

        def KM(d: torch.tensor, alg_params: KM_params = None) -> torch.tensor:
            """ Krasnoselski Mann (KM) algorithm:
                This scheme finds the fixed point of a contraction variant
                of a nonexpansive operator T. The output u* approximately
                satisfies

                    u* = alpha * gamma * u* + (1 - alpha) * T(u*; d),

                where d is measurement data (e.g., an image). The algorithm
                uses updates of the form

                        u^{k+1} = alpha * u^k + (1 - alpha) * T(u^k; d),

                and the fixed point is unique when gamma < 1.
            """
            # Initialize u to uniform probability concatenated with zeros
            u = torch.zeros((d.size()[0], net.op_dim())).to(net.device())
            u[:, 0:net.sig_dim()] = 1.0 / float(net.sig_dim())

            alg_params = KM_params() if alg_params is None else alg_params
            eps = alg_params.eps
            alpha = alg_params.alpha
            max_depth = alg_params.depth
            gamma = alg_params.gamma

            train_state = net.training
            if train_state:
                net.project_weights(u=u, d=d)

            net.eval()
            depth = 0.0
            u_prev = u.clone()
            indices = np.array(range(len(u[:, 0])))
            u = u.to(net.device())
            # Mask shows not converged 'nc' samples (False = converged)
            nc = np.ones((1, u[:, 0].size()[0]), dtype=bool)
            nc = nc.reshape((nc.shape[1]))
            with torch.no_grad():
                while nc.any() > 0 and depth < max_depth:
                    u_prev = u.clone()
                    u = alpha * gamma * u + (1 - alpha) * apply_T(u, d)
                    nc[nc > 0] = [torch.norm(u[i, :] - u_prev[i, :]) > eps
                                  for i in indices[nc > 0]]
                    depth += 1.0
            if depth >= max_depth:
                print("\nKM: Max Depth Reached - Break Forward Loop\n")

            if train_state:
                net.train()
                u = apply_T(u, d)

            net.depth = depth
            return u.to(net.device())

        net.forward = KM
        net.project_weights = project_weights
        return net
    return decorate


class LFPN(ABC, nn.Module):
    """ Learned Fixed Point Network (LFPN) transforms nn.Module
        into a network that uses KM iterations for forwardprop
        and the simple backprop through the final layer trick for
        backprop.

        Note: This class works in conjunction with the
              'replace_forward_with_KM' decorator.
    """

    @abstractmethod
    def name(self) -> str:
        """ Identify name of network
        """
        pass

    @abstractmethod
    def op_dim(self) -> int:
        """ Identify dimension of u in T(u,d)
        """
        pass

    @abstractmethod
    def sig_dim(self) -> int:
        """ Identify dimension of signal, which is a subvector of u,
            i.e., sig_dim <= op_dim.
        """
        pass

    @abstractmethod
    def device(self) -> str:
        """ Identify device on which to run network, typically
            'cpu' or 'cuda'
        """
        pass

    @abstractmethod
    def s_hi(self) -> str:
        """ Largest singular value for nn.Linear mappings
            that do depend only on d and do *not* depend on u.

            Note: All nn.Linear mappings that have inputs of
                  size self._op_dim are set have singular values
                  bounded by 1.0. This ensures convergence of the
                  KM method.
        """
        pass

    def proj_sing_val(self, matrix, s_hi: float = 1.0):
        """ Bound singular values of matrix by s_hi.
        """
        u, s, v = torch.svd(matrix)
        s[s > s_hi] = s_hi
        return torch.mm(torch.mm(u, torch.diag(s)), v.t())


@replace_forward_with_KM
class MNIST_CNN(LFPN):
    def __init__(self, op_dim, device):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc_y = nn.Linear(1000,    op_dim, bias=True)
        self.fc_u = nn.Linear(op_dim, op_dim, bias=False)
        self._op_dim = op_dim
        self._sig_dim = 10
        self._device = device
        self.drop_out = nn.Dropout(p=0.1)
        self.soft_max = nn.Softmax(dim=1)
        self.conv1 = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=1,
                                                            out_channels=115,
                                                            kernel_size=3,
                                                            stride=1),
                                                  n_power_iterations=1)
        self.conv2 = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=115,
                                                            out_channels=40,
                                                            kernel_size=3,
                                                            stride=1),
                                                  n_power_iterations=1)

    def name(self):
        return 'MNIST_CNN'

    def device(self):
        return self._device

    def op_dim(self):
        return self._op_dim

    def sig_dim(self):
        return self._sig_dim

    def s_hi(self):
        return 2.0

    def forward(self, u, d):
        y = self.maxpool(self.relu(self.conv1(d)))
        y = self.maxpool(self.relu(self.drop_out(self.conv2(y))))
        y = y.view(d.shape[0], -1)
        y = 0.5 * self.fc_u(u) + 0.5 * self.fc_y(y)
        u = self.soft_max(y[:, 0:self._sig_dim].clone())
        y[:, 0:self._sig_dim] = u
        return y
