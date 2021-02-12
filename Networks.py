import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np

EPS_DEFAULT = 1.0e-5
DEPTH_DEFAULT = 500


class LFPN(ABC, nn.Module):
    """ Learned Fixed Point Network (LFPN) transforms nn.Module
        into a network that uses fixed point iterations to forward prop
        and backprops only through through final "step" of network,
        once it reaches a fixed point. That is,

            forward(d) = map_latent_to_inference(u),

        where u approximately satisfies the fixed point condition

            u = latent_space_forward(u, data_space_forward(d)).
    """

    @abstractmethod
    def name(self) -> str:
        """ Identify name of network
        """
        pass

    @abstractmethod
    def lat_dim(self) -> int:
        """ Identify dimension of latent space variable "u" used in
            latent_space_forward(u, v).
        """
        pass

    @abstractmethod
    def device(self) -> str:
        """ Identify device on which to run network, typically
            'cpu' or 'cuda'.
        """
        pass

    @abstractmethod
    def s_hi(self) -> str:
        """ Largest singular value for nn.Linear mappings
            that do depend only on d and do *not* depend on u.

            Note: All nn.Linear mappings that have inputs of
                  size self.lat_dim() are set have singular values
                  bounded by 1.0. This ensures convergence of the
                  fixed point method in forward().
        """
        pass

    @abstractmethod
    def map_latent_to_inference(self, u: torch.tensor) -> torch.tensor:
        """ Map fixed point "u" of latent_space_forward into the
            inference space. This is the final transformation of the
            network.
        """
        pass

    @abstractmethod
    def latent_space_forward(self, u: torch.tensor,
                             v: torch.tensor) -> torch.tensor:
        """ Fixed point mapping inside the latent space.

            In practice, v = data_space_forward(d) is used to save
            the unnecessary computational cost of recomputing
            data_space_forward(d) in each fixed point update.
        """
        pass

    @abstractmethod
    def data_space_forward(self, d: torch.tensor) -> torch.tensor:
        """ Map input data to the latent space where fixed point
            iterations occur.
        """
        pass

    def bound_lipschitz_constants(self):
        """ Threshold the singular values of the nn.Linear mappings to be
            be bounded by 1.0 if the layer depends on u and s_hi
            otherwise.

            Note: We highly recommend using nn.utils.spectral_norm for
                  convolutional layers, which will automatically make
                  those layers 1-Lipschitz.
        """
        for mod in self.modules():
            if type(mod) == nn.Linear:
                s_hi = 1.0 if mod.weight.data.size()[0] == self.lat_dim \
                       else self.s_hi()
                u, s, v = torch.svd_lowrank(mod.weight.data)
                s[s > s_hi] = s_hi
                mod.weight.data = torch.mm(torch.mm(u, torch.diag(s)), v.t())

    def forward(self, d: torch.tensor, eps: float = EPS_DEFAULT,
                max_depth: int = DEPTH_DEFAULT) -> torch.tensor:
        """ Network inferences satisfy

                forward(d) = map_latent_to_inference(u),

            where u approximately satisfies the fixed point condition

                u = latent_space_forward(u, data_space_forward(d)).

            To obtain the fixed point, we use the iteration

                u <-- latent_space_forward(u, d),

            where we assume users will design the forward step to yield
            a contractive operator with respect to u.
        """
        train_state = self.training
        if train_state:
            self.bound_lipschitz_constants()

        self.eval()
        latent_data = self.data_space_forward(d)
        depth = 0.0
        u = torch.zeros((d.size()[0], self.lat_dim()), device=self.device())
        u_prev = u.clone()
        indices = np.array(range(len(u[:, 0])))
        # Mask shows not converged 'nc' samples (False = converged)
        nc = np.ones((1, u[:, 0].size()[0]), dtype=bool)
        nc = nc.reshape((nc.shape[1]))
        with torch.no_grad():
            while nc.any() > 0 and depth < max_depth:
                u_prev = u.clone()
                u = self.latent_space_forward(u, latent_data)
                nc[nc > 0] = [torch.norm(u[i, :] - u_prev[i, :]) > eps
                              for i in indices[nc > 0]]
                depth += 1.0
        if depth >= max_depth:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        self.depth = depth

        if train_state:
            self.train()
            u = self.latent_space_forward(u, self.data_space_forward(d))
        return self.map_latent_to_inference(u)


class MNIST_CNN(LFPN):
    def __init__(self, lat_dim, device, s_hi=1.0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc_y = nn.Linear(1250,    lat_dim, bias=False)
        self.fc_u = nn.Linear(lat_dim, lat_dim, bias=False)
        self._lat_dim = lat_dim
        self._device = device
        self._s_hi = s_hi
        self.drop_out = nn.Dropout(p=0.2)
        self.soft_max = nn.Softmax(dim=1)
        self.conv1 = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=1,
                                                            out_channels=85,
                                                            kernel_size=3,
                                                            stride=1),
                                                  n_power_iterations=20)
        self.conv2 = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels=85,
                                                            out_channels=50,
                                                            kernel_size=3,
                                                            stride=1),
                                                  n_power_iterations=20)

    def name(self):
        return 'MNIST_CNN'

    def device(self):
        return self._device

    def lat_dim(self):
        return self._lat_dim

    def s_hi(self):
        return self._s_hi

    def latent_space_forward(self, u, v):
        u = self.fc_u(0.5 * self.relu(u) + 0.5 * v)
        y = self.soft_max(u[:, 0:10].clone())
        u[:, 0:10] = y
        return u

    def data_space_forward(self, d):
        v = self.maxpool(self.relu(self.conv1(d)))
        v = self.maxpool(self.relu(self.drop_out(self.conv2(v))))
        v = v.view(d.shape[0], -1)
        v = self.relu(self.fc_y(v))
        return v

    def map_latent_to_inference(self, u):
        return u[:, 0:10]
