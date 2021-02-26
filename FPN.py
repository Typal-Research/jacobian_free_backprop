import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np

EPS_DEFAULT = 1.0e-5
MAX_DEPTH_DEFAULT = 500
MAX_SVD_ATTEMPTS = 10


class FPN(ABC, nn.Module):
    """ Fixed Point Network (FPN) transforms nn.Module in a network
        that uses fixed point iterations to forward prop, and backprops
        only through through final "step" of network, once it approximately
        reaches a fixed point. That is,
            forward(d) = map_latent_to_inference(u),
        where u approximately satisfies the fixed point condition
            u = latent_space_forward(u, data_space_forward(d)).
        Users must define each of these three functions, and the forward
        method of nn.Module is defined in terms of these in the abstract
        class.
    """

    @abstractmethod
    def name(self) -> str:
        """ Identify name of network; used for file saving.
        """
        pass

    @abstractmethod
    def lat_dim(self) -> int:
        """ Identify dimension of latent space variable "u" used in
            latent_space_forward(u, v).
        """
        pass

    def device(self) -> str:
        """ Identify device on which to run network, typically
            'cpu' or 'cuda'.
        """
        return next(self.parameters()).data.device

    @abstractmethod
    def s_hi(self) -> float:
        """ Largest singular value for nn.Linear mappings
            that depend only on d and do *not* depend on u.
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
            iterations occur. This is where typical neural network
            structures are inserted, e.g. CNNs or RNNs. The distinction
            is that they output is a latent space variable "u" rather
            than the final inference (in some special isnstances the
            latent space variable is in the same space as the inference
            space).
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
                is_lat_space_op = mod.weight.data.size()[0] == self.lat_dim() \
                                and mod.weight.data.size()[1] == self.lat_dim()
                s_hi = 1.0 if is_lat_space_op else self.s_hi()
                svd_attempts = 0
                compute_svd = False
                while not compute_svd and svd_attempts < MAX_SVD_ATTEMPTS:
                    try:
                        u, s, v = torch.svd(mod.weight.data)
                        compute_svd = True
                    except RuntimeError as e:
                        if 'SBDSDC did not converge' in str(e):
                            print('\nWarning: torch.svd() did not converge. ' +
                                  'Adding Gaussian noise and retrying.\n')
                            mat_size = mod.weight.data.size()
                            noise = torch.randn(mat_size, device=self.device())
                            mod.weight.data += 1.0e-2 * noise
                            svd_attempts += 1
                s[s > s_hi] = s_hi
                mod.weight.data = torch.mm(torch.mm(u, torch.diag(s)), v.t())

    def forward(self, d: torch.tensor, eps: float = EPS_DEFAULT,
                max_depth: int = MAX_DEPTH_DEFAULT) -> torch.tensor:
        """ Network inferences satisfy
                forward(d) = map_latent_to_inference(u),
            where u approximately satisfies the fixed point condition
                u = latent_space_forward(u, data_space_forward(d)).
            To obtain the fixed point, we use the iteration
                u <-- latent_space_forward(u, d),
            where we assume users design the forward step to yield
            a contractive operator with respect to u.
        """
        train_state = self.training
        if train_state:
            self.bound_lipschitz_constants()

        self.eval()
        latent_data = self.data_space_forward(d)
        depth = 0.0
        u = torch.zeros((d.size()[0], self.lat_dim()), device=self.device())
        u_prev = np.Inf*torch.ones(u.shape, device=self.device())

        with torch.no_grad():
            all_samp_conv = False
            while not all_samp_conv and depth < max_depth:
                u_prev = u.clone()
                u = self.latent_space_forward(u, latent_data)
                depth += 1.0
                all_samp_conv = torch.max(torch.norm(u - u_prev, dim=1)) <= eps

        if depth >= max_depth:
            print("\nWarning: Max Depth Reached - Break Forward Loop\n")

        self.depth = depth

        if train_state:
            self.train()
            u = self.latent_space_forward(u, self.data_space_forward(d))
        return self.map_latent_to_inference(u)
