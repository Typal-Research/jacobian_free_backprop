import numpy as np
import torch
from prettytable import PrettyTable


class KM_alg():
    """
    Krasnoselskii Mann (KM) Algorithm

    Purpose: Using input operators S(u) and T(u,d), solve the problem
             Find u  s.t.  u = (1 - alpha) * T(u; d) + alpha * S(u).
    We assume T is a neural network operator (from PyTorch).

    Example Code Snippet:
        KM  = KM_alg(S, T, alpha, device)
        u, depth  = KM(u0, d, eps)
        optimizer.zero_grad()
        y = KM.apply_T(u, d)
        output = loss(y, label)
        output.backward()
        optimizer.step()
    """
    def __init__(self, S, T, alpha: float, device,
                 max_depth=500, eps=1.0e-5):
        self.alpha = alpha
        self.S = S
        self.T = T
        self.max_depth = max_depth
        self.eps_tol = eps
        self._device = device
        self.T.apply_T()  # Force T to perform spectral normalization

    def __repr__(self):
        output = 'KM_alg(\n'
        output += '          alpha: %r\n'
        output += '      max depth: %r\n'
        output += '    default eps: %r\n'
        output += '         device: %r\n'
        output += '              T: %r\n'
        output += ')\n'
        output += str(self.T) + '\n'
        output += str(self.model_params()) + '\n'
        return output % (self.alpha, self.max_depth,
                         self.eps_tol, self._device,
                         self.T.name())

    def __call__(self, u: torch.tensor,
                 d: torch.tensor, eps=-1) -> torch.tensor:
        """ Apply the KM algorithm.

            Training: Use this for forward prop, but do NOT use
                      is for back prop.

            Note: Because u contains batches of samples, we loop
                  until every sample converges, unless we hit the
                  max depth/number of iterations.
        """
        self.T.eval()
        eps = eps if eps > 0 else self.eps_tol
        depth = 0.0
        u_prev = u.clone()
        indices = np.array(range(len(u[:, 0])))
        u = u.to(self._device)
        # Mask identifies not converged 'nc' samples (False = converged)
        nc = np.ones((1, u[:, 0].size()[0]), dtype=bool)
        nc = nc.reshape((nc.shape[1]))
        with torch.no_grad():
            while nc.any() > 0 and depth < self.max_depth:
                u_prev = u.clone()
                u = self.alpha * self.S(u) + (1 - self.alpha) * self.T(u, d)
                nc[nc > 0] = [torch.norm(u[i, :] - u_prev[i, :]) > eps
                              for i in indices[nc > 0]]
                depth += 1.0
        if depth >= self.max_depth:
            print("KM: Max Depth Reached - Break Forward Loop")
        self.T.train()
        return u.to(self._device), depth

    def apply_T(self, u: torch.tensor, d: torch.tensor) -> torch.tensor:
        """ Detach any gradients and then create gradient graph for
            a single application of T. This is used for backprop rather
            than calling the KM algorithm.
        """
        self.T.train()
        return self.T(u.detach(), d)

    def model_params(self):
        table = PrettyTable(["Network Component", "# Parameters"])
        num_params = 0
        for name, parameter in self.T.named_parameters():
            if not parameter.requires_grad:
                continue
            table.add_row([name, parameter.numel()])
            num_params += parameter.numel()
        table.add_row(['TOTAL', num_params])
        return table
