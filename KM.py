import numpy as np
import torch


class KM_alg():
    """
    Krasnoselskii Mann (KM) Algorithm
    Purpose: Using input operators S(u) and T(u,d), solve the problem
             Find u  s.t.  u = (1 - alpha) * T(u; d) + alpha * S(u).
    We assume T is a neural network operator.

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
                 max_depth=200, eps=1.0e-5):
        self._alpha = alpha
        self._S = S
        self._T = T
        self._max_depth = max_depth
        self._eps = eps
        self._device = device

    def __repr__(self):
        output = 'KM_alg(\n'
        output += '      alpha       = %r\n'
        output += '      max depth   = %r\n'
        output += '      default eps = %r\n'
        output += '      device      = %r\n'
        output += ')'
        return output % (self._alpha, self._max_depth, self._eps, self._device)

    def __call__(self, u: torch.tensor,
                 d: torch.tensor, eps=-1) -> torch.tensor:

        eps = eps if eps > 0 else self._eps
        depth = 0.0
        u_prev = u.clone()
        indices = np.array(range(len(u[:, 0])))
        u = u.to(self._device)
        # Mask identifies not converged 'nc' samples (False = converged)
        nc = np.ones((1, u[:, 0].size()[0]), dtype=bool)
        nc = nc.reshape((nc.shape[1]))
        with torch.no_grad():  # avoid storing grad in memory
            # loop until all samples converge or max out iterations
            while nc.any() > 0 and depth < self._max_depth:
                Tu = self._T(u, d)
                u_prev = u.clone()
                u = self._alpha * self._S(u) + (1-self._alpha) * Tu
                nc[nc > 0] = [torch.norm(u[i, :] - u_prev[i, :]) > eps
                              for i in indices[nc > 0]]
                depth += 1.0
        if depth >= self._max_depth:
            print("KM: Max Depth Reached - Break Forward Loop")
        return u.to(self._device), depth

    def apply_T(self, u: torch.tensor, d: torch.tensor) -> torch.tensor:
        """ Detach any gradients and then create gradient graph for
            a singular application of T. Use this in lieu of calling KM
            for network forward prop.
        """
        y = u.detach()
        return self._T(y, d)

    def assign_ops(self, S, T):
        """ Use this to update T after every step of optimizer during training
        """
        self._S = S
        self._T = T
