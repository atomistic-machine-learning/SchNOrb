import torch
import torch.nn as nn

import schnetpack as spk

class CosineBasis(nn.Module):

    def __init__(self, max_exp):
        super(CosineBasis, self).__init__()
        self.max_exp = max_exp
        self.n_cosines = (max_exp + 1) ** 3
        self.register_buffer('exponents', torch.arange(max_exp + 1, dtype=torch.float))

    def forward(self, cos_ij):
        basis = cos_ij[:, :, :, :, None] ** self.exponents[None, None, None, None]
        basis = basis.reshape(basis.shape[:3] + (-1,))
        print(basis.shape)
        return basis


class FTLayer(nn.Module):
    """
    Factorized Tensor Layer.

    Args:
        n_in (int): Number of input dimensions
        n_factors (int): Number of filter dimensions
        n_out (int): Number of output dimensions
        filter_network (nn.Module): Calculates filter
        cutoff_network (nn.Module): Calculates optional cutoff function (default: None)
        activation (function): Activation function
        normalize_filter (bool): If true, normalize filter to number of neighbors (default: false)
    """

    def __init__(self, n_in, n_factors, n_out, filter_network, cutoff_network=None,
                 activation=None):
        super(FTLayer, self).__init__()
        self.in2f = spk.nn.Dense(n_in, n_factors, bias=True)
        self.f2out = spk.nn.Dense(n_factors, n_out, activation=activation)
        self.filter_network = filter_network
        self.cutoff_network = cutoff_network

    def forward(self, x, r_ij, neighbors, pairwise_mask, f_ij=None):
        """
        Args:
            x (torch.Tensor): Input representation of atomic environments.
            r_ij (torch.Tensor): Interatomic distances.
            neighbors (torch.Tensor): Indices of neighboring atoms.
            pairwise_mask (torch.Tensor): Mask to filter out non-existing neighbors introduced via padding.
            f_ij (torch.Tensor): Use at your own risk. Set to None otherwise.

        Returns:
            torch.Tensor: Continuous convolution.

        """
        if f_ij is None:
            f_ij = r_ij.unsqueeze(-1)

        # calculate filter
        W = self.filter_network(f_ij)

        # apply optional cutoff
        if self.cutoff_network is not None:
            C = self.cutoff_network(r_ij)
            W = W * C.unsqueeze(-1)

        # factorized tensor product
        facts = self.in2f(x)

        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, facts.size(2))

        xi = facts[:, :, None]
        xi = xi.expand(-1, -1, nbh_size[2], -1)
        xj = torch.gather(facts, 1, nbh)
        xj = xj.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)

        y = xi * W * xj

        y = self.f2out(y)

        return y
