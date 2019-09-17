import math

import numpy as np
import torch
import torch.nn as nn

import schnetpack as spk
from schnetpack.nn.cutoff import HardCutoff
from schnetpack.representation import SchNetInteraction
from schnorb import SchNOrbProperties
from .nn import FTLayer


class SingleAtomHamiltonian(nn.Module):

    def __init__(self, orbital_energies, trainable=False):
        super(SingleAtomHamiltonian, self).__init__()

        if trainable:
            self.orbital_energies = nn.Parameter(
                torch.FloatTensor(orbital_energies))
        else:
            self.register_buffer('orbital_energies',
                                 torch.FloatTensor(orbital_energies))

    def forward(self, numbers, basis):
        tmp1 = (basis[:, None, :, 2] > 0).expand(-1, numbers.shape[1], -1)
        tmp2 = numbers[..., None].expand(-1, -1, basis.shape[-2])
        orb_mask = torch.gather(tmp1, 0, tmp2)
        h0 = self.orbital_energies[numbers]
        h0 = torch.masked_select(h0, orb_mask).reshape(numbers.shape[0], 1, -1)
        h0 = h0.expand(-1, h0.shape[2], -1)
        diag = torch.eye(h0.shape[1], device=h0.device)
        h0 = h0 * diag[None]
        return h0


class SchNorbInteraction(nn.Module):

    def __init__(self, n_spatial_basis, n_factors, n_cosine_basis,
                 cutoff, cutoff_network,
                 normalize_filter=False, dims=3, directions=None):
        super(SchNorbInteraction, self).__init__()
        self.n_cosine_basis = n_cosine_basis
        self._dims = dims
        self.directions = directions

        self.cutoff_network = cutoff_network(cutoff)

        # initialize filters
        self.filter_network = nn.Sequential(
            spk.nn.base.Dense(n_spatial_basis, n_factors,
                              activation=spk.nn.activations.shifted_softplus),
            spk.nn.base.Dense(n_factors, n_factors)
        )

        # initialize interaction blocks
        self.ftensor = FTLayer(n_cosine_basis, n_factors, n_factors,
                               self.filter_network,
                               cutoff_network=self.cutoff_network,
                               activation=spk.nn.activations.shifted_softplus)
        self.atomnet = nn.Sequential(
            spk.nn.Dense(n_factors, n_factors,
                         activation=spk.nn.activations.shifted_softplus),
            spk.nn.Dense(n_factors, n_cosine_basis)
        )

        self.pairnet = nn.Sequential(
            spk.nn.Dense(n_factors, n_factors,
                         activation=spk.nn.activations.shifted_softplus),
            spk.nn.Dense(n_factors, n_cosine_basis)
        )
        self.envnet = nn.Sequential(
            spk.nn.Dense(n_factors, n_factors,
                         activation=spk.nn.activations.shifted_softplus),
            spk.nn.Dense(n_factors, n_cosine_basis)
        )

        if self.directions is not None:
            self.pairnet_mult = spk.nn.Dense(self._dims, directions)
            self.envnet_mult1 = spk.nn.Dense(self._dims, directions)
            self.envnet_mult2 = spk.nn.Dense(self._dims, directions)

        self.agg = spk.nn.Aggregate(axis=2, mean=normalize_filter)
        self.pairagg = spk.nn.Aggregate(axis=2, mean=False)

    def forward(self, xi, r_ij, cos_ij, neighbors, neighbor_mask,
                f_ij=None):
        """
        Args:
            x (torch.Tensor): Atom-wise input representations.
            r_ij (torch.Tensor): Interatomic distances.
            neighbors (torch.Tensor): Indices of neighboring atoms.
            neighbor_mask (torch.Tensor): Mask to indicate virtual neighbors introduced via zeros padding.
            C (torch.Tensor): cosine basis
            f_ij (torch.Tensor): Use at your own risk.

        Returns:
            torch.Tensor: SchNet representation.
        """
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1, 1)
        nbh = nbh.expand(-1, -1, self.n_cosine_basis, cos_ij.shape[3])

        v = self.ftensor.forward(xi, r_ij, neighbors, neighbor_mask, f_ij=f_ij)

        ## energy

        # atomic corrections
        vi = self.agg(v, neighbor_mask)
        vi = self.atomnet(vi)

        ## hamiltonian

        # cosine basis corrections
        # i-j interactions
        vij = self.pairnet(v)
        Vij = vij[:, :, :, :, None] * cos_ij[:, :, :, None, :]
        if self.directions is not None:
            Vij = self.pairnet_mult(Vij)
        # Vij = Vij.reshape(Vij.shape[0], Vij.shape[1], Vij.shape[2],
        #                   Vij.shape[3]*Vij.shape[4])

        # # i-k/j-l interactions
        vik = self.envnet(v)
        vik = vik[:, :, :, :, None] * cos_ij[:, :, :, None, :]
        Vik = vik * neighbor_mask[:, :, :, None, None]
        Vik = self.pairagg(Vik)
        Vjl = torch.gather(Vik, 1, nbh)
        Vjl = Vjl.reshape(Vik.shape[0], nbh_size[1],
                          nbh_size[2], Vik.shape[2],
                          Vik.shape[3])

        if self.directions is not None:
            Vik = self.envnet_mult1(Vik)
            Vjl = self.envnet_mult2(Vjl)

        # Vik = Vik.reshape(Vik.shape[0], Vik.shape[1],
        #                   Vik.shape[2] * Vik.shape[3])
        # Vjl = Vjl.reshape(Vjl.shape[0], Vjl.shape[1], Vjl.shape[2],
        #                   Vjl.shape[3] * Vjl.shape[4])
        Vijkl = Vik[:, :, None] + Vjl

        # # environment-corrected interaction
        V = Vij + Vijkl
        return vi, V  # , Vij


class SchNOrb(nn.Module):

    def __init__(self, n_factors=64, lmax=4, n_interactions=2, cutoff=10.0,
                 n_gaussians=50, directions=4,
                 n_cosine_basis=5,
                 normalize_filter=False, coupled_interactions=False,
                 interaction_block=SchNorbInteraction,
                 cutoff_network=HardCutoff,
                 trainable_gaussians=False, max_z=100):
        super(SchNOrb, self).__init__()
        self.directions = directions

        # atom type embeddings
        self.embedding = nn.Embedding(max_z, n_cosine_basis, padding_idx=0)

        # distances
        self.distances = spk.nn.neighbors.AtomDistances(return_directions=True)
        self.distance_expansion = spk.nn.acsf.GaussianSmearing(
            0.0, cutoff, n_gaussians, trainable=trainable_gaussians
        )

        ### interactions ###
        ## SchNet interaction ##
        if coupled_interactions:
            self.schnet_interactions = nn.ModuleList(
                [
                    SchNetInteraction(
                        n_atom_basis=n_cosine_basis,
                        n_spatial_basis=n_gaussians,
                        n_filters=n_factors,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        normalize_filter=normalize_filter)
                ] * n_interactions)
        else:
            self.schnet_interactions = nn.ModuleList([
                SchNetInteraction(n_atom_basis=n_cosine_basis,
                                  n_spatial_basis=n_gaussians,
                                  n_filters=n_factors,
                                  cutoff=cutoff,
                                  cutoff_network=cutoff_network,
                                  normalize_filter=normalize_filter)
                for _ in range(n_interactions)
            ])

        self.first_interaction = \
            interaction_block(
                n_spatial_basis=n_gaussians,
                n_factors=n_factors,
                n_cosine_basis=n_cosine_basis,
                normalize_filter=normalize_filter,
                cutoff=cutoff,
                cutoff_network=cutoff_network,
                directions=None)

        if coupled_interactions:
            self.interactions = nn.ModuleList(
                [
                    interaction_block(
                        n_spatial_basis=n_gaussians,
                        n_factors=n_factors,
                        n_cosine_basis=n_cosine_basis,
                        directions=directions,
                        cutoff=cutoff,
                        cutoff_network=cutoff_network,
                        normalize_filter=normalize_filter)
                ] * (2 * lmax))
        else:
            self.interactions = nn.ModuleList([
                interaction_block(n_spatial_basis=n_gaussians,
                                  n_cosine_basis=n_cosine_basis,
                                  n_factors=n_factors,
                                  directions=directions,
                                  cutoff=cutoff,
                                  cutoff_network=cutoff_network,
                                  normalize_filter=normalize_filter)
                for _ in range(2 * lmax)
            ])

    def forward(self, inputs):
        atomic_numbers = inputs[SchNOrbProperties.Z]
        positions = inputs[SchNOrbProperties.R]
        cell = inputs[SchNOrbProperties.cell]
        cell_offset = inputs[SchNOrbProperties.cell_offset]
        neighbors = inputs[SchNOrbProperties.neighbors]
        neighbor_mask = inputs[SchNOrbProperties.neighbor_mask]

        # atom embedding
        x0 = self.embedding(atomic_numbers)

        # spatial features: r_ij - distances, cos_ij direction cosines
        r_ij, cos_ij = self.distances(positions, neighbors, cell, cell_offset)
        g_ij = self.distance_expansion(r_ij)
        ones = torch.ones(cos_ij.shape[:3] + (1,), device=cos_ij.device)

        xi = x0

        # atom environments (SchNet-style)
        for interaction in self.schnet_interactions:
            v = interaction(xi, r_ij, neighbors, neighbor_mask, f_ij=g_ij)
            xi = xi + v

        # l=0
        v, V = self.first_interaction(xi, r_ij, ones, neighbors,
                                      neighbor_mask, f_ij=g_ij)
        xi = xi + v
        dirs = self.directions if self.directions is not None else 3
        V = V.expand(-1, -1, -1, -1, dirs)
        # VS =VS.expand(-1, -1, -1, -1, self.directions)

        xij = [V.reshape(V.shape[:3] + (1, -1))]
        # sij = [VS.reshape(VS.shape[:3] + (1, -1))]

        # 1 <= l <= lmax
        for t, interaction in enumerate(self.interactions):
            v, V = interaction(xi, r_ij, cos_ij, neighbors,
                               neighbor_mask, f_ij=g_ij)
            xi = xi + v
            xij.append(V.reshape(V.shape[:3] + (1, -1)))
            # sij.append(VS.reshape(VS.shape[:3] + (1, -1)))

        Xij = torch.cumprod(torch.cat(xij, dim=3), dim=3)
        # Sij = torch.cumprod(torch.cat(sij, dim=3), dim=3)

        del ones
        return x0, xi, Xij


class Hamiltonian(nn.Module):

    def __init__(self, basis_definition, n_cosine_basis, lmax, directions,
                 orbital_energies=None, return_forces=False,
                 quambo=False, create_graph=False,
                 mean=None, stddev=None, max_z=30):
        super(Hamiltonian, self).__init__()
        if return_forces:
            self.derivative = 'forces'
        else:
            self.derivative = None

        self.create_graph = create_graph

        if orbital_energies is None:
            self.h0 = None
        else:
            self.h0 = SingleAtomHamiltonian(orbital_energies, True)
            self.s0 = SingleAtomHamiltonian(np.ones_like(orbital_energies),
                                            True)

        self.register_buffer('basis_definition',
                             torch.LongTensor(basis_definition))
        self.n_types = self.basis_definition.shape[0]
        self.n_orbs = self.basis_definition.shape[1]
        self.n_cosine_basis = n_cosine_basis
        self.quambo = quambo

        directions = directions if directions is not None else 3
        self.offsitenet = spk.nn.Dense(
            n_cosine_basis * directions * (2 * lmax + 1), self.n_orbs ** 2)
        self.onsitenet = spk.nn.Dense(
            n_cosine_basis * directions * (2 * lmax + 1), self.n_orbs ** 2)

        self.ov_offsitenet = spk.nn.Dense(
            n_cosine_basis * directions * (2 * lmax + 1),
            self.n_orbs ** 2)

        if self.quambo:
            self.ov_onsitenet = spk.nn.Dense(
                n_cosine_basis * directions * (2 * lmax + 1), self.n_orbs ** 2)
        else:
            self.ov_onsitenet = nn.Embedding(max_z, self.n_orbs ** 2,
                                             padding_idx=0)
            self.ov_onsitenet.weight.data = torch.diag_embed(
                torch.ones(max_z, self.n_orbs)
            ).reshape(max_z, self.n_orbs ** 2)
            self.ov_onsitenet.weight.data.zero_()
        self.pairagg = spk.nn.Aggregate(axis=2, mean=True)

        self.atom_net = nn.Sequential(
            spk.nn.Dense(n_cosine_basis, n_cosine_basis // 2,
                         activation=spk.nn.activations.shifted_softplus),
            spk.nn.Dense(n_cosine_basis // 2, 1),
            spk.nn.base.ScaleShift(mean, stddev)
        )
        self.atomagg = spk.nn.Aggregate(axis=1, mean=False)

    def forward(self, inputs):
        Z = inputs['_atomic_numbers']
        nbh = inputs[SchNOrbProperties.neighbors]
        # nbhmask = inputs[Properties.neighbor_mask]
        x0, x, Vijkl = inputs['representation']

        # Vijkl shape: batch, max_atoms, max_nbh, max_lr, feats

        batch = Vijkl.shape[0]
        max_atoms = Vijkl.shape[1]

        orb_mask_i = self.basis_definition[:, :, 2] > 0
        orb_mask_i = orb_mask_i[Z].float()
        orb_mask_i = orb_mask_i.reshape(batch, -1, 1)
        orb_mask_j = orb_mask_i.reshape(batch, 1, -1)
        orb_mask = orb_mask_i * orb_mask_j

        ar = torch.arange(max_atoms, device=nbh.device)[None, :, None].expand(
            nbh.shape[0], -1, 1)
        _, nbh = torch.cat([nbh, ar], dim=2).sort(dim=2)

        Vijkl = Vijkl.reshape(Vijkl.shape[:3] + (-1,))

        H_off = self.offsitenet(Vijkl)
        zeros = torch.zeros((batch, max_atoms, 1, self.n_orbs ** 2),
                            device=H_off.device,
                            dtype=H_off.dtype)
        H_off = torch.cat([H_off, zeros], dim=2)
        H_off = torch.gather(H_off, 2, nbh[..., None].expand(-1, -1, -1,
                                                             self.n_orbs ** 2))

        H_on = self.onsitenet(Vijkl)
        H_on = self.pairagg(H_on)
        id = torch.eye(max_atoms, device=H_on.device, dtype=H_on.dtype)[
            None, ..., None]
        H_on = id * H_on[:, :, None]

        H = H_off + H_on

        H = H.reshape(batch, max_atoms, max_atoms, self.n_orbs,
                      self.n_orbs).permute((0, 1, 3, 2, 4))
        H = H.reshape(batch, max_atoms * self.n_orbs, max_atoms * self.n_orbs)

        # symmetrize
        H = 0.5 * (H + H.permute((0, 2, 1)))

        # mask padded orbitals
        H = torch.masked_select(H, orb_mask > 0)
        orbs = int(math.sqrt(H.shape[0] / batch))
        H = H.reshape(batch, orbs, orbs)

        if self.h0 is not None:
            H = H + self.h0(Z, self.basis_definition)

        del zeros

        # overlap
        S_off = self.ov_offsitenet(Vijkl)
        zeros = torch.zeros((batch, max_atoms, 1, self.n_orbs ** 2),
                            device=S_off.device,
                            dtype=S_off.dtype)
        S_off = torch.cat([S_off, zeros], dim=2)
        S_off = torch.gather(S_off, 2, nbh[..., None].expand(-1, -1, -1,
                                                             self.n_orbs ** 2))
        del zeros
        if self.quambo:
            S_on = self.ov_onsitenet(Vijkl)
            S_on = self.pairagg(S_on)
        else:
            S_on = self.ov_onsitenet(Z)
        id = torch.eye(max_atoms, device=H_on.device, dtype=H_on.dtype)[
            None, ..., None]
        S_on = id * S_on[:, :, None]

        S = S_off + S_on

        S = S.reshape(batch, max_atoms, max_atoms, self.n_orbs,
                      self.n_orbs).permute((0, 1, 3, 2, 4))
        S = S.reshape(batch, max_atoms * self.n_orbs, max_atoms * self.n_orbs)

        # symmetrize
        S = 0.5 * (S + S.permute((0, 2, 1)))

        # mask padded orbitals
        S = torch.masked_select(S, orb_mask > 0)
        orbs = int(math.sqrt(S.shape[0] / batch))
        S = S.reshape(batch, orbs, orbs)

        if self.s0 is not None:
            S = S + self.s0(Z, self.basis_definition)

        # total energy
        Ei = self.atom_net(x)
        E = self.atomagg(Ei)

        if self.derivative is not None:
            F = -torch.autograd.grad(E, inputs[SchNOrbProperties.R],
                                     grad_outputs=torch.ones_like(E),
                                     create_graph=self.create_graph)[0]
        else:
            F = None

        return {
            SchNOrbProperties.ham_prop: H,
            SchNOrbProperties.ov_prop: S,
            SchNOrbProperties.en_prop: E,
            SchNOrbProperties.f_prop: F
        }
