from itertools import chain

import numpy as np
import quaternion
from scipy import linalg as la
from spherical_functions import Wigner_D_matrices


class Rotator:

    def __init__(self, basis, phase=np.array([1., 1., 1.])):
        self.basis = basis
        self.phase = phase
        self.lmax = self.basis[:, :, 3].max()
        self.lgroups = []

        self.lvec = []
        for elmnt in self.basis:
            ls = elmnt[elmnt[:, 2] > 0, 3]
            elmt_lvec = []
            row = 0
            while row < ls.shape[0]:
                l = ls[row]
                elmt_lvec.append(l)
                row += 2 * l + 1
            self.lvec.append(np.array(elmt_lvec, dtype=np.int))

        self.lsplit = np.cumsum(
            (2 * np.arange(0., self.lmax, dtype=np.int) + 1) ** 2)

        self._calc_U()
        self._calc_P()

    def _calc_P(self):
        self.Ps = []

        for l in range(self.lmax + 1):
            sidx = np.arange(0, 2 * l + 1, 1, dtype=np.int)
            self.Ps.append(sidx)

    def _calc_U(self):
        """Compute the U transformation matrix."""
        self.Us = []

        for l in range(self.lmax + 1):
            U = np.zeros((2 * l + 1, 2 * l + 1), dtype=np.complex)
            for m in range(-l, l + 1):
                for n in range(-l, l + 1):
                    U[m + l, n + l] = self.Umn(l, m, n)
            self.Us.append(U)

    def Umn(self, l, m, n):
        if n < 0:
            term1 = 1j
        elif n == 0:
            term1 = np.sqrt(2) / 2
        else:
            term1 = 1
        if (m > 0) and (n < 0) and (n % 2 == 0):
            term2 = -1
        elif (m > 0) and (n > 0) and (n % 2 != 0):
            term2 = -1
        else:
            term2 = 1
        return term1 * term2 / np.sqrt(2) * ((m == n) + (m == -n))

    def _calc_UDUs(self, q):
        Ds = Wigner_D_matrices(q, 0, self.lmax)
        Ds = np.split(Ds, self.lsplit)

        UDUs = []
        for U, pidx, D in zip(self.Us, self.Ps, Ds):
            D = D.reshape(U.shape)
            # udu = np.real(la.inv(U) @ D @ U)
            udu = np.real(U.conjugate().T @ D @ U)
            # print('pidx', pidx)
            # print('udu', udu)
            # if len(pidx) == 7:
            #     #print(pidx)
            #     # pidx = [3, 4, 2, 5, 1, 6, 0]
            #     pidx = np.array([
            #         [0, 0, 0, 1, 0, 0, 0],
            #         [0, 0, 0, 0, 1, 0, 0],
            #         [0, 0, 1, 0, 0, 0, 0],
            #         [0, 0, 0, 0, 0, 1, 0],
            #         [0, 1, 0, 0, 0, 0, 0],
            #         [0, 0, 0, 0, 0, 0, 1],
            #         [1, 0, 0, 0, 0, 0, 0],
            #     ])
            #     udu = pidx.dot(udu.dot(pidx.T))
            # else:
            udu = udu[np.ix_(pidx, pidx)]
            # print('udu', udu)

            UDUs.append(udu)
        UDUs = np.array(UDUs)
        return UDUs

    def transform(self, R, H, S, numbers, positions=None, forces=None):
        q = quaternion.from_rotation_matrix(R)
        vR = quaternion.as_rotation_vector(q) * self.phase
        q = quaternion.from_rotation_vector(vR)

        UDUs = self._calc_UDUs(q)

        M = chain(*[UDUs[self.lvec[z]] for z in numbers])
        M = la.block_diag(*M)

        H = M.T @ H @ M
        S = M.T @ S @ M

        pos_rot = positions @ R.T
        if forces is not None:
            force_rot = forces @ R.T
            return H, S, pos_rot, force_rot
        return H, S, pos_rot


class OrcaRotator(Rotator):

    def __init__(self, basis):
        phase = np.array([1., -1., 1.])
        self.T = np.array(
            [[1],
             [1, -1, 1],
             [1, 1, 1, 1, 1],
             [1, 1, 1, 1, -1, 1, -1],
             [1, 1, 1, 1, 1, -1, 1, -1, 1],
             [1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1],
             [1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
              1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
              -1, 1, -1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1,
              1, -1, 1, -1, 1]
            ]
        )

        super(OrcaRotator, self).__init__(basis, phase)

    def _calc_P(self):
        self.Ps = []

        for l in range(self.lmax + 1):
            ms = np.zeros((2 * l + 1,), dtype=np.int)
            ms[2::2] = -np.arange(1, l + 1, 1)
            self.Ps.append(np.argsort(np.argsort(ms)))
            ms[1:-1:2] = np.arange(1, l + 1, 1)
        print(self.Ps)

    def _calc_U(self):
        """Compute the U transformation matrix."""
        super(OrcaRotator, self)._calc_U()

        # for l, U in enumerate(self.Us):
        #     self.Us[l] = np.diag(self.T[l]).dot(self.Us[l].T).T


class AimsRotator(Rotator):

    def __init__(self, basis):
        phase = np.array([1., -1., 1.])
        self.T = np.array(
            [[1],
             [1, 1, -1],
             [1, 1, 1, -1, 1],
             [1, 1, 1, 1, -1, 1, -1],
             [1, 1, 1, 1, 1, -1, 1, -1, 1],
             [1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1],
             [1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
              1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
              -1, 1, -1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1,
              1, -1, 1, -1, 1]
            ]
        )

        super(AimsRotator, self).__init__(basis, phase)

    def _calc_U(self):
        """Compute the U transformation matrix."""
        super(AimsRotator, self)._calc_U()

        for l, U in enumerate(self.Us):
            self.Us[l] = np.diag(self.T[l]).dot(self.Us[l].T).T


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M