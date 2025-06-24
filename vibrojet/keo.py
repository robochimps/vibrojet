import functools
from typing import Callable, List, Tuple
from functools import reduce
import operator

import jax
import jax.numpy as jnp
import numpy as np
from scipy import constants

from .jet_prim import eigh, inv, lu, acos

jax.config.update("jax_enable_x64", True)

G_to_invcm = (
    constants.value("Planck constant")
    * constants.value("Avogadro constant")
    * 1e16
    / (4.0 * np.pi**2 * constants.value("speed of light in vacuum"))
    * 1e5
)

EPS = jnp.array(
    [
        [[int((i - j) * (j - k) * (k - i) * 0.5) for k in range(3)] for j in range(3)]
        for i in range(3)
    ],
    dtype=jnp.float64,
)


def com(masses: np.ndarray):
    """Wrapper function for `internal_to_cartesian` that computes the Cartesian coordinates
    of atoms from given internal coordinates and shifts them to the center of mass.

    Args:
        masses (np.ndarray): An array containing the masses of the atoms. The order of atoms
            in `masses` must match the order in the output of `internal_to_cartesian`.

    Returns:
        A function that first computes the Cartesian coordinates using `internal_to_cartesian`
        and then shifts them to the center of mass.
    """

    def wrapper(internal_to_cartesian):
        @functools.wraps(internal_to_cartesian)
        def wrapper_com(*args, **kwargs):
            xyz = internal_to_cartesian(*args, **kwargs)
            assert len(xyz) == len(masses), (
                "The number of elements in 'masses' must match the leading dimension of the array "
                "returned by the 'internal_to_cartesian' function"
            )
            masses_ = jnp.asarray(masses)
            com = masses_ @ xyz / jnp.sum(masses_)
            return xyz - com[None, :]

        return wrapper_com

    return wrapper


@functools.partial(jax.jit, static_argnums=2)
def gmat(q, masses, internal_to_cartesian):
    # xyz_g = jax.jacfwd(internal_to_cartesian)(jnp.asarray(q))
    xyz_g = jax.jacrev(internal_to_cartesian)(jnp.asarray(q))
    tvib = xyz_g
    xyz = internal_to_cartesian(jnp.asarray(q))
    trot = jnp.transpose(EPS @ xyz.T, (2, 0, 1))
    ttra = jnp.array([jnp.eye(3, dtype=jnp.float64) for _ in range(len(xyz))])
    tvec = jnp.concatenate((tvib, trot, ttra), axis=2)
    masses_sq = jnp.sqrt(jnp.asarray(masses))
    tvec = tvec * masses_sq[:, None, None]
    tvec = jnp.reshape(tvec, (len(xyz) * 3, len(q) + 6))
    return tvec.T @ tvec


@functools.partial(jax.jit, static_argnums=2)
def Gmat(
    q: np.ndarray,
    masses: np.ndarray,
    internal_to_cartesian: Callable[[jnp.ndarray], jnp.ndarray],
):
    """Computes the kinetic energy G-matrix for a molecular system.

    Args:
        q (np.ndarray): An array of internal coordinates with shape (3N-6,),
            where N is the number of atoms. Bond lengths are given in Angstroms,
            and angles are in radians.
        masses (np.ndarray): A 1D array containing the atomic masses. The order of atoms
            in `masses` must match the order of atoms in the output of `internal_to_cartesian`.
        internal_to_cartesian (Callable): A function that converts internal coordinates `q`
            into Cartesian coordinates, returning an array of shape (number of atoms, 3).

    Returns:
        np.ndarray: A square matrix of shape (ncoo+3+3, ncoo+3+3), representing the elements
        of the kinetic energy G-matrix. The first `ncoo` rows and columns correspond to
        vibrational coordinates, followed by three rotational and three translational
        coordinates. The units of the G-matrix are inverse centimeters.
    """
    return inv(gmat(q, masses, internal_to_cartesian)) * G_to_invcm


batch_Gmat = jax.jit(jax.vmap(Gmat, in_axes=(0, None, None)), static_argnums=2)


@functools.partial(jax.jit, static_argnums=(2, 3))
def _Gmat_s(q, masses, internal_to_cartesian, cartesian_to_internal):
    xyz = internal_to_cartesian(jnp.asarray(q))
    jac = jax.jacfwd(cartesian_to_internal)(xyz)
    return jnp.einsum("kia,lia,i->kl", jac, jac, 1 / masses) * G_to_invcm


batch_Gmat_s = jax.jit(jax.vmap(_Gmat_s, in_axes=(0, None, None, None)))


def zmatrix_coordinates(
    zmat_connections: List[Tuple[int, int, int, int]],
    no_bonds: int,
    no_angles: int,
    internal: jnp.ndarray,
    pm: int = 1,
):
    """Computes the Cartesian coordinates of atoms from given internal coordinates
    in the Z-matrix format.

    Args:
        zmat_connections (List[Tuple[int, int, int, int]]): A list of tuples defining
            the Z-matrix connectivity for each atom. Each tuple consists of:
            - The index of the reference atom for a direct bond connection.
            - The index of the second reference atom to define a bond angle.
            - The index of the third reference atom for either a dihedral angle
                or a secondary valence angle.
            - An integer specifying the type of dihedral or secondary valence angle.
        no_bonds (int): The number of bond distance internal coordinates.
        no_angles (int): The number of angular internal coordinates.
        internal (jnp.ndarray): A 1D array of shape (3N-6) containing the internal
            coordinate values in the following order:
            - Bond distances (first `no_bonds` elements).
            - Valence angles (next `no_angles` elements).
            - Dihedral angles or secondary valence angles (remaining elements).
        pm (int, optional): A factor for controlling the sign of the z-coordinate
            for the second atom. Defaults to 1.

    Returns:
        jnp.ndarray: A 2D array of shape (N, 3) containing the Cartesian coordinates
        of N atoms.
    """
    assert (
        abs(pm) == 1
    ), f"Illegal value for sign parameter 'pm' = {pm} (can be 1 or -1)"

    no_atoms = len(zmat_connections)
    xyz = [jnp.array([0.0, 0.0, 0.0])]

    n1 = jnp.array([0, 0, 1])
    xyz.append(xyz[0] + n1 * internal[0])

    if no_atoms > 2:
        p1, *_ = zmat_connections[1]
        p1 -= 1
        r = internal[1]
        alpha = internal[no_bonds]
        n2 = jnp.array([jnp.sin(alpha), 0.0, pm * jnp.cos(alpha)])
        xyz.append(xyz[p1 - 1] + n2 * r)

    iangle = 0
    idihedral = -1

    for iatom in range(3, no_atoms):

        r = internal[iatom]
        iangle += iangle
        alpha = internal[no_bonds + iangle]

        p1, p2, p3, j = zmat_connections[iatom]
        p1 -= 1
        p2 -= 1
        p3 -= 1

        if j in (-1, 0):
            iangle = iangle + 1
            beta = internal[no_bonds + iangle]

            v12 = xyz[p2] - xyz[p1]
            v23 = xyz[p3] - xyz[p1]
            n2 = v12 / jnp.linalg.norm(v12)
            n3 = jnp.cross(v23, v12)
            n3 = n3 / jnp.linalg.norm(n3)
            n1 = jnp.cross(n2, n3)

            cosa3 = jnp.sum(n2 * v23) / jnp.linalg.norm(v23)
            alpha3 = acos(cosa3)
            cosphi = (jnp.cos(beta) - jnp.cos(alpha) * jnp.cos(alpha3)) / (
                jnp.sin(alpha) * jnp.sin(alpha3)
            )
            phi = acos(cosphi)

            xyz.append(
                xyz[p1]
                + r
                * (
                    jnp.cos(alpha) * n2
                    + jnp.sin(alpha) * jnp.cos(phi) * n1
                    + jnp.sin(alpha) * jnp.sin(phi) * n3
                )
            )

        elif j in (1,):
            idihedral = idihedral + 1
            phi = internal[no_bonds + no_angles + idihedral]

            v12 = xyz[p2] - xyz[p1]
            v23 = xyz[p3] - xyz[p1]
            n2 = v12 / jnp.linalg.norm(v12)
            n3 = jnp.cross(v12, v23)
            n3 = n3 / jnp.linalg.norm(n3)
            n1 = jnp.cross(n2, n3)

            xyz.append(
                xyz[p1]
                + r
                * (
                    jnp.cos(alpha) * n2
                    + jnp.sin(alpha) * jnp.cos(phi) * n1
                    - jnp.sin(alpha) * jnp.sin(phi) * n3
                )
            )

        elif j in (-2, 2):
            idihedral = idihedral + 1
            phi = internal[no_bonds + no_angles + idihedral]

            v12 = xyz[p2] - xyz[p1]
            v23 = xyz[p3] - xyz[p2]
            n2 = v12 / jnp.linalg.norm(v12)
            n3 = jnp.cross(v23, v12)
            n3 = n3 / jnp.linalg.norm(n3)
            n1 = jnp.cross(n2, n3)

            if j < 0:
                n3 = -n3

            xyz.append(
                xyz[p1]
                + r
                * (
                    jnp.cos(alpha) * n2
                    + jnp.sin(alpha) * jnp.cos(phi) * n1
                    - jnp.sin(alpha) * jnp.sin(phi) * n3
                )
            )

        elif j >= 4 and j <= 100:
            iangle = iangle + 1
            beta = internal[no_bonds + iangle]

            v12 = xyz[p2] - xyz[p1]
            v23 = xyz[p3] - xyz[p1]
            n2 = v12 / jnp.linalg.norm(v12)
            n3 = jnp.cross(v12, v23)
            n3 = n3 / jnp.linalg.norm(n3)
            n1 = jnp.cross(n3, n2)

            cosa3 = jnp.sum(n2 * v23) / jnp.linalg.norm(v23)
            alpha3 = acos(cosa3)
            cosphi = (jnp.cos(beta) - jnp.cos(alpha) * jnp.cos(alpha3)) / (
                jnp.sin(alpha) * jnp.sin(alpha3)
            )
            phi = acos(cosphi)

            xyz.append(
                xyz[p1]
                + r
                * (
                    jnp.cos(alpha) * n2
                    + jnp.sin(alpha) * jnp.cos(phi) * n1
                    + jnp.sin(alpha) * jnp.sin(phi) * n3
                )
            )

        else:
            raise ValueError(
                f"Type of dihedral angle in 'zmat_connections' = {j}"
                + f"for atom number {iatom} is not implemented"
            )
    return jnp.array(xyz)


@functools.partial(jax.jit, static_argnums=(2,))
def jac_Gmat(q, masses, internal_to_cartesian):
    return jax.jacfwd(Gmat)(q, masses, internal_to_cartesian)


@functools.partial(jax.jit, static_argnums=(2,))
def jac_Gmat_vib(q, masses, internal_to_cartesian):
    nq = len(q)
    return jax.jacfwd(lambda *arg: Gmat(*arg)[:nq, :nq])(
        q, masses, internal_to_cartesian
    )


@jax.jit
def det(a):
    # NOTE: defines determinant up to a sign
    #   because we lose access to permutation
    #   by calling jax.scipy.linalg.lu(a, permute_l=True)
    #   in jet_prim.lu_impl
    l, u = lu(a)
    ud = [u[i, i] for i in range(len(u))]
    return reduce(operator.mul, ud, 1)


# @jax.jit
# def det(a):
#     e, v  = jnp.linalg.eigh(a)
#     return reduce(operator.mul, e, 1)


@jax.jit
def log_abs_det(a):
    l, u = lu(a)
    return jnp.sum(jnp.array([jnp.log(jnp.abs(u[i, i])) for i in range(len(u))]))


@functools.partial(jax.jit, static_argnums=(2,))
def det_gmat(q, masses, internal_to_cartesian):
    nq = len(q)
    return det(gmat(q, masses, internal_to_cartesian)[: nq + 3, : nq + 3])
    # return jnp.linalg.det(gmat(q, masses, internal_to_cartesian)[: nq + 3, : nq + 3])


@functools.partial(jax.jit, static_argnums=(2,))
def log_abs_det_gmat(q, masses, internal_to_cartesian):
    nq = len(q)
    return log_abs_det(gmat(q, masses, internal_to_cartesian)[: nq + 3, : nq + 3])


@functools.partial(jax.jit, static_argnums=(2,))
def jac_det_gmat(q, masses, internal_to_cartesian):
    return jax.jacrev(det_gmat)(q, masses, internal_to_cartesian)
    # return jax.jacfwd(det_gmat)(q, masses, internal_to_cartesian)


@functools.partial(jax.jit, static_argnums=(2,))
def jac_log_abs_det_gmat(q, masses, internal_to_cartesian):
    return jax.jacrev(log_abs_det_gmat)(q, masses, internal_to_cartesian)


@functools.partial(jax.jit, static_argnums=(2,))
def hess_det_gmat(q, masses, internal_to_cartesian):
    return jax.jacfwd(jax.jacrev(det_gmat))(q, masses, internal_to_cartesian)
    # return jax.jacfwd(det_gmat)(q, masses, internal_to_cartesian)


@functools.partial(jax.jit, static_argnums=(2,))
def hess_log_abs_det_gmat(q, masses, internal_to_cartesian):
    return jax.jacfwd(jax.jacrev(log_abs_det_gmat))(q, masses, internal_to_cartesian)


@functools.partial(jax.jit, static_argnums=(2,))
def pseudo2(
    q: np.ndarray,
    masses: np.ndarray,
    internal_to_cartesian: Callable[[jnp.ndarray], jnp.ndarray],
):
    nq = len(q)
    G = Gmat(q, masses, internal_to_cartesian)[:nq, :nq]
    dG = jac_Gmat(q, masses, internal_to_cartesian)[:nq, :nq, :]
    dG = jnp.transpose(dG, (0, 2, 1))
    det = det_gmat(q, masses, internal_to_cartesian)
    det2 = det * det
    ddet = jac_det_gmat(q, masses, internal_to_cartesian)
    hdet = hess_det_gmat(q, masses, internal_to_cartesian)
    pseudo1 = (jnp.dot(ddet, jnp.dot(G, ddet))) / det2
    # pseudo2 = (jnp.sum(dG * ddet.T) + jnp.sum(G * hdet)) / det
    pseudo2 = (jnp.sum(jnp.diag(jnp.dot(dG, ddet))) + jnp.sum(G * hdet)) / det
    return (-3 * pseudo1 + 4 * pseudo2) / 32.0


@functools.partial(jax.jit, static_argnums=(2,))
def pseudo(
    q: np.ndarray,
    masses: np.ndarray,
    internal_to_cartesian: Callable[[jnp.ndarray], jnp.ndarray],
):
    """Pseudopotential implementation according to Eq. (21)
    in Edit Mátyus, Gábor Czakó, and Attila G. Császár,
    J. Chem. Phys. 130, 134112 (2009)
    http://dx.doi.org/10.1063/1.3076742
    """
    nq = len(q)
    G = Gmat(q, masses, internal_to_cartesian)[:nq, :nq]
    dG = jac_Gmat_vib(q, masses, internal_to_cartesian)
    k = jnp.arange(nq)
    dG = dG[k, :, k]
    dlogdet = jac_log_abs_det_gmat(q, masses, internal_to_cartesian)
    hlogdet = hess_log_abs_det_gmat(q, masses, internal_to_cartesian)
    pseudo1 = dlogdet @ G @ dlogdet
    pseudo2 = jnp.sum(dG @ dlogdet) + jnp.sum(G * hlogdet)
    return (pseudo1 + 4 * pseudo2) / 32.0


batch_pseudo = jax.jit(jax.vmap(pseudo, in_axes=(0, None, None)), static_argnums=2)
