import functools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy import constants

from .jet_prim import eigh, inv

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


def eckart(q_ref: np.ndarray, masses: np.ndarray):
    """Wrapper function for `internal_to_cartesian` that computes the Cartesian coordinates
    of atoms from given internal coordinates and then rotates them to the Eckart frame.

    Args:
        q_ref (np.ndarray): An array containing reference values of internal coordinates
            for defining the Eckart frame.
        masses (np.ndarray): An array containing the masses of the atoms. The order of atoms
            in `masses` must match the order in the output of `internal_to_cartesian`.

    Returns:
        A function that first computes the Cartesian coordinates using `internal_to_cartesian`
        and then rotates them to the Eckart frame.
    """

    def _wrapper(internal_to_cartesian):
        @functools.wraps(internal_to_cartesian)
        def wrapper_eckart(*args, **kwargs):
            global c_mat
            xyz = internal_to_cartesian(*args, **kwargs)
            assert len(xyz) == len(masses), (
                "The number of elements in 'masses' must match the leading dimension of the array "
                "returned by the 'internal_to_cartesian' function"
            )
            masses_ = jnp.asarray(masses)
            com = masses_ @ xyz / jnp.sum(masses_)
            xyz -= com
            xyz_ref = internal_to_cartesian(q_ref, **kwargs)
            com_ref = masses_ @ xyz_ref / jnp.sum(masses_)
            xyz_ref -= com_ref

            xyz_ma = xyz_ref - xyz
            xyz_pa = xyz_ref + xyz
            x_ma, y_ma, z_ma = xyz_ma.T
            x_pa, y_pa, z_pa = xyz_pa.T

            c11 = jnp.sum(masses_ * (x_ma**2 + y_ma**2 + z_ma**2))
            c12 = jnp.sum(masses_ * (y_pa * z_ma - y_ma * z_pa))
            c13 = jnp.sum(masses_ * (x_ma * z_pa - x_pa * z_ma))
            c14 = jnp.sum(masses_ * (x_pa * y_ma - x_ma * y_pa))
            c22 = jnp.sum(masses_ * (x_ma**2 + y_pa**2 + z_pa**2))
            c23 = jnp.sum(masses_ * (x_ma * y_ma - x_pa * y_pa))
            c24 = jnp.sum(masses_ * (x_ma * z_ma - x_pa * z_pa))
            c33 = jnp.sum(masses_ * (x_pa**2 + y_ma**2 + z_pa**2))
            c34 = jnp.sum(masses_ * (y_ma * z_ma - y_pa * z_pa))
            c44 = jnp.sum(masses_ * (x_pa**2 + y_pa**2 + z_ma**2))

            c = jnp.array(
                [
                    [c11, c12, c13, c14],
                    [c12, c22, c23, c24],
                    [c13, c23, c33, c34],
                    [c14, c24, c34, c44],
                ]
            )

            e, v = eigh(c)
            quar = v[:, 0]

            u = jnp.array(
                [
                    [
                        quar[0] ** 2 + quar[1] ** 2 - quar[2] ** 2 - quar[3] ** 2,
                        2 * (quar[1] * quar[2] + quar[0] * quar[3]),
                        2 * (quar[1] * quar[3] - quar[0] * quar[2]),
                    ],
                    [
                        2 * (quar[1] * quar[2] - quar[0] * quar[3]),
                        quar[0] ** 2 - quar[1] ** 2 + quar[2] ** 2 - quar[3] ** 2,
                        2 * (quar[2] * quar[3] + quar[0] * quar[1]),
                    ],
                    [
                        2 * (quar[1] * quar[3] + quar[0] * quar[2]),
                        2 * (quar[2] * quar[3] - quar[0] * quar[1]),
                        quar[0] ** 2 - quar[1] ** 2 - quar[2] ** 2 + quar[3] ** 2,
                    ],
                ]
            )
            return xyz @ u.T

        return wrapper_eckart

    return _wrapper


@functools.partial(jax.jit, static_argnums=2)
def gmat(q, masses, internal_to_cartesian):
    xyz_g = jax.jacfwd(internal_to_cartesian)(jnp.asarray(q))
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


@functools.partial(jax.jit, static_argnums=(2,))
def dGmat(q, masses, internal_to_cartesian):
    return jax.jacfwd(Gmat)(q, masses, internal_to_cartesian)


batch_dGmat = jax.jit(jax.vmap(dGmat, in_axes=0))


@functools.partial(jax.jit, static_argnums=(2,))
def Detgmat(q, masses, internal_to_cartesian):
    nq = len(q)
    return jnp.linalg.det(gmat(q,masses, internal_to_cartesian)[: nq + 3, : nq + 3])


@functools.partial(jax.jit, static_argnums=(2,))
def dDetgmat(q, masses, internal_to_cartesian):
    return jax.grad(Detgmat)(q, masses, internal_to_cartesian)


@functools.partial(jax.jit, static_argnums=(2,))
def hDetgmat(q, masses, internal_to_cartesian):
    # return jax.jacfwd(jax.jacrev(Detgmat))(q, masses, internal_to_cartesian)
    return jax.jacfwd(jax.jacfwd(Detgmat))(q, masses, internal_to_cartesian)

@functools.partial(jax.jit, static_argnums=(2,))
def pseudo(
    q: np.ndarray,
    masses: np.ndarray,
    internal_to_cartesian: Callable[[jnp.ndarray], jnp.ndarray],
):
    nq = len(q)
    G = Gmat(q, masses, internal_to_cartesian)[:nq, :nq]
    dG = dGmat(q, masses, internal_to_cartesian)[:nq, :nq, :]
    dG = jnp.transpose(dG, (0, 2, 1))
    det = Detgmat(q, masses, internal_to_cartesian)
    det2 = det * det
    ddet = dDetgmat(q, masses, internal_to_cartesian)
    hdet = hDetgmat(q, masses, internal_to_cartesian)
    pseudo1 = (jnp.dot(ddet, jnp.dot(G, ddet))) / det2
    pseudo2 = (jnp.sum(jnp.diag(jnp.dot(dG, ddet))) + jnp.sum(G * hdet)) / det
    return (-3 * pseudo1 + 4 * pseudo2) / 32.0

batch_pseudo = jax.jit(jax.vmap(pseudo, in_axes=(0, None, None)), static_argnums=2)
