import functools
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np

from .jet_prim import eckart_kappa, eigh
from .jet_prim import set_params, _solve_eckart

jax.config.update("jax_enable_x64", True)


class EckartMethod(Enum):
    exp_kappa = "exp(-kappa) method from https://doi.org/10.1063/1.4923039"
    quaternion = "quaternion algebra method from https://doi.org/10.1063/1.4870936"
    exp_kappa_direct = (
        "exp(-kappa) method from https://doi.org/10.1063/1.4923039, "
        + "direct differentiation through iterative solver"
    )


def eckart(
    q_ref: np.ndarray,
    masses: np.ndarray,
    no_iters: int = 10,
    no_taylor: int = 10,
    no_squaring: int = 4,
    method: EckartMethod = EckartMethod.exp_kappa,
):

    def _wrapper(internal_to_cartesian):
        @functools.wraps(internal_to_cartesian)
        def wrapper_eckart(*args, **kwargs):

            masses_ = jnp.asarray(masses)

            xyz = internal_to_cartesian(*args, **kwargs)

            assert len(xyz) == len(masses), (
                "The number of elements in 'masses' (i.e., number of atoms) must match the leading"
                "dimension of the Cartesian coordinates array returned by the 'internal_to_cartesian'"
                "function"
            )

            com = masses_ @ xyz / jnp.sum(masses_)
            xyz -= com

            xyz_ref = internal_to_cartesian(q_ref, **kwargs)
            com_ref = masses_ @ xyz_ref / jnp.sum(masses_)
            xyz_ref -= com_ref

            if method == EckartMethod.exp_kappa:
                set_params(
                    NO_ITERS_ECKART=no_iters,
                    EXP_TAYLOR_ORDER=no_taylor,
                    EXP_TAYLOR_SQUARING=no_squaring,
                )
                rot_mat = _eckart_expkappa(xyz, xyz_ref, masses_)
            elif method == EckartMethod.quaternion:
                rot_mat = _eckart_quaternion(xyz, xyz_ref, masses_)
            else:
                rot_mat = _eckart_expkappa_direct(xyz, xyz_ref, masses_)

            return xyz @ rot_mat.T

        return wrapper_eckart

    return _wrapper


def _eckart_expkappa(xyz, xyz_ref, masses):
    rot_mat = eckart_kappa(xyz, xyz_ref, masses)
    return rot_mat


def _eckart_expkappa_direct(xyz, xyz_ref, masses):
    rot_mat, _ = _solve_eckart(xyz, xyz_ref, masses)
    return rot_mat


def _eckart_quaternion(xyz, xyz_ref, masses):
    xyz_ma = xyz_ref - xyz
    xyz_pa = xyz_ref + xyz
    x_ma, y_ma, z_ma = xyz_ma.T
    x_pa, y_pa, z_pa = xyz_pa.T

    c11 = jnp.sum(masses * (x_ma**2 + y_ma**2 + z_ma**2))
    c12 = jnp.sum(masses * (y_pa * z_ma - y_ma * z_pa))
    c13 = jnp.sum(masses * (x_ma * z_pa - x_pa * z_ma))
    c14 = jnp.sum(masses * (x_pa * y_ma - x_ma * y_pa))
    c22 = jnp.sum(masses * (x_ma**2 + y_pa**2 + z_pa**2))
    c23 = jnp.sum(masses * (x_ma * y_ma - x_pa * y_pa))
    c24 = jnp.sum(masses * (x_ma * z_ma - x_pa * z_pa))
    c33 = jnp.sum(masses * (x_pa**2 + y_ma**2 + z_pa**2))
    c34 = jnp.sum(masses * (y_ma * z_ma - y_pa * z_pa))
    c44 = jnp.sum(masses * (x_pa**2 + y_pa**2 + z_ma**2))

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

    rot_mat = jnp.array(
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
    return rot_mat
