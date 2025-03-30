import functools

import jax
import jax.numpy as jnp
import numpy as np
from .jet_prim import inv, eckart_kappa

jax.config.update("jax_enable_x64", True)


def eckart_rotation(xyz, xyz_ref, masses, no_iters: int = 10):
    u = jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * xyz[:, None, :], axis=0)

    mat = jnp.array(
        [
            [u[0, 0] + u[1, 1], u[1, 2], -u[0, 2]],
            [u[2, 1], u[0, 0] + u[2, 2], u[0, 1]],
            [-u[2, 0], u[1, 0], u[1, 1] + u[2, 2]],
        ]
    )
    imat = inv(mat)

    exp_kappa = jnp.eye(3)
    l = jnp.eye(3)

    for _ in range(no_iters):
        rhs = jnp.sum(
            jnp.array(
                [
                    l[0] * u[1] - l[1] * u[0],
                    l[0] * u[2] - l[2] * u[0],
                    l[1] * u[2] - l[2] * u[1],
                ]
            ),
            axis=-1,
        )
        kxy, kxz, kyz = imat @ rhs
        kappa = jnp.array(
            [
                [0.0, kxy, kxz],
                [-kxy, 0.0, kyz],
                [-kxz, -kyz, 0.0],
            ]
        )
        exp_kappa = _expm_pade(-kappa)
        l = exp_kappa + kappa

    return xyz @ exp_kappa.T


def eckart(q_ref: np.ndarray, masses: np.ndarray, no_iters: int = 10):

    def _wrapper(internal_to_cartesian):
        @functools.wraps(internal_to_cartesian)
        def wrapper_eckart(*args, **kwargs):

            masses_ = jnp.asarray(masses)

            xyz = internal_to_cartesian(*args, **kwargs)

            assert len(xyz) == len(masses), (
                "The number of elements in 'masses' must match the leading dimension of the array "
                "returned by the 'internal_to_cartesian' function"
            )

            com = masses_ @ xyz / jnp.sum(masses_)
            xyz -= com

            xyz_ref = internal_to_cartesian(q_ref, **kwargs)
            com_ref = masses_ @ xyz_ref / jnp.sum(masses_)
            xyz_ref -= com_ref

            # kappa = eckart_kappa(xyz, xyz_ref, masses_)
            # exp_kappa = _expm_pade(-kappa)
            exp_kappa = eckart_kappa(xyz, xyz_ref, masses_)

            # u = jnp.sum(
            #     masses_[:, None, None] * xyz_ref[:, :, None] * xyz[:, None, :], axis=0
            # )
            # mat = jnp.array(
            #     [
            #         [u[0, 0] + u[1, 1], u[1, 2], -u[0, 2]],
            #         [u[2, 1], u[0, 0] + u[2, 2], u[0, 1]],
            #         [-u[2, 0], u[1, 0], u[1, 1] + u[2, 2]],
            #     ]
            # )
            # imat = inv(mat)
            # exp_kappa = jnp.eye(3)
            # l = jnp.eye(3)
            # for _ in range(no_iters):
            #     rhs = jnp.sum(
            #         jnp.array(
            #             [
            #                 l[0] * u[1] - l[1] * u[0],
            #                 l[0] * u[2] - l[2] * u[0],
            #                 l[1] * u[2] - l[2] * u[1],
            #             ]
            #         ),
            #         axis=-1,
            #     )
            #     kxy, kxz, kyz = imat @ rhs
            #     kappa = jnp.array(
            #         [
            #             [0.0, kxy, kxz],
            #             [-kxy, 0.0, kyz],
            #             [-kxz, -kyz, 0.0],
            #         ]
            #     )
            #     # exp_kappa = _expm_taylor(-kappa, order=10)
            #     exp_kappa = _expm_pade(-kappa)
            #     l = exp_kappa + kappa

            return xyz @ exp_kappa.T

        return wrapper_eckart

    return _wrapper


def _expm_pade(a):
    b = jnp.array(
        [
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ],
        dtype=jnp.float64,
    )

    a2 = a @ a
    a4 = a2 @ a2
    a6 = a2 @ a4

    u = a @ (
        b[13] * a6 @ a6
        + b[11] * a6 @ a4
        + b[9] * a6 @ a2
        + b[7] * a6
        + b[5] * a4
        + b[3] * a2
        + b[1] * jnp.eye(3)
    )

    v = (
        b[12] * a6 @ a6
        + b[10] * a6 @ a4
        + b[8] * a6 @ a2
        + b[6] * a6
        + b[4] * a4
        + b[2] * a2
        + b[0] * jnp.eye(3)
    )

    return inv(v - u) @ (v + u)


def _expm_taylor(a, order: int = 10):
    # norm = jnp.linalg.norm(a)
    # scaling = jnp.maximum(0, jnp.ceil(jnp.log2(norm)).astype(int))
    scaling = 2
    a_scaled = a / (2**scaling)

    exp_a = jnp.eye(a.shape[0])
    a_ = jnp.eye(a.shape[0])

    for k in range(1, order + 1):
        a_ = a_ @ a_scaled / k
        exp_a += a_

    for _ in range(scaling):
        exp_a = exp_a @ exp_a
    return exp_a
