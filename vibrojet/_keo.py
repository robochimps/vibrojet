"""Standard implementation of KEO that can be differentiated using jax.jacfwd and jax.jacrev"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy import constants

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


def com(masses):
    def wrapper(internal_to_cartesian):
        @functools.wraps(internal_to_cartesian)
        def wrapper_com(*args, **kwargs):
            xyz = internal_to_cartesian(*args, **kwargs)
            masses_ = jnp.asarray(masses)
            com = masses_ @ xyz / jnp.sum(masses_)
            return xyz - com[None, :]

        return wrapper_com

    return wrapper


def eckart(q_ref, masses):
    def _wrapper(internal_to_cartesian):
        @functools.wraps(internal_to_cartesian)
        def wrapper_eckart(*args, **kwargs):
            global c_mat
            masses_ = jnp.asarray(masses)
            xyz = internal_to_cartesian(*args, **kwargs)
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

            e, v = jnp.linalg.eigh(c)
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
def Gmat(q, masses, internal_to_cartesian):
    return jnp.linalg.inv(gmat(q, masses, internal_to_cartesian)) * G_to_invcm


batch_Gmat = jax.jit(jax.vmap(Gmat, in_axes=(0, None, None)), static_argnums=2)


@functools.partial(jax.jit, static_argnums=(2, 3))
def _Gmat_s(q, masses, internal_to_cartesian, cartesian_to_internal):
    xyz = internal_to_cartesian(jnp.asarray(q))
    jac = jax.jacfwd(cartesian_to_internal)(xyz)
    return jnp.einsum("kia,lia,i->kl", jac, jac, 1 / masses) * G_to_invcm


batch_Gmat_s = jax.jit(jax.vmap(_Gmat_s, in_axes=(0, None, None, None)))
