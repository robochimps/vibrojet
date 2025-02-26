import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import factorial
from taylor import deriv, deriv_list, inv_taylor
from jacobi import jacobi
from jax.experimental import jet
from jax import lax
from functools import partial

from jax.extend.core import Primitive
from jax.interpreters import ad

jax.config.update("jax_enable_x64", True)


my_inv_p = Primitive("my_inv")

def my_inv(a):
    return my_inv_p.bind(a)

def my_inv_impl(a):
    return jnp.linalg.inv(a)

my_inv_p.def_impl(my_inv_impl)

def my_inv_jvp(primals, tangents):
    a = primals
    da = tangents
    ai = jnp.linalg.inv(a)
    dai = -jnp.dot(ai, jnp.dot(a, ai))
    return ai, dai

ad.primitive_jvps[my_inv_p] = my_inv_jvp


def fact(n):
    return lax.exp(lax.lgamma(n + 1.0))


def _bilinear_taylor_rule2(prim, primals_in, series_in, **params):
    # print('_bilinear_taylor_rule2')
    x, y = primals_in
    x_terms, y_terms = series_in
    u = [x] + x_terms
    w = [y] + y_terms
    v = [None] * len(u)
    op = partial(prim.bind, **params)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(0, len(v)):
        v[k] = fact(k) * sum(scale(k, j) * op(u[j], w[k - j]) for j in range(0, k + 1))
    primal_out, *series_out = v
    return primal_out, series_out

@jax.jit
def _inverse_taylor_rule(primals_in, series_in):
    # print('_inverse_taylor_rule')
    x, = primals_in
    x_terms ,= series_in
    u = [x] + x_terms
    v = [None] * len(u)
    xi = jnp.linalg.inv(x)
    v[0] = xi
    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))
    for k in range(1, len(v)):
        v[k] = -fact(k)*jnp.dot(xi, sum(scale(k,j) *jnp.dot(u[j], v[k - j]) for j in range(1, k + 1)))
    primal_out, *series_out = v
    return primal_out, series_out

jet.jet_rules[lax.dot_general_p] = partial(_bilinear_taylor_rule2, lax.dot_general_p)
# jet.jet_rules[lax.linear_solve_p] = partial(_inverse_taylor_rule, lax.linear_solve_p)
jet.jet_rules[my_inv_p] = _inverse_taylor_rule

eps = jnp.array(
    [
        [[int((i - j) * (j - k) * (k - i) * 0.5) for k in range(3)] for j in range(3)]
        for i in range(3)
    ],
    dtype=jnp.float64,
)


def com(masses):
    def wrapper(cart):
        @functools.wraps(cart)
        def wrapper_com(*args, **kwargs):
            xyz = cart(*args, **kwargs)
            com = jnp.dot(jnp.asarray(masses), xyz) / jnp.sum(jnp.asarray(masses))
            return xyz - com[None, :]

        return wrapper_com

    return wrapper


# @functools.partial(jax.jit, static_argnums=2)
def gmat(q, masses, internal_to_cartesian):
    xyz_g = jax.jacfwd(internal_to_cartesian)(jnp.asarray(q))
    tvib = xyz_g
    xyz = internal_to_cartesian(jnp.asarray(q))
    natoms = xyz.shape[0]
    trot = jnp.transpose(jnp.dot(eps, xyz.T), (2, 0, 1))
    ttra = jnp.array([jnp.eye(3, dtype=jnp.float64) for _ in range(natoms)])
    tvec = jnp.concatenate((tvib, trot, ttra), axis=2)
    masses_sq = jnp.sqrt(jnp.asarray(masses))
    tvec = tvec * masses_sq[:, None, None]
    tvec = jnp.reshape(tvec, (natoms * 3, len(q) + 6))
    return jnp.dot(tvec.T, tvec)


# @functools.partial(jax.jit, static_argnums=2)
def Gmat(q, masses, internal_to_cartesian):
    g = gmat(q, masses, internal_to_cartesian)
    # return jnp.linalg.inv(g)
    return my_inv(g)


batch_Gmat = jax.jit(jax.vmap(Gmat, in_axes=(0, None, None)))


@functools.partial(jax.jit, static_argnums=(2, 3))
def Gmat_s(q, masses, internal_to_cartesian, cartesian_to_internal):
    xyz = internal_to_cartesian(jnp.asarray(q))
    jac = jax.jacfwd(cartesian_to_internal)(xyz)
    return jnp.einsum("kia,lia,i->kl", jac, jac, 1 / masses)


batch_Gmat_s = jax.jit(jax.vmap(Gmat_s, in_axes=(0, None, None, None)))

if __name__ == "__main__":
    import itertools

    masses = np.array([15.9994, 1.00782505, 1.00782505])
    r1 = 0.958
    r2 = 0.958
    alpha = 1.824

    x0 = jnp.array([r1, r2, alpha], dtype=jnp.float64)

    # @eckart(x0, masses)
    @com(masses)
    def internal_to_cartesian(internal_coords):
        r1, r2, a = internal_coords
        return jnp.array(
            [
                [0.0, 0.0, 0.0],
                [r1 * jnp.sin(a / 2), 0.0, r1 * jnp.cos(a / 2)],
                [-r2 * jnp.sin(a / 2), 0.0, r2 * jnp.cos(a / 2)],
            ]
        )

    max_order = 4
    deriv_ind = [
        elem
        for elem in itertools.product(
            *[range(0, max_order + 1) for _ in range(len(x0))]
        )
        if sum(elem) <= max_order
    ]

    # func = lambda q: gmat(q, masses, internal_to_cartesian)
    func = lambda q: Gmat(q, masses, internal_to_cartesian)
    gmat_coefs = deriv_list(func, deriv_ind, x0, if_taylor=True)
    # gmat_coefs = inv_taylor(np.array(deriv_ind), gmat_coefs)
    print(gmat_coefs[:,2,2])
    # Gmat_coefs = inv_taylor(np.array(deriv_ind), gmat_coefs)[..., 2, 2]
    # Gmat_coefs = gmat_coefs[...,2,2]

    # func = lambda q: Gmat(q, masses, internal_to_cartesian)[2, 2]

    # def jacfwd(x0, ind):
    #     f = func
    #     for _ in range(sum(ind)):
    #         f = jax.jacfwd(f)
    #     i = sum([(i,) * o for i, o in enumerate(ind)], start=tuple())
    #     # print(i)
    #     return f(x0)[i]

    # Gmat_coefs2 = np.array(
    #     [jacfwd(x0, ind) / np.prod(factorial(ind)) for ind in deriv_ind]
    # )

    # # print(Gmat_coefs.shape, Gmat_coefs2.shape)
    # for i, ind in enumerate(deriv_ind):
    #     print(
    #         ind,
    # #         np.round(Gmat_coefs[i], 8),
    #         np.round(Gmat_coefs2[i], 8),
    # #         np.round(Gmat_coefs[i] - Gmat_coefs2[i], 8),
    #     )
