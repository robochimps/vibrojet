import jax
import jax.numpy as jnp
from jax import lax
from jax.core import ShapedArray
from jax.experimental import jet
from jax.extend.core import Primitive
from jax.interpreters import ad, mlir, batching

jax.config.update("jax_enable_x64", True)


def fact(n):
    return lax.exp(lax.lgamma(n + 1.0))


# Define JAX.jet rules for missing primitive functions


##################
# matrix inversion
##################

inv_p = Primitive("_inv")


def inv(a, **kw):
    return inv_p.bind(a, **kw)


def inv_impl(a, **kw):
    return jnp.linalg.inv(a)
    # return jnp.linalg.pinv(a, hermitian=True)


inv_p.def_impl(inv_impl)


@jax.jit
def inv_jvp(primals, tangents, **kw):
    (a,) = primals
    (da,) = tangents
    a_inv = inv(a)
    da_inv = -a_inv @ da @ a_inv
    return a_inv, da_inv


ad.primitive_jvps[inv_p] = inv_jvp


def inv_abstact_eval(a, **kw):
    shape = a.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Input to 'inv' must be a square matrix")
    N = shape[0]
    dtype = a.dtype
    return ShapedArray((N, N), dtype)


inv_p.def_abstract_eval(inv_abstact_eval)


def inv_lowering(ctx, a, **kw):
    return mlir.lower_fun(jnp.linalg.inv, multiple_results=False)(ctx, a)
    # return mlir.lower_fun(
    #     lambda x: jnp.linalg.pinv(x, hermitian=True), multiple_results=False
    # )(ctx, a)


mlir.register_lowering(inv_p, inv_lowering)


def inv_batch_rule(args, dims):
    (mat,) = args
    (dim,) = dims
    if dim is None:
        return inv_p(mat), None
    m = jax.vmap(jnp.linalg.inv)(mat)
    return m, dim


batching.primitive_batchers[inv_p] = inv_batch_rule


@jax.jit
def _inverse_taylor_rule(primals_in, series_in, **kw):
    (x,) = primals_in
    (x_terms,) = series_in
    u = [x] + x_terms
    v = [None] * len(u)

    if "taylor_coef0" in kw:
        v[0] = kw["taylor_coef0"]
    else:
        # v[0] = jnp.linalg.inv(x)
        v[0] = inv(x)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(1, len(v)):
        v[k] = (
            -fact(k)
            * v[0]
            @ sum(scale(k, j) * u[j] @ v[k - j] for j in range(1, k + 1))
        )
    primal_out, *series_out = v
    return primal_out, series_out


jet.jet_rules[inv_p] = _inverse_taylor_rule

#################################
# matrix eigenvalue decomposition
#################################

eigh_p = Primitive("_eigh")


def eigh(a, **kw):
    e, v = eigh_p.bind(a, **kw)
    return e, v


def eigh_impl(a, **kw):
    e, v = jnp.linalg.eigh(a)
    return e, v


eigh_p.def_impl(eigh_impl)
eigh_p.multiple_results = True


@jax.jit
def eigh_jvp(primals, tangents, **kw):
    (a,) = primals
    (da,) = tangents
    e, v = eigh(a)
    de = jnp.array([v[:, i].T @ da @ v[:, i] for i in range(len(v))])
    inv_de = jnp.pow(e[:, None] - e[None, :] + jnp.eye(len(e)), -1) - jnp.eye(len(e))
    dv = (v.T @ da @ v.T) * inv_de * v
    return (e, v), (de, dv)


ad.primitive_jvps[eigh_p] = eigh_jvp


# def eigh_abstact_eval(a, **kw):
#     return ShapedArray(a.shape[0], a.dtype), ShapedArray(a.shape, a.dtype)


def eigh_abstract_eval(a):
    shape = a.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Input to 'eigh' must be a square matrix")
    N = shape[0]
    dtype = a.dtype
    return ShapedArray((N,), dtype), ShapedArray((N, N), dtype)


eigh_p.def_abstract_eval(eigh_abstract_eval)


def eigh_lowering(ctx, a, **kw):
    return mlir.lower_fun(jnp.linalg.eigh, multiple_results=True)(ctx, a)


mlir.register_lowering(eigh_p, eigh_lowering)


def eigh_batch_rule(args, dims):
    (mat,) = args
    (dim,) = dims
    if dim is None:
        return eigh_p(mat), None
    e, v = jax.vmap(jnp.linalg.eigh)(mat)
    return (e, v), (dim, dim)


batching.primitive_batchers[eigh_p] = eigh_batch_rule


@jax.jit
def _eigh_taylor_rule(primals_in, series_in, **params):
    (x,) = primals_in
    (x_terms,) = series_in
    u = [x] + x_terms
    e = [None] * len(u)
    v = [None] * len(u)

    if "taylor_coef0" in params:
        e[0], v[0] = params["taylor_coef0"]
    else:
        # e[0], v[0] = jnp.linalg.eigh(u[0])
        e[0], v[0] = eigh(u[0])

    eye_ = jnp.eye(len(e[0]))
    inv_de = jnp.pow(e[0][:, None] - e[0][None, :] + eye_, -1) - eye_
    inv_de_v0 = inv_de * v[0]

    for k in range(1, len(v)):
        u_k = v[0].T @ u[k] @ v[0]
        uv_k = v[0].T @ sum(u[j] @ v[k - j] for j in range(1, k + 1))
        e[k] = jnp.array([u_k[i, i] + uv_k[i, i] for i in range(len(v[0]))])
        ev_k = sum((eye_ * e[j]) @ v[k - j] for j in range(1, k + 1))
        v[k] = (-u_k - uv_k + ev_k) * inv_de_v0

    e_primal_out, *e_series_out = e
    v_primal_out, *v_series_out = v
    return (e_primal_out, v_primal_out), (e_series_out, v_series_out)


jet.jet_rules[eigh_p] = _eigh_taylor_rule

######
# Acos
######