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
        v[0] = inv(x)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(1, len(v)):
        v[k] = (
            -fact(k)
            * v[0]  # T?
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


eigh_p.multiple_results = True
eigh_p.def_impl(eigh_impl)


@jax.jit
def eigh_jvp(primals, tangents, **kw):
    (a,) = primals
    (da,) = tangents
    e, v = eigh(a)
    eye_ = jnp.eye(len(e))
    de = jnp.array([v[:, i] @ da @ v[:, i] for i in range(len(v))])
    inv_de = jnp.pow(e[:, None] - e[None, :] + eye_, -1) - eye_
    c = (v.T @ da @ v) * inv_de
    dv = v @ c.T
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
    a = [x] + x_terms
    e = [None] * len(a)
    v = [None] * len(a)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    e[0], v[0] = eigh(a[0])

    nprim, nvec = v[0].shape
    eye_ = jnp.eye(nprim)

    mi = [None] * nvec
    for i in range(nvec):
        m1 = jnp.concatenate((jnp.array([[0]]), v[0][:, i : i + 1].T), axis=-1)
        m2 = jnp.concatenate((v[0][:, i : i + 1], eye_ * e[0][i] - a[0]), axis=-1)
        m = jnp.concatenate((m1, m2), axis=0)
        mi[i] = inv(m)

    for k in range(1, len(a)):
        if k == 1:
            b1 = jnp.zeros((nvec, nvec), dtype=jnp.float64)
            b2 = a[k] @ v[0]
        else:
            b1 = (
                -0.5
                * fact(k)
                * sum(scale(k, m) * v[k - m].T @ v[m] for m in range(1, k))
            )
            b2 = fact(k) * (
                sum(scale(k, m) * a[k - m] @ v[m] for m in range(k))
                - sum(scale(k, m) * v[k - m] @ (eye_ * e[m]) for m in range(1, k))
            )

        e_, *v_ = jnp.array(
            [
                mi[i] @ jnp.concatenate((b1[i : i + 1, i], b2[:, i]), axis=0)
                for i in range(nvec)
            ]
        ).T
        e[k] = jnp.array(e_)
        v[k] = jnp.array(v_)

    e_primal_out, *e_series_out = e
    v_primal_out, *v_series_out = v
    return (e_primal_out, v_primal_out), (e_series_out, v_series_out)


jet.jet_rules[eigh_p] = _eigh_taylor_rule


@jax.jit
def _eigh_taylor_rule2(primals_in, series_in, **params):
    (x,) = primals_in
    (x_terms,) = series_in
    u = [x] + x_terms
    e = [None] * len(u)
    v = [None] * len(u)
    e_m = [None] * len(u)

    e[0], v[0] = eigh(u[0])

    n = len(e[0])
    eye_ = jnp.eye(n)
    inv_de = jnp.pow(e[0][:, None] - e[0][None, :] + eye_, -1) - eye_

    e_m[0] = eye_ * e[0]

    def scale_(k, j):
        return 1.0 / (fact(k - j) * fact(j - 1))

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(1, len(v)):

        res = fact(k) * sum(
            scale(k, m) * v[0].T @ u[m] @ v[k - m] for m in range(1, k + 1)
        )
        if k == 1:
            s = jnp.zeros((n, n), dtype=jnp.float64)
        else:
            s = (
                -0.5
                * fact(k)
                * sum(scale(k, m) * v[m].T @ v[k - m] for m in range(1, k))
            )
            # s = -0.5 * sum(v[m].T @ v[k - m] for m in range(1, k))
        res = res + s @ e_m[0] + e_m[0] @ s
        e[k] = jnp.array([res[i, i] for i in range(n)])
        c = -res * inv_de + s

        v[k] = v[0] @ c

        # if k>1:
        #     s = -0.5 * fact(k) * sum(scale(k, m) * v[m].T @ v[k - m] for m in range(1, k))
        #     # s = -0.5 * sum(v[m].T @ v[k - m] for m in range(1, k))
        #     v[k] = v[0] @ s

        # e_m[k] = fact(k) * sum(
        #     scale(k, m) * v[0].T @ u[m] @ v[k - m] for m in range(1, k + 1)
        # )
        # e[k] = jnp.array([e_m[k][i, i] for i in range(n)])

        # c_j = fact(k) * sum(
        #     # scale(k, m) * v[0].T @ (u[m]-e_m[m]) @ v[k - m] for m in range(1, k + 1)
        #     scale(k, m) * v[0].T @ u[m] @ v[k - m]
        #     for m in range(1, k + 1)
        # )
        # c_i = fact(k) * sum(scale(k, m) * v[m].T @ v[k - m] for m in range(1, k))

        # c = -c_j * inv_de - 0.5 * c_i
        # v[k] = v[0] @ c

    e_primal_out, *e_series_out = e
    v_primal_out, *v_series_out = v
    return (e_primal_out, v_primal_out), (e_series_out, v_series_out)


#########################
# matrix LU decomposition
#########################

lu_p = Primitive("_lu")


def lu(a, **kw):
    l, u = lu_p.bind(a, **kw)
    return l, u


def lu_impl(a, **kw):
    l, u = jax.scipy.linalg.lu(a, permute_l=True)
    return l, u


lu_p.def_impl(lu_impl)
lu_p.multiple_results = True


@jax.jit
def lu_jvp(primals, tangents, **kw):
    (a,) = primals
    (da,) = tangents
    l, u = lu(a)
    li = inv(l)
    ui = inv(u)
    f = li @ da @ ui
    du = jnp.triu(f) @ u
    dl = l @ jnp.tril(f, -1)
    return (l, u), (dl, du)


ad.primitive_jvps[lu_p] = lu_jvp


def lu_abstract_eval(a):
    shape = a.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Current implementation of 'lu' works only for square matrix")
    N = shape[0]
    dtype = a.dtype
    return (ShapedArray((N, N), dtype), ShapedArray((N, N), dtype))


lu_p.def_abstract_eval(lu_abstract_eval)


def lu_lowering(ctx, a, **kw):
    return mlir.lower_fun(
        lambda a: jax.scipy.linalg.lu(a, permute_l=True), multiple_results=True
    )(ctx, a)


mlir.register_lowering(lu_p, lu_lowering)


def lu_batch_rule(args, dims):
    (mat,) = args
    (dim,) = dims
    if dim is None:
        return lu_p(mat), None
    l, u = jax.vmap(lambda a: jax.scipy.linalg.lu(a, permute_l=True))(mat)
    return (l, u), (dim, dim)


batching.primitive_batchers[lu_p] = lu_batch_rule


@jax.jit
def _lu_taylor_rule(primals_in, series_in, **params):
    (x,) = primals_in
    (x_terms,) = series_in
    a = [x] + x_terms
    l = [None] * len(a)
    u = [None] * len(a)

    l[0], u[0] = lu(a[0])
    li = inv(l[0])
    ui = inv(u[0])

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(1, len(a)):
        f = (
            li
            @ (a[k] - fact(k) * sum(scale(k, i) * l[i] @ u[k - i] for i in range(1, k)))
            @ ui
        )
        u[k] = jnp.triu(f) @ u[0]
        l[k] = l[0] @ jnp.tril(f, -1)

    l_primal_out, *l_series_out = l
    u_primal_out, *u_series_out = u
    return (l_primal_out, u_primal_out), (l_series_out, u_series_out)


jet.jet_rules[lu_p] = _lu_taylor_rule
