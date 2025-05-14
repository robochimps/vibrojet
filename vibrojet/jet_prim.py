import jax
import jax.numpy as jnp
from jax import lax
from jax.core import ShapedArray
from jax.experimental import jet
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir

jax.config.update("jax_enable_x64", True)


PARAMS = {
    "NO_ITERS_ECKART": 10,
    "EXP_TAYLOR_ORDER": 10,
    "EXP_TAYLOR_SQUARING": 4,
}


def set_params(**kw):
    for key, val in kw.items():
        if key in PARAMS:
            # print(f"set parameter '{key}' = {val}")
            PARAMS[key] = val
        else:
            raise KeyError(f"Unknown parameter name: {key}")


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


# @jax.jit
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


# @jax.jit
def _inverse_taylor_rule(primals_in, series_in, **kw):
    (x,) = primals_in
    (x_terms,) = series_in
    u = [x] + x_terms
    v = [None] * len(u)

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


# @jax.jit
def eigh_jvp(primals, tangents, **kw):
    (a,) = primals
    (da,) = tangents
    e, v = eigh(a)
    eye_ = jnp.eye(len(e))
    de = jnp.array([v[:, i] @ da @ v[:, i] for i in range(len(v))])
    inv_de = jnp.pow(e[:, None] - e[None, :] + eye_, -1) - eye_
    c = (v.T @ da @ v) * inv_de
    dv = -v @ c
    return (e, v), (de, dv)


ad.primitive_jvps[eigh_p] = eigh_jvp


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


# @jax.jit
def _eigh_taylor_rule(primals_in, series_in, **kw):
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


# @jax.jit
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


# @jax.jit
def _lu_taylor_rule(primals_in, series_in, **kw):
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


#######
# acos
#######

acos_p = Primitive("_acos")


def acos(a, **kw):
    return acos_p.bind(a, **kw)


def acos_impl(a, **kw):
    return jnp.acos(a)


acos_p.def_impl(acos_impl)
acos_p.multiple_results = False


@jax.jit
def acos_jvp(primals, tangents, **kw):
    (a,) = primals
    (da,) = tangents
    x = acos(a)
    dx = -1 / jnp.sqrt(1 - a * a) * da
    return x, dx


ad.primitive_jvps[acos_p] = acos_jvp


def acos_abstract_eval(a):
    return ShapedArray(a.shape, a.dtype)


acos_p.def_abstract_eval(acos_abstract_eval)


def acos_lowering(ctx, a, **kw):
    return mlir.lower_fun(jnp.acos, multiple_results=False)(ctx, a)


mlir.register_lowering(acos_p, acos_lowering)


def acos_batch_rule(args, dims):
    (mat,) = args
    (dim,) = dims
    if dim is None:
        return acos_p(mat), None
    res = jax.vmap(jnp.acos)(mat)
    return res, dim


batching.primitive_batchers[acos_p] = acos_batch_rule


# @jax.jit
def _acos_taylor_rule(primals_in, series_in, **kw):
    (x,) = primals_in
    (series,) = series_in

    primal_out = jnp.acos(x)

    c0, cs = jet.jet(
        lambda x: lax.div(jnp.ones_like(x), -lax.sqrt(1 - lax.square(x))),
        (x,),
        (series,),
    )

    def scale_(k, j):
        return 1.0 / (fact(k - j) * fact(j - 1))

    c = [c0] + cs
    u = [x] + series
    v = [primal_out] + [None] * len(series)
    for k in range(1, len(v)):
        v[k] = fact(k - 1) * sum(
            scale_(k, j) * c[k - j] * u[j] for j in range(1, k + 1)
        )
    primal_out, *series_out = v
    return primal_out, series_out


jet.jet_rules[acos_p] = _acos_taylor_rule


##############
# Eckart kappa
##############

# Implementation of frame rotation matrix that satisfy the Eckart conditions
# A. Yachmenev, S. N. Yurchenko, J. Chem. Phys. 143, 014105 (2015),
# https://doi.org/10.1063/1.4923039

eckart_kappa_p = Primitive("_eckart_kappa")


def eckart_kappa(xyz, xyz_ref, masses, **kw):
    return eckart_kappa_p.bind(xyz, xyz_ref, masses, **kw)


@jax.jit
def _solve_eckart(xyz, xyz_ref, masses):
    u = jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * xyz[:, None, :], axis=0)
    umat = jnp.array(
        [
            [u[0, 0] + u[1, 1], u[1, 2], -u[0, 2]],
            [u[2, 1], u[0, 0] + u[2, 2], u[0, 1]],
            [-u[2, 0], u[1, 0], u[1, 1] + u[2, 2]],
        ]
    )
    inv_umat = inv(umat)

    exp_kappa = jnp.eye(3)
    kappa = jnp.zeros_like(exp_kappa)
    l = jnp.eye(3)

    for _ in range(PARAMS["NO_ITERS_ECKART"]):
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
        kxy, kxz, kyz = inv_umat @ rhs
        kappa = jnp.array(
            [
                [0.0, kxy, kxz],
                [-kxy, 0.0, kyz],
                [-kxz, -kyz, 0.0],
            ]
        )
        exp_kappa = _expm_pade(-kappa)
        l = exp_kappa + kappa
    return exp_kappa, kappa


def eckart_kappa_impl(xyz, xyz_ref, masses, **kw):
    exp_kappa, _ = _solve_eckart(xyz, xyz_ref, masses, **kw)
    return exp_kappa


eckart_kappa_p.def_impl(eckart_kappa_impl)
eckart_kappa_p.multiple_results = False


@jax.jit
def eckart_kappa_jvp(primals, tangents, **kw):
    xyz, xyz_ref, masses = primals
    dxyz, dxyz_ref, dmasses = tangents

    exp_kappa, kappa = _solve_eckart(xyz, xyz_ref, masses, **kw)

    u = jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * xyz[:, None, :], axis=0)
    du = jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * dxyz[:, None, :], axis=0)

    umat = jnp.array(
        [
            [u[0, 0] + u[1, 1], u[1, 2], -u[0, 2]],
            [u[2, 1], u[0, 0] + u[2, 2], u[0, 1]],
            [-u[2, 0], u[1, 0], u[1, 1] + u[2, 2]],
        ]
    )
    inv_umat = inv(umat)

    dl = jnp.zeros((3, 3))
    dexp_kappa = jnp.zeros((3, 3))
    rhs = exp_kappa @ du.T - du @ exp_kappa.T

    for _ in range(PARAMS["NO_ITERS_ECKART"]):
        rhs_ = rhs + dl @ u.T - u @ dl.T
        dkxy, dkxz, dkyz = inv_umat @ jnp.array([rhs_[0, 1], rhs_[0, 2], rhs_[1, 2]])
        dkappa = jnp.array(
            [
                [0.0, dkxy, dkxz],
                [-dkxy, 0.0, dkyz],
                [-dkxz, -dkyz, 0.0],
            ]
        )
        dexp_kappa = _expm_taylor_squaring([-kappa, -dkappa])[1]
        dl = dexp_kappa + dkappa

    return exp_kappa, dexp_kappa


ad.primitive_jvps[eckart_kappa_p] = eckart_kappa_jvp


def eckart_kappa_abstact_eval(xyz, xyz_ref, masses, **kw):
    return ShapedArray((3, 3), xyz.dtype)


eckart_kappa_p.def_abstract_eval(eckart_kappa_abstact_eval)


def eckart_kappa_lowering(ctx, *ar, **kw):
    return mlir.lower_fun(
        lambda *ar, **kw: _solve_eckart(*ar, **kw)[0], multiple_results=False
    )(ctx, *ar, **kw)


mlir.register_lowering(eckart_kappa_p, eckart_kappa_lowering)


def eckart_kappa_batch_rule(args, dims):
    xyz, xyz_ref, masses = args
    dim1, dim2, dim3 = dims
    assert dim1 == 0
    assert dim2 is None
    assert dim3 is None
    out = jax.vmap(lambda x: eckart_kappa_impl(x, xyz_ref, masses))(xyz)
    return out, 0


batching.primitive_batchers[eckart_kappa_p] = eckart_kappa_batch_rule


@jax.jit
def eckart_kappa_taylor_rule(primals_in, series_in, **kw):
    xyz, xyz_ref, masses = primals_in
    dxyz, dxyz_ref, dmasses = series_in

    exp_kappa, kappa = _solve_eckart(xyz, xyz_ref, masses, **kw)

    u = jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * xyz[:, None, :], axis=0)
    du = [u] + [
        jnp.sum(masses[:, None, None] * xyz_ref[:, :, None] * elem[:, None, :], axis=0)
        for elem in dxyz
    ]

    umat = jnp.array(
        [
            [u[0, 0] + u[1, 1], u[1, 2], -u[0, 2]],
            [u[2, 1], u[0, 0] + u[2, 2], u[0, 1]],
            [-u[2, 0], u[1, 0], u[1, 1] + u[2, 2]],
        ]
    )
    inv_umat = inv(umat)

    dkappa = [kappa] + [None] * len(dxyz)
    dexp_kappa = [exp_kappa] + [None] * len(dxyz)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(1, len(du)):

        rhs = dexp_kappa[0] @ du[k].T - du[k] @ dexp_kappa[0].T

        if k > 1:
            rhs2 = fact(k) * sum(
                scale(k, m) * dexp_kappa[m] @ du[k - m].T for m in range(1, k)
            )
            rhs += rhs2 - rhs2.T

        dl = jnp.zeros((3, 3))
        for _ in range(PARAMS["NO_ITERS_ECKART"]):
            rhs_ = rhs + (dl @ du[0].T - du[0] @ dl.T)
            dkxy, dkxz, dkyz = inv_umat @ jnp.array(
                [rhs_[0, 1], rhs_[0, 2], rhs_[1, 2]]
            )
            dkappa[k] = jnp.array(
                [
                    [0.0, dkxy, dkxz],
                    [-dkxy, 0.0, dkyz],
                    [-dkxz, -dkyz, 0.0],
                ]
            )
            dexp_kappa[k] = _expm_taylor_squaring(
                [-elem for elem in dkappa[: k + 1]],
            )[k]
            dl = dexp_kappa[k] + dkappa[k]

    primal_out, *series_out = dexp_kappa
    return primal_out, series_out


jet.jet_rules[eckart_kappa_p] = eckart_kappa_taylor_rule


@jax.jit
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


@jax.jit
def _matprod_taylor(a, b):
    ab = [None] * len(a)

    def scale(k, j):
        return 1.0 / (fact(k - j) * fact(j))

    for k in range(len(a)):
        ab[k] = fact(k) * sum(scale(k, m) * a[m] @ b[k - m] for m in range(k + 1))
    return ab


@jax.jit
def _expm_taylor(a, k):
    if k == 0:
        at = jnp.eye(len(a[k])) + a[k]
    else:
        at = a[k]
    ap = _matprod_taylor(a, a)
    fac = 0.5
    at = at + ap[k] * fac
    for i in range(3, PARAMS["EXP_TAYLOR_ORDER"]):
        ap = _matprod_taylor(ap, a)
        fac = fac / i
        at = at + ap[k] * fac
    return at


@jax.jit
def _expm_taylor_squaring(a):
    no_sq = PARAMS["EXP_TAYLOR_SQUARING"]
    sq = 1 / 2**no_sq
    a_sq = [elem * sq for elem in a]
    at = [elem for elem in a_sq]
    at[0] += jnp.eye(len(a[0]))
    ap = _matprod_taylor(a_sq, a_sq)
    fac = 0.5
    at = [el1 + el2 * fac for el1, el2 in zip(at, ap)]
    for i in range(3, PARAMS["EXP_TAYLOR_ORDER"]):
        ap = _matprod_taylor(ap, a_sq)
        fac = fac / i
        at = [el1 + el2 * fac for el1, el2 in zip(at, ap)]
    # squaring
    for _ in range(no_sq):
        at = _matprod_taylor(at, at)
    return at
