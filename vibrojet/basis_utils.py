from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermder, hermgauss, hermval
from scipy.special import factorial
from itertools import product

jax.config.update("jax_enable_x64", True)


def hermite(x, n):
    sqsqpi = np.sqrt(np.sqrt(np.pi))
    c = np.diag(1.0 / np.sqrt(2.0**n * factorial(n)) / sqsqpi)
    f = hermval(np.asarray(x), c) * np.exp(-(x**2) / 2)
    return f.T


def hermite_deriv(x, n):
    sqsqpi = np.sqrt(np.sqrt(np.pi))
    c = np.diag(1.0 / np.sqrt(2.0**n * factorial(n)) / sqsqpi)
    h = hermval(np.asarray(x), c)
    dh = hermval(np.asarray(x), hermder(c, m=1))
    f = (dh - h * x) * np.exp(-(x**2) / 2)
    return f.T


class HermiteBasis:
    def __init__(
        self,
        icoo: int,
        nbas: int,
        npoints: int,
        x_to_r: Callable[[np.ndarray], jnp.ndarray],
        r0: float,
        deriv_ind: List[int],
    ):
        self.bas_ind = np.arange(nbas)
        self.coo_ind = (icoo,)
        x, w = hermgauss(npoints)
        w /= np.exp(-(x**2))
        r = x_to_r(x)
        dr_dx = jax.vmap(jax.grad(x_to_r), in_axes=0)(x)
        dr = r - r0
        r_pow = dr[:, None] ** np.array(deriv_ind)[None, :]
        psi = hermite(x, np.array(self.bas_ind))
        dpsi = hermite_deriv(x, np.array(self.bas_ind)) / dr_dx[:, None]
        me = jnp.einsum("gi,gt,gj,g->tij", psi, r_pow, psi, w)
        dme = jnp.einsum("gi,gt,gj,g->tij", psi, r_pow, dpsi, w)
        d2me = jnp.einsum("gi,gt,gj,g->tij", dpsi, r_pow, dpsi, w)
        self.me = me.reshape(len(deriv_ind), -1)
        self.dme = {icoo: dme.reshape(len(deriv_ind), -1)}
        self.d2me = {(icoo, icoo): d2me.reshape(len(deriv_ind), -1)}


class ContrBasis:
    def __init__(
        self,
        coupl_ind: List[int],
        bas_list,
        bas_select: Callable[[List[int]], bool],
        Gmat_coefs: np.ndarray,
        poten_coefs: np.ndarray,
    ):
        ind = [
            bas.bas_ind if ibas in coupl_ind else bas.bas_ind[0:1]
            for ibas, bas in enumerate(bas_list)
        ]
        ind, m_ind = next(generate_prod_ind(ind, select=bas_select))

        # basis indices for flattened matrix element arrays
        # i.e, if B = A[:, m_ind[ibas], m_ind[ibas]]
        #    then B.reshape(len(B), -1) = A.reshape(len(A), -1)[:, m_ind_flat[ibas]]
        m_ind_flat = []
        for m_i, bas in zip(m_ind, bas_list):
            i1, i2 = np.meshgrid(m_i, m_i, indexing="ij")
            flat_ind = i1 * len(bas.bas_ind) + i2
            m_ind_flat.append(flat_ind.reshape(-1))
            #   ... without numpy
            # n = len(bas.bas_ind)
            # flat_ind = [i * n + j for i in m_i for j in m_i]
            # m_ind_flat.append(flat_ind)

        coupl_bas = [bas_list[ibas] for ibas in coupl_ind]
        coupl_m_ind_flat = [m_ind_flat[ibas] for ibas in coupl_ind]

        coo_ind = [coo_ind for bas in coupl_bas for coo_ind in bas.coo_ind]
        assert len(set(coo_ind)) == len(
            coo_ind
        ), "Input basis sets have overlapping coordinates"

        self.bas_ind = np.arange(len(ind))
        self.coo_ind = coo_ind

        me = _me(coupl_bas, coupl_m_ind_flat)
        dme = _dme(coupl_bas, coupl_m_ind_flat, coo_ind)
        d2me = _d2me(coupl_bas, coupl_m_ind_flat, coo_ind)

        # define and solve reduced-mode eigenvalue problem
        # for basis sets that are involved in contraction

        # matrix elements of expansion terms for basis sets
        # that are not involved in contraction
        me0 = jnp.prod(
            jnp.asarray(
                [
                    bas.me[:, ind]
                    for i, (bas, ind) in enumerate(zip(bas_list, m_ind_flat))
                    if i not in coupl_ind
                ]
            ),
            axis=0,
        )

        # potential matrix elements
        me_ = me * me0
        vme = me_.T @ poten_coefs

        # keo matrix elements
        gme = 0
        for icoo in self.coo_ind:
            for jcoo in self.coo_ind:
                me_ = d2me[(icoo, jcoo)] * me0
                gme += me_.T @ Gmat_coefs[:, icoo, jcoo]

        hme = 0.5 * gme + vme
        n = len(self.bas_ind)
        e, v = np.linalg.eigh(hme.reshape(n, n))
        print(e[0], e[0:10] - e[0])

        # transform matrix elements to eigenbasis

        n = v.shape[0]
        nt = me.shape[0]
        self.me = (v.T @ me.reshape(-1, n, n) @ v).reshape(nt, -1)
        self.dme = {
            key: (v.T @ val.reshape(-1, n, n) @ v).reshape(nt, -1)
            for key, val in dme.items()
        }
        self.d2me = {
            key: (v.T @ val.reshape(-1, n, n) @ v).reshape(nt, -1)
            for key, val in d2me.items()
        }


def _me(coupl_bas, bas_m_ind):
    me = jnp.prod(
        jnp.asarray([bas.me[:, ind] for bas, ind in zip(coupl_bas, bas_m_ind)]),
        axis=0,
    )
    return me


def _dme(coupl_bas, bas_m_ind, coo_ind):
    dme = {}
    for icoo in coo_ind:
        me = jnp.prod(
            jnp.asarray(
                [
                    bas.dme[icoo][:, ind] if icoo in bas.coo_ind else bas.me[:, ind]
                    for bas, ind in zip(coupl_bas, bas_m_ind)
                ]
            ),
            axis=0,
        )
        dme[icoo] = me
    return dme


def _d2me(coupl_bas, bas_m_ind, coo_ind):
    d2me = {}
    for icoo in coo_ind:
        for jcoo in coo_ind:
            me = jnp.prod(
                jnp.asarray(
                    [
                        (
                            bas.d2me[(icoo, jcoo)][:, ind]
                            if icoo in bas.coo_ind and jcoo in bas.coo_ind
                            else (
                                -bas.dme[icoo][:, ind]
                                if icoo in bas.coo_ind
                                else (
                                    bas.dme[jcoo][:, ind]
                                    if jcoo in bas.coo_ind
                                    else bas.me[:, ind]
                                )
                            )
                        )
                        for bas, ind in zip(coupl_bas, bas_m_ind)
                    ]
                ),
                axis=0,
            )
            d2me[(icoo, jcoo)] = me
    return d2me

def generate_prod_ind(
    indices: List[List[int]],
    select: Callable[[List[int]], bool] = lambda _: True,
    batch_size: int = None,
):
    
    no_elem = np.array([len(elem) for elem in indices])
    tot_size = np.prod(no_elem)
    list_ = indices[0]
    for i in range(1, len(indices)):
        list_ = list(product(list_,indices[i]))
        list_ = [tuple(a) + (b,) if isinstance(a, tuple) else (a, b) for a, b in list_]
        list_ = [elem for elem in list_ if select(elem)]
    yield np.array(list_), np.array(list_).T #indices_out, #multi_ind

#def generate_prod_ind(
#    indices: List[List[int]],
#    select: Callable[[List[int]], bool] = lambda _: True,
#    batch_size: int = None,
#):
#    no_elem = np.array([len(elem) for elem in indices])
#    tot_size = np.prod(no_elem)
#    if batch_size is None:
#        batch_size = tot_size
#    no_batches = (tot_size + batch_size - 1) // batch_size

#    for ibatch in range(no_batches):
#        start_ind = ibatch * batch_size
#        end_ind = np.minimum(start_ind + batch_size, tot_size)
#        batch_ind = np.arange(start_ind, end_ind)
#        multi_ind = np.array(np.unravel_index(batch_ind, no_elem))
#        indices_out = np.array(
#            [indices[icoo][multi_ind[icoo, :]] for icoo in range(len(indices))]
#        ).T
#        select_ind = np.where(np.asarray([select(elem) for elem in indices_out]))
#        yield indices_out[select_ind], multi_ind[:, select_ind[0]]
