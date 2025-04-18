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

        me = jnp.einsum("gi,gt,gj,g->ijt", psi, r_pow, psi, w)
        dme_r = jnp.einsum("gi,gt,gj,g->ijt", psi, r_pow, dpsi, w)
        dme_l = -jnp.einsum("gi,gt,gj,g->jit", psi, r_pow, dpsi, w)
        d2me = jnp.einsum("gi,gt,gj,g->ijt", dpsi, r_pow, dpsi, w)
        self.me = me.reshape(-1, len(deriv_ind))
        self.dme = {icoo: dme_r.reshape(-1, len(deriv_ind))}
        self.dme_r = {icoo: dme_r.reshape(-1, len(deriv_ind))}
        self.dme_l = {icoo: dme_l.reshape(-1, len(deriv_ind))}
        self.d2me = {(icoo, icoo): d2me.reshape(-1, len(deriv_ind))}

        # compute derivative matrix elements as
        #    <d/dx x**i> = <x**(i-1)> + <x**i d/dx>
        # deriv_ind_min_one = np.array(deriv_ind) - 1
        # r_pow_min_one = jnp.where(
        #     deriv_ind_min_one >= 0,
        #     dr[:, None] ** deriv_ind_min_one[None, :],
        #     dr[:, None] * np.zeros_like(deriv_ind_min_one)[None, :],
        # )
        # dme_l = jnp.einsum("gi,gt,gj,g->ijt", psi, r_pow_min_one, psi, w) + jnp.einsum(
        #     "gi,gt,gj,g->ijt", psi, r_pow, dpsi, w
        # )


class ContrBasis:
    def __init__(
        self,
        coupl_ind: List[int],
        bas_list,
        bas_select: Callable[[List[int]], bool],
        Gmat_coefs: np.ndarray,
        poten_coefs: np.ndarray,
        emax: float = 1e8,
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

        coo_ind = [coo_ind for i in coupl_ind for coo_ind in bas_list[i].coo_ind]
        assert len(set(coo_ind)) == len(
            coo_ind
        ), "Input basis sets have overlapping coordinates"

        self.coo_ind = coo_ind
        nbas = len(ind)

        me = _me(
            [bas_list[i] for i in coupl_ind],
            [m_ind_flat[i] for i in coupl_ind],
        )
        dme_r = _dme(
            [bas_list[i] for i in coupl_ind],
            [m_ind_flat[i] for i in coupl_ind],
            coo_ind,
        )
        dme_l = {
            key: -val.reshape(nbas, nbas, -1)
            .transpose(1, 0, 2)
            .reshape(nbas * nbas, -1)
            for key, val in dme_r.items()
        }
        d2me = _d2me(
            [bas_list[i] for i in coupl_ind],
            [m_ind_flat[i] for i in coupl_ind],
            coo_ind,
        )

        # define and solve reduced-mode eigenvalue problem
        # for basis sets that are involved in contraction

        # matrix elements of expansion terms for basis sets
        # that are not involved in contraction
        me0 = jnp.prod(
            jnp.asarray(
                [
                    bas.me[ind]
                    for i, (bas, ind) in enumerate(zip(bas_list, m_ind_flat))
                    if i not in coupl_ind
                ]
            ),
            axis=0,
        )

        # potential matrix elements
        vme = (me * me0) @ poten_coefs

        # keo matrix elements
        gme = 0
        for icoo in self.coo_ind:
            for jcoo in self.coo_ind:
                gme += (d2me[(icoo, jcoo)] * me0) @ Gmat_coefs[:, icoo, jcoo]

        hme = 0.5 * gme + vme
        e, v = jnp.linalg.eigh(hme.reshape(nbas, nbas))
        v_ind = jnp.where(e <= emax)[0]
        v = v[:, v_ind]
        self.enr = e[v_ind]
        nstates = len(v_ind)
        self.bas_ind = np.arange(nstates)

        # transform matrix elements to eigenbasis

        self.me = jnp.einsum(
            "pi,pqt,qj->ijt", v, me.reshape(nbas, nbas, -1), v
        ).reshape(nstates * nstates, -1)

        self.dme_r = {
            key: jnp.einsum(
                "pi,pqt,qj->ijt", v, val.reshape(nbas, nbas, -1), v
            ).reshape(nstates * nstates, -1)
            for key, val in dme_r.items()
        }

        self.dme_l = {
            key: jnp.einsum(
                "pi,pqt,qj->ijt", v, val.reshape(nbas, nbas, -1), v
            ).reshape(nstates * nstates, -1)
            for key, val in dme_l.items()
        }

        self.d2me = {
            key: jnp.einsum(
                "pi,pqt,qj->ijt", v, val.reshape(nbas, nbas, -1), v
            ).reshape(nstates * nstates, -1)
            for key, val in d2me.items()
        }


def _me(coupl_bas, bas_m_ind):
    me = jnp.prod(
        jnp.asarray([bas.me[ind] for bas, ind in zip(coupl_bas, bas_m_ind)]),
        axis=0,
    )
    return me


def _dme(coupl_bas, bas_m_ind, coo_ind):
    dme = {}
    for icoo in coo_ind:
        me = jnp.prod(
            jnp.asarray(
                [
                    bas.dme_r[icoo][ind] if icoo in bas.coo_ind else bas.me[ind]
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
                            bas.d2me[(icoo, jcoo)][ind]
                            if icoo in bas.coo_ind and jcoo in bas.coo_ind
                            else (
                                bas.dme_l[icoo][ind]
                                if icoo in bas.coo_ind
                                else (
                                    bas.dme_r[jcoo][ind]
                                    if jcoo in bas.coo_ind
                                    else bas.me[ind]
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
