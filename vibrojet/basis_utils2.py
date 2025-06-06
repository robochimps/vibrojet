from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
import time

from numpy.polynomial.hermite import hermder, hermgauss, hermval
from numpy.polynomial.legendre import legder, leggauss, legval
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
        r_to_y_pes: Callable[[np.ndarray], jnp.ndarray],
        r_to_y_gmat: Callable[[np.ndarray], jnp.ndarray],
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
        
        y_pes = r_to_y_pes(r) 
        y_pow_pes = y_pes[:, None] ** np.array(deriv_ind)[None, :] 

        y_gmat = r_to_y_gmat(r) 
        y_pow_gmat = y_gmat[:, None] ** np.array(deriv_ind)[None, :] 
        
        psi = hermite(x, np.array(self.bas_ind))
        dpsi = hermite_deriv(x, np.array(self.bas_ind)) / dr_dx[:, None]

        me_pes = jnp.einsum("gi,gt,gj,g->ijt", psi, y_pow_pes, psi, w)
        me = jnp.einsum("gi,gt,gj,g->ijt", psi, y_pow_gmat, psi, w)
        dme_r = jnp.einsum("gi,gt,gj,g->ijt", psi, y_pow_gmat, dpsi, w)
        dme_l = -jnp.einsum("gi,gt,gj,g->jit", psi, y_pow_gmat, dpsi, w)
        d2me = -jnp.einsum("gi,gt,gj,g->ijt", dpsi, y_pow_gmat, dpsi, w)
        
        self.me_pes = me_pes.reshape(-1, len(deriv_ind))
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

def legendre(x, l, m):
    l = np.append(l,[np.arange(m)+l[-1]+1])
    sin_x = np.sin(x)
    coef = np.diag(np.sqrt((2*l+1)*factorial(l-m) / (2*factorial(l+m))))
    f = (-1 * sin_x)**m * legval(np.asarray(np.cos(x)), legder(coef, m)) * jnp.sqrt(sin_x)
    return f[m:,:].T

def legendre_deriv(x, l, m):
    l = np.append(l,[np.arange(m)+l[-1]+1])
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    coef = np.diag(np.sqrt((2*l+1)*factorial(l-m) / (2*factorial(l+m))))
    half = 0.5
    term1 = legval(cos_x, legder(coef, m)) * (m+half) * sin_x**(m-half) * cos_x
    term2 = -legval(cos_x, legder(coef, m+1)) * sin_x**(m+3*half)
    df = (-1)**m * (term1 + term2)
    return df[m:,:].T

def leggauss2(n):
    xpre, wpre = leggauss(n)
    x = jnp.arccos(xpre)#x = (xpre+1)*jnp.pi/2
    w = wpre / np.sqrt(1 - xpre**2)#w = wpre * (np.pi/2)
    return x,w

class AssociateLegendreBasis:
    def __init__(
        self,
        icoo: int,
        nbas: int,
        npoints: int,
        x_to_r: Callable[[np.ndarray], jnp.ndarray],
        r_to_y_pes: Callable[[np.ndarray], jnp.ndarray],
        r_to_y_gmat: Callable[[np.ndarray], jnp.ndarray],
        r0: float,
        deriv_ind: List[int],
        m = 0, #Default m value
    ):

        self.bas_ind = np.arange(nbas)
        self.coo_ind = (icoo,)
        x, w = leggauss2(npoints)
        
        r = x_to_r(x)
        dr_dx = jax.vmap(jax.grad(x_to_r), in_axes=0)(x)
        dr = r - r0
        r_pow = dr[:, None] ** np.array(deriv_ind)[None, :]
        
        y_pes = r_to_y_pes(r) 
        y_pow_pes = y_pes[:, None] ** np.array(deriv_ind)[None, :] 

        y_gmat = r_to_y_gmat(r) 
        y_pow_gmat = y_gmat[:, None] ** np.array(deriv_ind)[None, :] 
        
        psi = legendre(x, np.array(self.bas_ind), m)
        dpsi = legendre_deriv(x, np.array(self.bas_ind), m) / dr_dx[:, None]

        me_pes = jnp.einsum("gi,gt,gj,g->ijt", psi, y_pow_pes, psi, w)
        me = jnp.einsum("gi,gt,gj,g->ijt", psi, y_pow_gmat, psi, w)
        dme_r = jnp.einsum("gi,gt,gj,g->ijt", psi, y_pow_gmat, dpsi, w)
        dme_l = -jnp.einsum("gi,gt,gj,g->jit", psi, y_pow_gmat, dpsi, w)
        d2me = -jnp.einsum("gi,gt,gj,g->ijt", dpsi, y_pow_gmat, dpsi, w)
        
        self.me_pes = me_pes.reshape(-1, len(deriv_ind))
        self.me = me.reshape(-1, len(deriv_ind))
        self.dme = {icoo: dme_r.reshape(-1, len(deriv_ind))}
        self.dme_r = {icoo: dme_r.reshape(-1, len(deriv_ind))}
        self.dme_l = {icoo: dme_l.reshape(-1, len(deriv_ind))}
        self.d2me = {(icoo, icoo): d2me.reshape(-1, len(deriv_ind))}


def fourier(x, n):
    f = np.sin(np.ceil(n[None, :]/2) * x[:,None] + np.pi/2 * np.mod(n[None,:] + 1, 2))/ np.sqrt(np.pi)
    f[:,0] /= np.sqrt(2)
    return f

def fourier_deriv(x, n):
    f = np.ceil(n[None, :]/2) * np.cos(np.ceil(n[None, :]/2) * x[:,None] + np.pi/2 * np.mod(n[None,:] + 1, 2))/ np.sqrt(np.pi)
    return f

def fourgauss(N):
    w = np.ones(N) * 2*np.pi/N
    x = np.linspace(0, 2*np.pi - 2*np.pi/N, N)
    return x, w

class FourierBasis:
    def __init__(
        self,
        icoo: int,
        nbas: int,
        npoints: int,
        x_to_r: Callable[[np.ndarray], jnp.ndarray],
        r_to_y_pes: Callable[[np.ndarray], jnp.ndarray],
        r_to_y_gmat: Callable[[np.ndarray], jnp.ndarray],
        r0: float,
        deriv_ind: List[int],
    ):
        self.bas_ind = np.arange(nbas)
        self.coo_ind = (icoo,)
        x, w = fourgauss(npoints)
        
        r = x_to_r(x)
        dr_dx = jax.vmap(jax.grad(x_to_r), in_axes=0)(x)
        dr = r - r0
        r_pow = dr[:, None] ** np.array(deriv_ind)[None, :]
        
        y_pes = r_to_y_pes(r) 
        y_pow_pes = y_pes[:, None] ** np.array(deriv_ind)[None, :] 

        y_gmat = r_to_y_gmat(r) 
        y_pow_gmat = y_gmat[:, None] ** np.array(deriv_ind)[None, :] 
        
        psi = fourier(x, np.array(self.bas_ind))
        dpsi = fourier_deriv(x, np.array(self.bas_ind)) / dr_dx[:, None]

        me_pes = jnp.einsum("gi,gt,gj,g->ijt", psi, y_pow_pes, psi, w)
        me = jnp.einsum("gi,gt,gj,g->ijt", psi, y_pow_gmat, psi, w)
        dme_r = jnp.einsum("gi,gt,gj,g->ijt", psi, y_pow_gmat, dpsi, w)
        dme_l = -jnp.einsum("gi,gt,gj,g->jit", psi, y_pow_gmat, dpsi, w)
        d2me = -jnp.einsum("gi,gt,gj,g->ijt", dpsi, y_pow_gmat, dpsi, w)
        
        self.me_pes = me_pes.reshape(-1, len(deriv_ind))
        self.me = me.reshape(-1, len(deriv_ind))
        self.dme = {icoo: dme_r.reshape(-1, len(deriv_ind))}
        self.dme_r = {icoo: dme_r.reshape(-1, len(deriv_ind))}
        self.dme_l = {icoo: dme_l.reshape(-1, len(deriv_ind))}
        self.d2me = {(icoo, icoo): d2me.reshape(-1, len(deriv_ind))}


class ContrBasis:
    def __init__(
        self,
        coupl_ind: List[int],
        bas_list,
        bas_select: Callable[[List[int]], bool],
        Gmat_coefs: np.ndarray,
        poten_coefs: np.ndarray,
        emax: float = 1e8,
        batch_size: int = 0,
        store_int = True,
    ):
        
        ind = [
            bas.bas_ind if ibas in coupl_ind else bas.bas_ind[0:1]
            for ibas, bas in enumerate(bas_list)
        ]
        ind, m_ind = next(generate_prod_ind(ind, select=bas_select))

        print('Number of states:',len(ind))
        
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

        batch_size_ = nbas**2 if batch_size is 0 else batch_size
        n_batches = int(np.ceil(nbas**2 / batch_size_))

        vme = np.zeros(nbas**2)
        gme = np.zeros(nbas**2)
        coupl_bas_list = [bas_list[i] for i in coupl_ind]
        
        for ibatch in range(n_batches):
            start_time = time.time()
            print('batch no:',ibatch,' out of:',int(jnp.ceil(nbas**2 / batch_size_)))
            index_batch = jnp.arange(ibatch * batch_size_, np.min([(ibatch+1) * batch_size_, nbas**2]))
            m_ind_flat_batch = []
            for z in range(len(m_ind_flat)):
                m_ind_flat_batch.append(m_ind_flat[z][index_batch])
            
            me_pes = _me_pes(
                coupl_bas_list,
                [m_ind_flat_batch[i] for i in coupl_ind],
            )
    
            # define and solve reduced-mode eigenvalue problem
            # for basis sets that are involved in contraction
    
            # matrix elements of expansion terms for basis sets
            # that are not involved in contraction
            me0_pes = jnp.prod(
                jnp.asarray(
                    [
                        bas.me_pes[ind]
                        for i, (bas, ind) in enumerate(zip(bas_list, m_ind_flat_batch))
                        if i not in coupl_ind
                    ]
                ),
                axis=0,
            )
            
            me0 = jnp.prod(
                jnp.asarray(
                    [
                        bas.me[ind]
                        for i, (bas, ind) in enumerate(zip(bas_list, m_ind_flat_batch))
                        if i not in coupl_ind
                    ]
                ),
                axis=0,
            )
    
            # potential matrix elements
            vme[index_batch] = (me_pes * me0_pes) @ poten_coefs
    
            # keo matrix elements
            # gme = 0
            # for icoo in self.coo_ind:
            #     for jcoo in self.coo_ind:
            #         gme += (d2me[(icoo, jcoo)] * me0) @ Gmat_coefs[:, icoo, jcoo]
            gme[index_batch] = compute_gme(bas_list, m_ind_flat_batch, coupl_ind, coo_ind, me0, Gmat_coefs)
            end_time = time.time()
            print('iteration time:',np.round(end_time-start_time,2))

        hme = -0.5 * gme + vme
        e, v = jnp.linalg.eigh(hme.reshape(nbas, nbas))
        print(e[0:10])
        v_ind = jnp.where(e - e[0] <= emax)[0]
        nstates = len(v_ind)
        v = v[:, v_ind]
        self.enr = e[v_ind]
        
        # transform matrix elements to eigenbasis
        if store_int:    
            print("Saving integrals...")
        
            nbas = len(v)
            batch_size_ = nbas**2 if batch_size == 0 else batch_size
            n_batches = int(np.ceil(nbas**2 / batch_size_))
        
            me = jnp.zeros((nstates * nstates, len(poten_coefs)))
            me_pes = jnp.zeros_like(me)
            dme_r = None
            dme_l = None
            d2me = None
            
            m_ind_flat = [jnp.asarray(m) for m in m_ind_flat]
            input_bas = [bas_list[i] for i in coupl_ind]
            
            for ibatch in range(n_batches):
                start_time = time.time()
                print(f'batch no: {ibatch} out of: {n_batches}')
        
                index_batch = np.arange(ibatch * batch_size_, min((ibatch + 1) * batch_size_, nbas**2))
        
                index_bra = index_batch // nbas
                index_ket = index_batch % nbas
                v_bra = v[index_bra, :]
                v_ket = v[index_ket, :]
                v_braket = jnp.einsum('qi,qj->qij', v_bra, v_ket).reshape((len(index_batch), -1))

                m_ind_flat_batch = [m[index_batch] for m in m_ind_flat]
                input_m_ind = [m_ind_flat_batch[i] for i in coupl_ind]

                me_batch = _me(input_bas, input_m_ind)
                me_pes_batch = _me_pes(input_bas, input_m_ind)
                dme_r_batch = _dme_r(input_bas, input_m_ind, coo_ind)
                dme_l_batch = _dme_l(input_bas, input_m_ind, coo_ind)
                d2me_batch = _d2me(input_bas, input_m_ind, coo_ind)
        
                # Contract into (nstates², n_terms)
                me += jnp.einsum("qi,qt->it", v_braket, me_batch)
                me_pes += jnp.einsum("qi,qt->it", v_braket, me_pes_batch)
        
                einsum_r = jnp.einsum("qi,aqt->ait", v_braket, dme_r_batch)
                dme_r = einsum_r if dme_r is None else dme_r + einsum_r
        
                einsum_l = jnp.einsum("qi,aqt->ait", v_braket, dme_l_batch)
                dme_l = einsum_l if dme_l is None else dme_l + einsum_l
        
                einsum_2 = jnp.einsum("qi,abqt->abit", v_braket, d2me_batch)
                d2me = einsum_2 if d2me is None else d2me + einsum_2
        
                end_time = time.time()
                print('iteration time compute objects to store:', np.round(end_time - start_time, 2))

        
            # Store transformed eigenbasis
            self.bas_ind = jnp.arange(nstates)
            start_time = time.time()
            self.me = me
            self.me_pes = me_pes
            self.dme_r = {k: dme_r[k] for k in coo_ind}
            self.dme_l = {k: dme_l[k] for k in coo_ind}
            self.d2me = {(k,j): d2me[k,j] for k in coo_ind for j in coo_ind}
            end_time = time.time()
            print('Storing time:',np.round(end_time-start_time,2))
            
def compute_gme(bas_list, m_ind_flat, coupl_ind, coo_ind, me0, Gmat_coefs):
    gme = 0
    for icoo in coo_ind:
        for jcoo in coo_ind:
            val_prod = None
            for i in coupl_ind:
                bas = bas_list[i]
                ind = m_ind_flat[i]
                if (icoo, jcoo) in bas.d2me:
                    val = bas.d2me[(icoo, jcoo)][ind]
                elif icoo in bas.coo_ind and jcoo in bas.coo_ind:
                    val = bas.dme_l[icoo][ind] * bas.dme_r[jcoo][ind]
                elif icoo in bas.coo_ind:
                    val = bas.dme_l[icoo][ind]
                elif jcoo in bas.coo_ind:
                    val = bas.dme_r[jcoo][ind]
                else:
                    val = bas.me[ind]
                val_prod = val if val_prod is None else val_prod * val
            gme += (val_prod * me0) @ Gmat_coefs[:, icoo, jcoo]
    return gme

def _me_pes_precomputed(me_pes_list, bas_m_ind):
    me = None
    for me_array, ind in zip(me_pes_list, bas_m_ind):
        val = me_array[ind]  # Now safe, no object access
        me = val if me is None else me * val
    return me

def _me_pes(coupl_bas, bas_m_ind):
    me = None
    for bas, ind in zip(coupl_bas, bas_m_ind):
        val = bas.me_pes[ind]
        me = val if me is None else me * val
    return me

def _me(coupl_bas, bas_m_ind):
    me = None
    for bas, ind in zip(coupl_bas, bas_m_ind):
        val = bas.me[ind]
        me = val if me is None else me * val
    return me

def _dme_l(coupl_bas, bas_m_ind, coo_ind):
    dme = []#{}
    for icoo in coo_ind:
        val_prod = None
        for bas, ind in zip(coupl_bas, bas_m_ind):
            if icoo in bas.coo_ind:
                val = bas.dme_l[icoo][ind]
            else:
                val = bas.me[ind]
            val_prod = val if val_prod is None else val_prod * val
        #dme[icoo] = val_prod
        dme.append(val_prod)
    return jnp.array(dme)

def _dme_r(coupl_bas, bas_m_ind, coo_ind):
    dme = [] #{} 
    for icoo in coo_ind:
        val_prod = None
        for bas, ind in zip(coupl_bas, bas_m_ind):
            if icoo in bas.coo_ind:
                val = bas.dme_r[icoo][ind]
            else:
                val = bas.me[ind]
            val_prod = val if val_prod is None else val_prod * val
        #dme[icoo] = val_prod
        dme.append(val_prod)
    return jnp.array(dme)


def _d2me(coupl_bas, bas_m_ind, coo_ind):
    #d2me = {}
    d2me = []
    for icoo in coo_ind:
        for jcoo in coo_ind:
            #for_bas = []
            val_prod = None
            for bas, ind in zip(coupl_bas, bas_m_ind):
                if (icoo, jcoo) in bas.d2me:
                    val = bas.d2me[(icoo, jcoo)][ind]
                elif icoo in bas.coo_ind and jcoo in bas.coo_ind:
                    val = bas.dme_l[icoo][ind] * bas.dme_r[jcoo][ind]
                elif icoo in bas.coo_ind:
                    val = bas.dme_l[icoo][ind]
                elif jcoo in bas.coo_ind:
                    val = bas.dme_r[jcoo][ind]
                else:
                    val = bas.me[ind]
                val_prod = val if val_prod is None else val_prod * val
            #d2me[(icoo, jcoo)] = val_prod
            d2me.append(val_prod)
    return jnp.array(d2me).reshape((len(coo_ind),len(coo_ind),*jnp.shape(d2me[0])))

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
