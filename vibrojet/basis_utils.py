from itertools import product
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite import hermder, hermgauss, hermval
from scipy.sparse import csr_array
from scipy.special import factorial

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
        quad_coords: Callable[[jnp.ndarray], jnp.ndarray],
        keo_coords: Callable[[jnp.ndarray], jnp.ndarray],
        poten_coords: Callable[[jnp.ndarray], jnp.ndarray],
        gmat_terms: np.ndarray,
        poten_terms: np.ndarray,
        pseudo_terms: np.ndarray | None = None,
        thresh: float = 1e-8,
    ):
        self.bas_ind = np.arange(nbas)
        self.coo_ind = (icoo,)
        x, w = hermgauss(npoints)
        w /= np.exp(-(x**2))

        # `q` are internal coordinates used in variational solution
        q = quad_coords(jnp.asarray(x))
        dq_dx = jax.vmap(jax.grad(quad_coords), in_axes=0)(x)

        psi = hermite(x, np.array(self.bas_ind))
        dpsi = hermite_deriv(x, np.array(self.bas_ind)) / dq_dx[:, None]

        # potential expansion matrix elements
        #   `poten_coords` are coordinates, functions of `q` in which PES is expanded
        if poten_terms.ndim > 1:
            poten_terms = np.unique(poten_terms[:, icoo])
        else:
            poten_terms = np.unique(poten_terms)
        poten_pow = poten_coords(q)[:, None] ** poten_terms[None, :]
        poten_me = np.einsum("gi,gt,gj,g->ijt", psi, poten_pow, psi, w)
        self.poten_me = poten_me.reshape(-1, len(poten_terms))
        self.poten_terms = poten_terms[:, np.newaxis]

        # G-matrix expansion matrix elements
        #   `keo_coords` are coordinates, functions of `q` in which KEO is expanded
        if gmat_terms.ndim > 1:
            gmat_terms = np.unique(gmat_terms[:, icoo])
        else:
            gmat_terms = np.unique(gmat_terms)
        gmat_pow = keo_coords(q)[:, None] ** gmat_terms[None, :]
        gmat_me = jnp.einsum("gi,gt,gj,g->ijt", psi, gmat_pow, psi, w)
        dme_r = jnp.einsum("gi,gt,gj,g->ijt", psi, gmat_pow, dpsi, w)
        dme_l = -jnp.einsum("gi,gt,gj,g->jit", psi, gmat_pow, dpsi, w)
        d2me = -jnp.einsum("gi,gt,gj,g->ijt", dpsi, gmat_pow, dpsi, w)
        self.gmat_me = gmat_me.reshape(-1, len(gmat_terms))
        self.dme = {icoo: dme_r.reshape(-1, len(gmat_terms))}
        self.dme_r = {icoo: dme_r.reshape(-1, len(gmat_terms))}
        self.dme_l = {icoo: dme_l.reshape(-1, len(gmat_terms))}
        self.d2me = {(icoo, icoo): d2me.reshape(-1, len(gmat_terms))}
        self.gmat_terms = gmat_terms[:, np.newaxis]

        # pseudopotential expansion matrix elements
        self.pseudo_me = None
        if pseudo_terms is not None:
            if pseudo_terms.ndim > 1:
                pseudo_terms = np.unique(pseudo_terms[:, icoo])
            else:
                pseudo_terms = np.unique(pseudo_terms)
            pseudo_pow = keo_coords(q)[:, None] ** np.array(pseudo_terms)[None, :]
            pseudo_me = jnp.einsum("gi,gt,gj,g->ijt", psi, pseudo_pow, psi, w)
            self.pseudo_me = pseudo_me.reshape(-1, len(pseudo_terms))
            self.pseudo_terms = pseudo_terms[:, np.newaxis]


class ContrBasis:
    def __init__(
        self,
        coupl_ind: list[int],
        bas_list,
        bas_select: Callable[[list[int]], bool],
        gmat_terms: np.ndarray,
        gmat_coefs: np.ndarray,
        poten_terms: np.ndarray,
        poten_coefs: np.ndarray,
        pseudo_terms: np.ndarray | None = None,
        pseudo_coefs: np.ndarray | None = None,
        emax: float = 1e8,
        thresh: float = 1e-8,
        store_me: bool = True,
        no_batches: int = 10,
    ):
        if (pseudo_terms is None) != (pseudo_coefs is None):
            raise ValueError(
                "Parameters 'pseudo_terms' and 'pseudo_coefs' must ",
                "both be provided or both left as None.",
            )

        ind = [
            bas.bas_ind if ibas in coupl_ind else np.array([0])
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
        if len(set(coo_ind)) != len(coo_ind):
            raise ValueError("Input basis sets have overlapping coordinates")

        self.coo_ind = coo_ind
        nbas = len(ind)

        # Define set of unique expansion terms (e.g., `self.poten_terms`)
        # spanned by the coordinates of the coupled basis sets.

        # Construct mapping (e.g., `poten_coefs_map`) from the full set
        # of expansion terms (e.g., `poten_terms`) to the unique (reduced)
        # set of terms (e.g., `self.poten_terms`).

        # For the coupled basis sets, construct mapping (e.g., `poten_terms_map`)
        # from the unique set of terms (e.g., `self.poten_terms`) to the unique
        # sets of terms within each basis (e.g., bas_list[i].poten_terms for i in coupl_ind).

        # For the basis sets that are not coupled, construct mapping
        # from the full set of expansion terms (e.g., `poten_terms`).

        self.poten_terms, poten_coefs_map, poten_terms_map = expansion_subspace(
            poten_terms, bas_list, coupl_ind, self.coo_ind, "poten"
        )
        self.gmat_terms, gmat_coefs_map, gmat_terms_map = expansion_subspace(
            gmat_terms, bas_list, coupl_ind, self.coo_ind, "gmat"
        )
        if pseudo_terms is not None:
            self.pseudo_terms, pseudo_coefs_map, pseudo_terms_map = expansion_subspace(
                pseudo_terms, bas_list, coupl_ind, self.coo_ind, "pseudo"
            )

        # Compute matrix elements <0,0,..| q^t |0,0,..> between products
        # of zero-order basis functions, for basis sets that are not coupled.
        # Matrix elements are computed for full set of expansion terms
        # and only one zero-order product function.

        not_coupl_ind = [i for i in range(len(bas_list)) if i not in coupl_ind]

        # check to be safe
        for i in not_coupl_ind:
            if any(m_ind_flat[i] != 0):
                raise ValueError(
                    f"Basis set index {i} is not included in the contraction, "
                    f"but its index array contains non zero-order states: {np.unique(m_ind_flat[i])}. "
                    "Only the |0> function is allowed in non-contracted bases."
                )

        poten_me0 = _poten_me(
            [bas_list[i] for i in not_coupl_ind],
            [m_ind_flat[i][:1] for i in not_coupl_ind],
            [poten_terms_map[i] for i in not_coupl_ind],
        )
        try:
            poten_me0 = poten_me0[0]
        except TypeError:
            poten_me0 = poten_me0  # =1

        gmat_me0 = _gmat_me(
            [bas_list[i] for i in not_coupl_ind],
            [m_ind_flat[i][:1] for i in not_coupl_ind],
            [gmat_terms_map[i] for i in not_coupl_ind],
        )
        try:
            gmat_me0 = gmat_me0[0]
        except TypeError:
            gmat_me0 = gmat_me0  # =1

        if pseudo_terms is not None:
            pseudo_me0 = _pseudo_me(
                [bas_list[i] for i in not_coupl_ind],
                [m_ind_flat[i][:1] for i in not_coupl_ind],
                [pseudo_terms_map[i] for i in not_coupl_ind],
            )
            try:
                pseudo_me0 = pseudo_me0[0]
            except TypeError:
                pseudo_me0 = pseudo_me0  # =1

        # Define and solve reduced-mode eigenvalue problem
        # for basis sets that are involved in contraction

        hme = np.zeros(nbas * nbas, dtype=np.float64)

        # potential matrix elements
        c = poten_coefs * poten_me0
        c = np.array([c[ind].sum() for ind in poten_coefs_map])
        nz_ind = np.where(np.abs(c) > thresh)[0]
        nz_ind_batched = np.array_split(nz_ind, no_batches)
        c_batched = np.array_split(c[nz_ind], no_batches)
        for ibatch, (nz_ind, c) in enumerate(zip(nz_ind_batched, c_batched)):
            me = _poten_me(
                [bas_list[i] for i in coupl_ind],
                [m_ind_flat[i] for i in coupl_ind],
                [poten_terms_map[i][nz_ind] for i in coupl_ind],
            )
            hme += me @ c

        # keo matrix elements
        for icoo in self.coo_ind:
            for jcoo in self.coo_ind:
                c = gmat_coefs[:, icoo, jcoo] * gmat_me0
                c = np.array([c[ind].sum() for ind in gmat_coefs_map])
                nz_ind = np.where(np.abs(c) > thresh)[0]
                nz_ind_batched = np.array_split(nz_ind, no_batches)
                c_batched = np.array_split(c[nz_ind], no_batches)
                for ibatch, (nz_ind, c) in enumerate(zip(nz_ind_batched, c_batched)):
                    me = _d2me_i_j(
                        [bas_list[i] for i in coupl_ind],
                        [m_ind_flat[i] for i in coupl_ind],
                        [gmat_terms_map[i][nz_ind] for i in coupl_ind],
                        icoo,
                        jcoo,
                    )
                    hme -= 0.5 * me @ c

        # pseudopotential matrix elements
        if pseudo_coefs is not None:
            c = pseudo_coefs * pseudo_me0
            c = np.array([c[ind].sum() for ind in pseudo_coefs_map])
            nz_ind = np.where(np.abs(c) > thresh)[0]
            nz_ind_batched = np.array_split(nz_ind, no_batches)
            c_batched = np.array_split(c[nz_ind], no_batches)
            for ibatch, (nz_ind, c) in enumerate(zip(nz_ind_batched, c_batched)):
                me = _pseudo_me(
                    [bas_list[i] for i in coupl_ind],
                    [m_ind_flat[i] for i in coupl_ind],
                    [pseudo_terms_map[i][nz_ind] for i in coupl_ind],
                )
                hme += me @ c

        e, v = jnp.linalg.eigh(hme.reshape(nbas, nbas))
        v_ind = jnp.where(e - e[0] <= emax)[0]
        v = v[:, v_ind]
        self.enr = e[v_ind]
        nstates = len(v_ind)
        self.bas_ind = np.arange(nstates)

        # compute matrix elements in eigenbasis

        if store_me:

            self.poten_me = jnp.einsum(
                "pqt,pi,qj->ijt",
                _poten_me(
                    [bas_list[i] for i in coupl_ind],
                    [m_ind_flat[i] for i in coupl_ind],
                    [poten_terms_map[i] for i in coupl_ind],
                ).reshape(nbas, nbas, -1),
                v,
                v,
            ).reshape(nstates * nstates, -1)

            self.gmat_me = jnp.einsum(
                "pqt,pi,qj->ijt",
                _gmat_me(
                    [bas_list[i] for i in coupl_ind],
                    [m_ind_flat[i] for i in coupl_ind],
                    [poten_terms_map[i] for i in coupl_ind],
                ).reshape(nbas, nbas, -1),
                v,
                v,
            ).reshape(nstates * nstates, -1)

            self.dme_r = {
                icoo: jnp.einsum(
                    "pqt,pi,qj->ijt",
                    _dme_r_i(
                        [bas_list[i] for i in coupl_ind],
                        [m_ind_flat[i] for i in coupl_ind],
                        [poten_terms_map[i] for i in coupl_ind],
                        icoo,
                    ).reshape(nbas, nbas, -1),
                    v,
                    v,
                ).reshape(nstates * nstates, -1)
                for icoo in self.coo_ind
            }

            self.dme_l = {
                icoo: jnp.einsum(
                    "pqt,pi,qj->ijt",
                    _dme_l_i(
                        [bas_list[i] for i in coupl_ind],
                        [m_ind_flat[i] for i in coupl_ind],
                        [poten_terms_map[i] for i in coupl_ind],
                        icoo,
                    ).reshape(nbas, nbas, -1),
                    v,
                    v,
                ).reshape(nstates * nstates, -1)
                for icoo in self.coo_ind
            }

            self.d2me = {
                (icoo, jcoo): jnp.einsum(
                    "pqt,pi,qj->ijt",
                    _d2me_i_j(
                        [bas_list[i] for i in coupl_ind],
                        [m_ind_flat[i] for i in coupl_ind],
                        [poten_terms_map[i] for i in coupl_ind],
                        icoo,
                        jcoo,
                    ).reshape(nbas, nbas, -1),
                    v,
                    v,
                ).reshape(nstates * nstates, -1)
                for icoo in self.coo_ind
                for jcoo in self.coo_ind
            }

            if pseudo_coefs is not None:
                self.pseudo_me = jnp.einsum(
                    "pqt,pi,qj->ijt",
                    _pseudo_me(
                        [bas_list[i] for i in coupl_ind],
                        [m_ind_flat[i] for i in coupl_ind],
                        [poten_terms_map[i] for i in coupl_ind],
                    ).reshape(nbas, nbas, -1),
                    v,
                    v,
                ).reshape(nstates * nstates, -1)


def _poten_me(coupl_bas, bas_m_ind, term_ind):
    val_prod = 1
    for bas, b_ind, t_ind in zip(coupl_bas, bas_m_ind, term_ind):
        val = bas.poten_me[np.ix_(b_ind, t_ind)]
        # val = bas.poten_me[b_ind, :][:, t_ind]
        val_prod *= val
    return val_prod


def _gmat_me(coupl_bas, bas_m_ind, term_ind):
    val_prod = 1
    for bas, b_ind, t_ind in zip(coupl_bas, bas_m_ind, term_ind):
        val = bas.gmat_me[np.ix_(b_ind, t_ind)]
        val_prod *= val
    return val_prod


def _pseudo_me(coupl_bas, bas_m_ind, term_ind):
    val_prod = 1
    for bas, b_ind, t_ind in zip(coupl_bas, bas_m_ind, term_ind):
        val = bas.pseudo_me[np.ix_(b_ind, t_ind)]
        val_prod *= val
    return val_prod


def _dme_l_i(coupl_bas, bas_m_ind, term_ind, icoo):
    val_prod = 1
    for bas, b_ind, t_ind in zip(coupl_bas, bas_m_ind, term_ind):
        if icoo in bas.coo_ind:
            val = bas.dme_l[icoo][np.ix_(b_ind, t_ind)]
        else:
            val = bas.gmat_me[np.ix_(b_ind, t_ind)]
        val_prod *= val
    return val_prod


def _dme_l(coupl_bas, bas_m_ind, term_ind, coo_ind):
    dme = {}
    for icoo in coo_ind:
        val_prod = 1
        for bas, b_ind, t_ind in zip(coupl_bas, bas_m_ind, term_ind):
            if icoo in bas.coo_ind:
                val = bas.dme_l[icoo][np.ix_(b_ind, t_ind)]
            else:
                val = bas.gmat_me[np.ix_(b_ind, t_ind)]
            val_prod *= val
        dme[icoo] = val_prod
    return dme


def _dme_r_i(coupl_bas, bas_m_ind, term_ind, icoo):
    val_prod = 1
    for bas, b_ind, t_ind in zip(coupl_bas, bas_m_ind, term_ind):
        if icoo in bas.coo_ind:
            val = bas.dme_r[icoo][np.ix_(b_ind, t_ind)]
        else:
            val = bas.gmat_me[np.ix_(b_ind, t_ind)]
        val_prod *= val
    return val_prod


def _dme_r(coupl_bas, bas_m_ind, term_ind, coo_ind):
    dme = {}
    for icoo in coo_ind:
        val_prod = 1
        for bas, b_ind, t_ind in zip(coupl_bas, bas_m_ind, term_ind):
            if icoo in bas.coo_ind:
                val = bas.dme_r[icoo][np.ix_(b_ind, t_ind)]
            else:
                val = bas.gmat_me[np.ix_(b_ind, t_ind)]
            val_prod *= val
        dme[icoo] = val_prod
    return dme


def _d2me_i_j(coupl_bas, bas_m_ind, term_ind, icoo, jcoo):
    val_prod = 1
    for bas, b_ind, t_ind in zip(coupl_bas, bas_m_ind, term_ind):
        if (icoo, jcoo) in bas.d2me:
            val = bas.d2me[(icoo, jcoo)][np.ix_(b_ind, t_ind)]
        elif icoo in bas.coo_ind and jcoo in bas.coo_ind:
            val = (
                bas.dme_l[icoo][np.ix_(b_ind, t_ind)]
                * bas.dme_r[jcoo][np.ix_(b_ind, t_ind)]
            )
        elif icoo in bas.coo_ind:
            val = bas.dme_l[icoo][np.ix_(b_ind, t_ind)]
        elif jcoo in bas.coo_ind:
            val = bas.dme_r[jcoo][np.ix_(b_ind, t_ind)]
        else:
            val = bas.gmat_me[np.ix_(b_ind, t_ind)]
        val_prod *= val
    return val_prod


def _d2me(coupl_bas, bas_m_ind, term_ind, coo_ind):
    d2me = {}
    for icoo in coo_ind:
        for jcoo in coo_ind:
            val_prod = 1
            for bas, b_ind, t_ind in zip(coupl_bas, bas_m_ind, term_ind):
                if (icoo, jcoo) in bas.d2me:
                    val = bas.d2me[(icoo, jcoo)][np.ix_(b_ind, t_ind)]
                elif icoo in bas.coo_ind and jcoo in bas.coo_ind:
                    val = (
                        bas.dme_l[icoo][np.ix_(b_ind, t_ind)]
                        * bas.dme_r[jcoo][np.ix_(b_ind, t_ind)]
                    )
                elif icoo in bas.coo_ind:
                    val = bas.dme_l[icoo][np.ix_(b_ind, t_ind)]
                elif jcoo in bas.coo_ind:
                    val = bas.dme_r[jcoo][np.ix_(b_ind, t_ind)]
                else:
                    val = bas.gmat_me[np.ix_(b_ind, t_ind)]
                val_prod *= val
            d2me[(icoo, jcoo)] = val_prod
    return d2me


def expansion_subspace(
    terms: np.ndarray,
    bas_list,
    coupl_ind: list[int],
    coo_ind: list[int],
    oper_type: str,
):
    unique_terms = np.unique(terms[:, coo_ind], axis=0)
    terms_coo = terms[:, coo_ind]
    terms_map = [
        np.where(np.all(terms_coo == term, axis=1))[0] for term in unique_terms
    ]

    unique_terms_map = []
    for ibas, bas in enumerate(bas_list):
        if oper_type == "poten":
            term_to_ind = {tuple(term): i for i, term in enumerate(bas.poten_terms)}
        elif oper_type == "gmat":
            term_to_ind = {tuple(term): i for i, term in enumerate(bas.gmat_terms)}
        elif oper_type == "pseudo":
            term_to_ind = {tuple(term): i for i, term in enumerate(bas.pseudo_terms)}
        else:
            raise ValueError(f"Invalind operator type: {oper_type}")
        if ibas in coupl_ind:
            ind = [coo_ind.index(icoo) for icoo in bas.coo_ind]
            unique_terms_map.append(
                np.array([term_to_ind[tuple(term)] for term in unique_terms[:, ind]])
            )
        else:
            unique_terms_map.append(
                np.array([term_to_ind[tuple(term)] for term in terms[:, bas.coo_ind]])
            )
    return unique_terms, terms_map, unique_terms_map


def _generate_prod_ind(
    indices: list[list[int]],
    select: Callable[[list[int]], bool] = lambda _: True,
    batch_size: int = None,
):

    no_elem = np.array([len(elem) for elem in indices])
    tot_size = np.prod(no_elem)
    list_ = indices[0]
    for i in range(1, len(indices)):
        list_ = list(product(list_, indices[i]))
        list_ = [tuple(a) + (b,) if isinstance(a, tuple) else (a, b) for a, b in list_]
        list_ = [elem for elem in list_ if select(elem)]
    yield np.array(list_), np.array(list_).T  # indices_out, #multi_ind


def generate_prod_ind(
    indices: list[list[int]],
    select: Callable[[list[int]], bool] = lambda _: True,
    batch_size: int = None,
):
    no_elem = np.array([len(elem) for elem in indices])
    tot_size = np.prod(no_elem)
    if batch_size is None:
        batch_size = tot_size
    no_batches = (tot_size + batch_size - 1) // batch_size

    for ibatch in range(no_batches):
        start_ind = ibatch * batch_size
        end_ind = np.minimum(start_ind + batch_size, tot_size)
        batch_ind = np.arange(start_ind, end_ind)
        multi_ind = np.array(np.unravel_index(batch_ind, no_elem))
        indices_out = np.array(
            [indices[icoo][multi_ind[icoo, :]] for icoo in range(len(indices))]
        ).T
        select_ind = np.where(np.asarray([select(elem) for elem in indices_out]))
        yield indices_out[select_ind], multi_ind[:, select_ind[0]]
