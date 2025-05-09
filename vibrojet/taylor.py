import itertools
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import jet
from scipy.special import binom, comb, factorial
from functools import partial
import time

jax.config.update("jax_enable_x64", True)


def _coefs_i(i: List[int]):
    """Coefficients from Eq. (13) of Griewank's paper

    Rerefence:
        - Andreas Griewank, Jean Utke and Andrea Walther,
        "Evaluating higher derivative tensors by forward propagation of univariate Taylor series",
        Math. Comp. 69 (2000), 1117-1130, https://doi.org/10.1090/S0025-5718-00-01120-0
    """
    k_ind = [elem for elem in itertools.product(*[range(0, k + 1) for k in i])]
    sum_i_min_k = (-1) ** np.sum(np.array(i)[None, :] - np.array(k_ind), axis=-1)
    c = jnp.array([np.prod(comb(i, k)) for k in k_ind]) * sum_i_min_k
    return k_ind, c


def _coefs_ij(i: List[int], j: List[int], d: int):
    """Coefficients c_{i,j} from Eq. (17) of Griewank's paper

    Rerefence:
        - Andreas Griewank, Jean Utke and Andrea Walther,
        "Evaluating higher derivative tensors by forward propagation of univariate Taylor series",
        Math. Comp. 69 (2000), 1117-1130, https://doi.org/10.1090/S0025-5718-00-01120-0
    """
    sum_i = sum(i)
    k_ind = [elem for elem in itertools.product(*[range(0, k + 1) for k in i])]
    fac1 = (-1) ** np.sum(np.array(i)[None, :] - np.array(k_ind), axis=-1)
    if d == 0:
        fac2 = np.ones_like(len(k_ind))
    else:
        fac2 = (np.sum(k_ind, axis=-1) / d) ** sum_i
    x = [d / sum(k) * np.array(k) if sum(k) > 0 else np.array(k) for k in k_ind]
    c = np.sum(
        np.array([np.prod(comb(i, k) * binom(x_, j)) for k, x_ in zip(k_ind, x)])
        * fac1
        * fac2
    )
    return c


def deriv_list(
    func: Callable[[jnp.ndarray], float],
    deriv_ind_list: List[List[int]],
    x0: jnp.ndarray,
    if_taylor: bool = False,
) -> jnp.ndarray:
    """Computes partial derivatives of a function `func` at a point `x0`
    given by the list of derivative multi-indices `deriv_ind_list`.

    The function uses interpolation formulas in Eq. (13) and (17) of Andreas Griewank,
    Jean Utke and Andrea Walther, "Evaluating higher derivative tensors by forward
    propagation of univariate Taylor series", Math. Comp. 69 (2000), 1117-1130,
    https://doi.org/10.1090/S0025-5718-00-01120-0

    Args:
        func (Callable[[jnp.ndarray], jnp.ndarray]): The function to be differentiated.
            It should accept an array of coordinate values as input and return
            an array or scalar value.
        deriv_ind_list (List[List[int]]): A list of multi-indices specifying the order
            of differentiation along each coordinate.
        x0 (jnp.ndarray): The point at which the partial derivative is computed.
        if_taylor (bool): If True, returns Taylor expansion coefficients.

    Returns:
        array(float): Array of computed partial derivative values (of Taylor expansion coefficients)
            of `func` at `x0`.
    """

    # @partial(jax.jit, static_argnums=1)
    # @partial(jax.vmap, in_axes=(0, None))
    def _jet(j, d):
        _, (*_, res) = jet.jet(
            func,
            (x0_arr,),
            (
                (jnp.asarray(j, dtype=jnp.float64),)
                + (jnp.zeros_like(x0_arr),) * (d - 1),
            ),
        )
        return res

    x0_arr = jnp.asarray(x0)
    ncoo = len(x0)
    deg_list = np.sort(np.unique(np.sum(deriv_ind_list, axis=-1)))

    f_d = {}
    j_d = {}
    for d in deg_list:
        j = np.array(
            [
                elem
                for elem in itertools.product(*[range(0, d + 1) for _ in range(ncoo)])
                if np.sum(elem) == d
            ]
        )
        j_d[d] = j

        start_time = time.time()

        f_d[d] = jnp.array([_jet(j_, d) for j_ in j])
        # f_d[d] = _jet(j, d)

        end_time = time.time()
        print("Time for d=", d, ":", np.round(end_time - start_time, 2), "s")

    coefs = []

    for i in deriv_ind_list:
        if np.all(np.array(i) == 0):
            c = func(x0_arr)
        else:
            d = sum(i)
            j = j_d[d]
            c = jnp.array([_coefs_ij(i, j_, d) for j_ in j])
            c = jnp.einsum("i,i...->...", c, f_d[d]) / factorial(d)
            if if_taylor:
                c = c / jnp.prod(factorial(i))
        coefs.append(c)

    return jnp.array(coefs)


def _inv_taylor(pow_ind: np.ndarray, coefs: np.ndarray) -> np.ndarray:
    """Given a Taylor series expansion of a matrix or tensor, specified by multi-indices `pow_ind`
    and corresponding coefficients `coefs`, computes the Taylor series expansion of its inverse.

    Args:
        pow_ind (np.ndarray): A 2D array of shape `(num_terms, num_coords)` containing integer
            exponents for each coordinate in the Taylor series expansion. Here, `num_terms` is
            the number of expansion terms, and `num_coords` is the number of coordinates.
        coefs (np.ndarray): A multi-dimensional array of shape `(num_terms, ...)` containing the
            Taylor series coefficients in the same order as `pow_ind`. The trailing dimensions
            represent the matrix or tensor whose inverse is to be computed.

    Returns:
        np.ndarray: An array of the same shape as `coefs`, containing the Taylor series expansion
            coefficients of the inverse of the input matrix or tensor.
    """

    assert len(np.unique(pow_ind, axis=-1)) == len(
        pow_ind
    ), f"Argument 'pow_ind' contains duplicate multi-indices"

    ind0 = np.where(jnp.all(pow_ind == 0, axis=-1))[0][0]
    inv = np.linalg.inv(coefs[ind0])
    inv_coefs = {tuple(pow_ind[ind0]): inv}

    sort_ind = np.argsort(np.sum(pow_ind, axis=-1))
    pow_ind_sorted = pow_ind[sort_ind]
    coefs_sorted = coefs[sort_ind]

    for i in pow_ind_sorted:
        if np.sum(i, axis=-1) == 0:
            continue
        ind = jnp.where(
            (np.any(pow_ind_sorted > 0, axis=-1))
            & (np.all(i[None, :] - pow_ind_sorted >= 0, axis=-1))
        )
        j = pow_ind_sorted[ind]
        c = coefs_sorted[ind]
        ci = np.array(
            [
                inv_coefs.get(tuple(elem), jnp.zeros_like(inv))
                # inv_coefs[tuple(elem)]
                for elem in i[None, :] - j
            ]
        )
        inv_coefs[tuple(i)] = -np.einsum(
            "ac,tci,tib->ab", inv, c, ci, optimize="optimal"
        )

    return np.array(
        [inv_coefs.get(tuple(elem), np.zeros_like(inv)) for elem in pow_ind]
    )
