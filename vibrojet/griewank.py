"""This module implements a partial derivative computation method based on univariate 
Taylor series expansion, as described in the work of Andreas Griewank, Jean Utke, 
and Andrea Walther.

Rerefence:
Andreas Griewank, Jean Utke and Andrea Walther,
"Evaluating higher derivative tensors by forward propagation of univariate Taylor series",
Math. Comp. 69 (2000), 1117-1130, https://doi.org/10.1090/S0025-5718-00-01120-0
"""

import itertools
from typing import Callable, List

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import jet
from scipy.special import comb, binom

jax.config.update("jax_enable_x64", True)


def _coefs_i(i: List[int]):
    """Coefficients from Eq. (13) of Griewank's paper"""
    k_ind = [elem for elem in itertools.product(*[range(0, k + 1) for k in i])]
    sum_i_min_k = (-1) ** np.sum(np.array(i)[None, :] - np.array(k_ind), axis=-1)
    c = jnp.array([np.prod(comb(i, k)) for k in k_ind]) * sum_i_min_k
    return k_ind, c


def _coefs_ij(i: List[int], j: List[int], d: int):
    """Coefficients c_{i,j} from Eq. (17) of Griewank's paper"""
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


def deriv(
    func: Callable[[jnp.ndarray], float], deriv_ind: List[int], x0: jnp.ndarray
) -> float:
    """Computes partial derivative of a function `func` at a point `x0`
    given by the derivative multi-index `deriv_ind`.

    The function uses interpolation formula in Eq. (13) of Andreas Griewank,
    Jean Utke and Andrea Walther, "Evaluating higher derivative tensors by forward
    propagation of univariate Taylor series", Math. Comp. 69 (2000), 1117-1130,
    https://doi.org/10.1090/S0025-5718-00-01120-0

    Args:
        func (Callable[[jnp.ndarray], float]): The function to be differentiated.
            It should accept an array of coordinate values as input and return
            a scalar value.
        deriv_ind (List[int]): A multi-index specifying the order of differentiation along
            each coordinate.
        x0 (jnp.ndarray): The point at which the partial derivative is computed.

    Returns:
        float: The computed partial derivative value of `func` at `x0`.
    """
    k, c = _coefs_i(deriv_ind)
    sum_i = sum(deriv_ind)

    @jax.jit
    def _sum(carry, i):
        _, (*_, res) = jet.jet(
            func, (x0,), ((jnp.asarray(k)[i],) + (jnp.zeros_like(x0),) * (sum_i - 1),)
        )
        return carry + res * c[i], 0

    res, _ = jax.lax.scan(_sum, 0, jnp.arange(len(k)))
    return res


def deriv_list(
    func: Callable[[jnp.ndarray], float],
    deriv_ind_list: List[List[int]],
    x0: jnp.ndarray,
) -> jnp.ndarray:
    """Computes partial derivatives of a function `func` at a point `x0`
    given by the list of derivative multi-indices `deriv_ind_list`.

    The function uses interpolation formulas in Eq. (13) and (17) of Andreas Griewank,
    Jean Utke and Andrea Walther, "Evaluating higher derivative tensors by forward
    propagation of univariate Taylor series", Math. Comp. 69 (2000), 1117-1130,
    https://doi.org/10.1090/S0025-5718-00-01120-0

    Args:
        func (Callable[[jnp.ndarray], float]): The function to be differentiated.
            It should accept an array of coordinate values as input and return
            a scalar value.
        deriv_ind_list (List[List[int]]): A list of multi-indices specifying the order
            of differentiation along each coordinate.
        x0 (jnp.ndarray): The point at which the partial derivative is computed.

    Returns:
        array(float): Array of computed partial derivative values of `func` at `x0`.
    """
    d = np.max(np.sum(deriv_ind_list, axis=-1))
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

        @jax.jit
        def _sum(carry, i):
            _, (*_, res) = jet.jet(
                func,
                (x0,),
                ((jnp.asarray(j)[i],) + (jnp.zeros_like(x0),) * (d - 1),),
            )
            return 0, res

        _, res = jax.lax.scan(_sum, 0, jnp.arange(len(j)))
        f_d[d] = res

    coefs = []

    for i in deriv_ind_list:
        d = sum(i)
        j = j_d[d]
        c = np.array([_coefs_ij(i, j_, d) for j_ in j])
        coefs.append(jnp.sum(c * f_d[d]))

    return jnp.array(coefs)
