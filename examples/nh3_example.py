r"""Example of Taylor series expansion of kinetic
and potential energy operators for ammonia.

Here, we use the following internal coordinates of ammonia
molecule $\text{NH}_3$:
- r_1
- r_2
- r_3
- s_4 = (2 alpha_{23} - alpha_{13} - alpha_{12}) / \sqrt{6}
- s_5 = (alpha_{13} - alpha_{12}) / \sqrt{2}
- rho

where r_i = N-Hi (i=1,2,3), \alpha_{ij} = Hi-C-Hj,
and rho is an 'umbrella' angle [0, \pi].

We build expansion of the potential energy surface (PES)
in terms of the following 1D functions of internal coordinates:
- y_1 = 1 - exp(-a_m(r_1 - r_1^(eq)))
- y_2 = 1 - exp(-a_m(r_2 - r_2^(eq)))
- y_3 = 1 - exp(-a_m(r_3 - r_3^(eq)))
- y_4 = s_4
- y_5 = s_5
- y_6 = cos(rho)

For expansion of the kinetic energy operator (KEO),
we use the following 1D functions of internal coordinates:
- z_1 = r_1 - r_1^(eq)
- z_2 = r_2 - r_2^(eq)
- z_3 = r_3 - r_3^(eq)
- z_4 = s_4
- z_5 = s_5
- z_6 = cos(rho)
"""

import itertools
import os

import jax
import numpy as np
from jax import config
from jax import numpy as jnp
from scipy import optimize

from vibrojet.jet_prim import acos
from vibrojet.keo import Gmat, com, pseudo
from vibrojet.potentials import nh3_POK
from vibrojet.taylor import deriv_list

config.update("jax_enable_x64", True)

# masses of N, H1, H2, H3
MASSES = [14.00307400, 1.007825035, 1.007825035, 1.007825035]

# Define a function `find_alpha_from_s_delta` to obtain three \alpha_{ij}
# valence angular coordinates from the two symmetrized s_4, s_5 and
# 'umbrella' angle delta = rho - pi/2 coordinates.


def find_alpha_from_s_delta(s4, s5, delta, no_iter: int = 10):

    sqrt2 = jnp.sqrt(2.0)
    sqrt3 = jnp.sqrt(3.0)
    sqrt6 = jnp.sqrt(6.0)

    def calc_s_to_sin_delta(s6, s4, s5):
        alpha1 = (sqrt2 * s6 + 2 * s4) / sqrt6
        alpha2 = (sqrt2 * s6 - s4 + sqrt3 * s5) / sqrt6
        alpha3 = (sqrt2 * s6 - s4 - sqrt3 * s5) / sqrt6
        cos_alpha1 = jnp.cos(alpha1)
        cos_alpha2 = jnp.cos(alpha2)
        cos_alpha3 = jnp.cos(alpha3)
        sin_alpha1 = jnp.sin(alpha1)
        sin_alpha2 = jnp.sin(alpha2)
        sin_alpha3 = jnp.sin(alpha3)
        tau_2 = (
            1
            - cos_alpha1**2
            - cos_alpha2**2
            - cos_alpha3**2
            + 2 * cos_alpha1 * cos_alpha2 * cos_alpha3
        )
        norm_2 = (
            sin_alpha3**2
            + sin_alpha2**2
            + sin_alpha1**2
            + 2 * cos_alpha3 * cos_alpha1
            - 2 * cos_alpha2
            + 2 * cos_alpha2 * cos_alpha3
            - 2 * cos_alpha1
            + 2 * cos_alpha2 * cos_alpha1
            - 2 * cos_alpha3
        )
        return tau_2 / norm_2

    # initial value for s6
    alpha1 = 2 * jnp.pi / 3
    s6 = alpha1 * sqrt3
    sin_delta = jnp.sin(delta)
    sin_delta2 = sin_delta**2

    for _ in range(no_iter):
        f = calc_s_to_sin_delta(s6, s4, s5)
        eps = f - sin_delta2
        grad = jax.grad(calc_s_to_sin_delta)(s6, s4, s5)
        dx = eps / grad
        dx0 = dx
        s6 = s6 - dx0

    alpha1 = (sqrt2 * s6 + 2 * s4) / sqrt6
    alpha2 = (sqrt2 * s6 - s4 + sqrt3 * s5) / sqrt6
    alpha3 = (sqrt2 * s6 - s4 - sqrt3 * s5) / sqrt6

    return alpha1, alpha2, alpha3


# internal-to-Cartesian coordinate transformation


@com(MASSES)
def internal_to_cartesian(internal_coords):
    r1, r2, r3, s4, s5, rho = internal_coords
    delta = rho - jnp.pi / 2
    alpha1, alpha2, alpha3 = find_alpha_from_s_delta(s4, s5, delta)

    cos_rho = jnp.cos(rho)
    sin_rho = jnp.sin(rho)

    # beta3 = acos((jnp.cos(alpha3) - jnp.cos(rho) ** 2) / jnp.sin(rho) ** 2)
    # beta2 = acos((jnp.cos(alpha2) - jnp.cos(rho) ** 2) / jnp.sin(rho) ** 2)

    cos_beta3 = (jnp.cos(alpha3) - cos_rho**2) / sin_rho**2
    cos_beta2 = (jnp.cos(alpha2) - cos_rho**2) / sin_rho**2

    sin_beta3 = jnp.sin(acos(cos_beta3))
    sin_beta2 = jnp.sin(acos(cos_beta2))

    # sin_beta3 = jnp.sqrt(1 - cos_beta3**2)  # 0 < beta3 < pi
    # sin_beta2 = jnp.sqrt(1 - cos_beta2**2)  # 0 < beta2 < pi

    xyz = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [r1 * sin_rho, 0.0, r1 * cos_rho],
            [r2 * sin_rho * cos_beta3, r2 * sin_rho * sin_beta3, r2 * cos_rho],
            [r3 * sin_rho * cos_beta2, -r3 * sin_rho * sin_beta2, r3 * cos_rho],
        ]
    )
    return xyz


# y-coordinates for expansion of PES

# Morse constant necessary for defining y-coordinates for stretches
A_MORSE = 2.0


def internal_to_y(q, q0):
    r1, r2, r3, s4, s5, rho = q
    y1 = 1 - jnp.exp(-A_MORSE * (r1 - q0[0]))
    y2 = 1 - jnp.exp(-A_MORSE * (r2 - q0[1]))
    y3 = 1 - jnp.exp(-A_MORSE * (r3 - q0[2]))
    y4 = s4
    y5 = s5
    y6 = jnp.sin(rho)
    return jnp.array([y1, y2, y3, y4, y5, y6])


def y_to_internal(y, q0):
    y1, y2, y3, y4, y5, y6 = y
    r1 = -jnp.log(1 - y1) / A_MORSE + q0[0]
    r2 = -jnp.log(1 - y2) / A_MORSE + q0[1]
    r3 = -jnp.log(1 - y3) / A_MORSE + q0[2]
    s4 = y4
    s5 = y5
    rho = np.pi / 2 - acos(y6)  # = asin(y6)
    return jnp.array([r1, r2, r3, s4, s5, rho])


# z-coordinates for expansion of KEO


def internal_to_z(q, q0):
    r1, r2, r3, s4, s5, rho = q
    z1 = r1 - q0[0]
    z2 = r2 - q0[1]
    z3 = r3 - q0[2]
    z4 = s4
    z5 = s5
    z6 = jnp.sin(rho)
    return jnp.array([z1, z2, z3, z4, z5, z6])


def z_to_internal(z, q0):
    z1, z2, z3, z4, z5, z6 = z
    r1 = z1 + q0[0]
    r2 = z2 + q0[1]
    r3 = z3 + q0[2]
    s4 = z4
    s5 = z5
    rho = np.pi / 2 - acos(z6)  # = asin(z6)
    return jnp.array([r1, r2, r3, s4, s5, rho])


# potential function in internal coordinates


@jax.jit
def poten(q):
    r1, r2, r3, s4, s5, rho = q
    delta = rho - jnp.pi / 2
    alpha1, alpha2, alpha3 = find_alpha_from_s_delta(s4, s5, delta)
    v = nh3_POK.poten((r1, r2, r3, alpha1, alpha2, alpha3))
    return v


# potential function in y-coordinates


@jax.jit
def poten_in_y(y, q0):
    q = y_to_internal(y, q0)
    r1, r2, r3, s4, s5, rho = q
    delta = rho - jnp.pi / 2
    alpha1, alpha2, alpha3 = find_alpha_from_s_delta(s4, s5, delta)
    v = nh3_POK.poten((r1, r2, r3, alpha1, alpha2, alpha3))
    return v


if __name__ == "__main__":

    vmin = optimize.minimize(poten, [1.1, 1.1, 1.1, 0.5, 0.5, np.pi / 2])
    q0 = vmin.x
    v0 = vmin.fun
    print("Equilibrium internal coordinates:", q0)
    print("Min of potential:", v0)

    y0 = internal_to_y(q0, q0)
    z0 = internal_to_z(q0, q0)

    xyz = internal_to_cartesian(q0)
    print("Reference values of internal coordinates:\n", q0)
    print("Reference values of expansion y-coordinates:\n", y0)
    print("Reference values of expansion z-coordinates:\n", z0)
    print("Reference values of Cartesian coordinates:\n", xyz)

    # generate expansion power indices

    max_order = 6  # max total expansion order
    deriv_ind = [
        elem
        for elem in itertools.product(
            *[range(0, max_order + 1) for _ in range(len(q0))]
        )
        if sum(elem) <= max_order
    ]
    print("Max expansion order:", max_order)
    print("Number of expansion terms:", len(deriv_ind))

    # PES expansion in internal coordinates

    poten_file = f"nh3_poten_coefs_{max_order}.npz"
    if os.path.exists(poten_file):
        print(f"load potential expansion from file {poten_file}")
        data = np.load(poten_file)
        poten_coefs = data['coefs']
        poten_deriv_ind = data['deriv_ind']
    else:
        print("expand potential in internal coordinates ...")
        poten_coefs = deriv_list(poten, deriv_ind, q0, if_taylor=True)
        np.savez(poten_file, coefs=poten_coefs, deriv_ind=deriv_ind, q0=q0)

    # PES expansion in y-coordinates

    poten_file = f"nh3_poten_in_y_coefs_{max_order}.npz"
    if os.path.exists(poten_file):
        print(f"load potential expansion from file {poten_file}")
        data = np.load(poten_file)
        poten_in_y_coefs = data['coefs']
        poten_in_y_deriv_ind = data['deriv_ind']
    else:
        print("expand potential in y-coordinates ...")
        poten_in_y_coefs = deriv_list(
            lambda y: poten_in_y(y, q0), deriv_ind, y0, if_taylor=True
        )
        np.savez(poten_file, coefs=poten_in_y_coefs, deriv_ind=deriv_ind, q0=q0)

    # KEO expansion in internal coordinates

    gmat_file = f"nh3_gmat_coefs_{max_order}.npz"
    if os.path.exists(gmat_file):
        print(f"load G-matrix expansion from file {gmat_file}")
        data = np.load(gmat_file)
        gmat_coefs = data['coefs']
        gmat_deriv_ind = data['deriv_ind']
    else:
        print("expand keo in internal coordinates ...")
        gmat_coefs = deriv_list(
            lambda q: Gmat(q, MASSES, internal_to_cartesian),
            deriv_ind,
            q0,
            if_taylor=True,
        )
        np.savez(gmat_file, coefs=gmat_coefs, deriv_ind=deriv_ind, q0=q0)

    # KEO expansion in z-coordinates

    gmat_file = f"nh3_gmat_in_z_coefs_{max_order}.npz"
    if os.path.exists(gmat_file):
        print(f"load G-matrix expansion from file {gmat_file}")
        data = np.load(gmat_file)
        gmat_in_z_coefs = data['coefs']
        gmat_in_z_deriv_ind = data['deriv_ind']
    else:
        print("expand keo in z-coordinates ...")
        gmat_in_z_coefs = deriv_list(
            lambda z: Gmat(z_to_internal(z, q0), MASSES, internal_to_cartesian),
            deriv_ind,
            z0,
            if_taylor=True,
        )
        np.savez(gmat_file, coefs=gmat_in_z_coefs, deriv_ind=deriv_ind, q0=q0)
