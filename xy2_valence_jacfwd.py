"""Taylor series expansion of the kinetic energy operator for a triatomic molecule

Uses jax.jacfwd for derivative propagation, for testing only.
"""

import itertools

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import factorial

from vibrojet._keo import Gmat, com, eckart, pseudo, gmat

jax.config.update("jax_enable_x64", True)


def water_example(max_order: int = 4):
    """Calculates and stores in files Taylor series expansions
    of kinetic energy G-matrix and pseudopotential,
    using the Eckart frame.

    Args:
        max_order (int): The total expansion order.
    """

    # Masses of O, H, H atoms
    masses = np.array([15.9994, 1.00782505, 1.00782505])

    # Equilibrium values of valence coordinates
    r1, r2, alpha = 1.958, 0.958, 1.824
    x0 = jnp.array([r1, r2, alpha], dtype=jnp.float64)

    # Valence-to-Cartesian coordinate transformation
    #   input: array of three valence coordinates
    #   output: array of shape (number of atoms, 3)
    #       containing Cartesian coordinates of atoms
    # `com` shifts coordinates to the centre of mass
    # `eckart` rotates coordinates to the Eckart frame

    @eckart(x0, masses)
    # @com(masses)
    def valence_to_cartesian(internal_coords):
        r1, r2, a = internal_coords
        return jnp.array(
            [
                [0.0, 0.0, 0.0],
                [r1 * jnp.sin(a / 2), 0.0, r1 * jnp.cos(a / 2)],
                [-r2 * jnp.sin(a / 2), 0.0, r2 * jnp.cos(a / 2)],
            ]
        )

    xyz0 = valence_to_cartesian(x0)
    print("masses:", masses)
    print("reference internal coordinates:\n", x0)
    print("reference Cartesian coordinates:\n", xyz0)

    # Generate list of multi-indices specifying the integer exponents
    # for each coordinate in the Taylor series expansion

    deriv_ind = [
        elem
        for elem in itertools.product(
            *[range(0, max_order + 1) for _ in range(len(x0))]
        )
        if sum(elem) <= max_order
    ]

    print("max expansion order:", max_order)
    print("number of expansion terms:", len(deriv_ind))

    # Compute Taylor series expansion coefficients for G-matrix

    print("compute expansion of G-matrix ...")

    # Gmat_coefs = np.zeros((len(deriv_ind), 3, 3), dtype=np.float64)
    # for i in range(3):
        # for j in range(3):
            # func = lambda x: Gmat(x, masses, valence_to_cartesian)[i,j]
            # func = lambda x: gmat(x, masses, valence_to_cartesian)[i,j]
    # func = lambda x: valence_to_cartesian(x)
    func = lambda x: Gmat(x, masses, valence_to_cartesian)

    def jacfwd(x0, ind):
        f = func
        for _ in range(sum(ind)):
            f = jax.jacfwd(f)
        i = sum([(i,) * o for i, o in enumerate(ind)], start=tuple())
        return f(x0)[:, :, *i]
        # return f(x0)[i]

    Gmat_coefs = np.array(
        [jacfwd(x0, ind) / np.prod(factorial(ind)) for ind in deriv_ind]
    )
    # Gmat_coefs = np.array([gmat(x0, masses, valence_to_cartesian)])

    # Store G-matrix coefficients in ASCII file

    coefs_file = "water_gmat_valence_jacfwd"
    print("store expansion of G-matrix in file", coefs_file)
    with open(coefs_file + ".txt", "w") as fl:
        fl.write("Reference Cartesian coordinates (Angstrom)\n")
        for m, x in zip(masses, xyz0):
            fl.write(
                "%20.12e" % m + "  " + "  ".join("%20.12e" % elem for elem in x) + "\n"
            )
        fl.write("G-matrix expansion (cm^-1)\n")
        for c, i in zip(Gmat_coefs, deriv_ind):
            fl.write(
                " ".join("%2i" % elem for elem in i)
                + "   "
                + "  ".join("%20.12e" % elem for elem in c.ravel())
                + "\n",
            )

    # Additionally, store G-matrix coefficients in hdf5 file

    with h5py.File(coefs_file + ".h5", "w") as fl:
        d_xyz = fl.create_dataset("xyz0", data=xyz0)
        d_xyz.attrs["units"] = "Angstrom"
        fl.create_dataset("deriv_ind", data=deriv_ind)
        d_coefs = fl.create_dataset("coefs", data=Gmat_coefs)
        d_coefs.attrs["units"] = "cm^-1"

    # Compute Taylor series expansion coefficients for pseudopotential

    # print("compute expansion of pseudopotential ...")

    # func = lambda x: pseudo(x, masses, valence_to_cartesian)

    # def jacfwd2(x0, ind):
    #     f = func
    #     for _ in range(sum(ind)):
    #         f = jax.jacfwd(f)
    #     i = sum([(i,) * o for i, o in enumerate(ind)], start=tuple())
    #     return f(x0)[i]

    # pseudo_coefs = np.array(
    #     [jacfwd2(x0, ind) / np.prod(factorial(ind)) for ind in deriv_ind]
    # )

    # Store pseudopotential coefficients in ASCII file

    # coefs_file = "water_pseudo_valence_jacfwd"
    # print("store expansion of pseudopotential in file", coefs_file)
    # with open(coefs_file + ".txt", "w") as fl:
    #     fl.write("Reference Cartesian coordinates (Angstrom)\n")
    #     for m, x in zip(masses, xyz0):
    #         fl.write(
    #             "%20.12e" % m + "  " + "  ".join("%20.12e" % elem for elem in x) + "\n"
    #         )
    #     fl.write("pseudopotential expansion (cm^-1)\n")
    #     for c, i in zip(pseudo_coefs, deriv_ind):
    #         fl.write(
    #             " ".join("%2i" % elem for elem in i) + "   " + "%20.12e" % c + "\n",
    #         )

    # Additionally, store pseudopotential coefficients in hdf5 file

    # with h5py.File(coefs_file + ".h5", "w") as fl:
    #     d_xyz = fl.create_dataset("xyz0", data=xyz0)
    #     d_xyz.attrs["units"] = "Angstrom"
    #     fl.create_dataset("deriv_ind", data=deriv_ind)
    #     d_coefs = fl.create_dataset("coefs", data=pseudo_coefs)
    #     d_coefs.attrs["units"] = "cm^-1"


if __name__ == "__main__":
    water_example()
