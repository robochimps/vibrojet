import h5py
import numpy as np

from xy2_valence import water_example
from xy2_valence_jacfwd import water_example as water_example_jacfwd

if __name__ == "__main__":
    print("--- water_example ---")
    water_example(max_order=2)

    print("--- water_example_jacfwd ---")
    water_example_jacfwd(max_order=2)

    with h5py.File("water_gmat_valence.h5") as fl:
        gmat = fl["coefs"][()]
        deriv_ind = fl["deriv_ind"][()]
    with h5py.File("water_pseudo_valence.h5") as fl:
        pseudo = fl["coefs"][()]

    with h5py.File("water_gmat_valence_jacfwd.h5") as fl:
        gmat_jacfwd = fl["coefs"][()]
        deriv_ind_ = fl["deriv_ind"][()]
    with h5py.File("water_pseudo_valence_jacfwd.h5") as fl:
        pseudo_jacfwd = fl["coefs"][()]

    print(len(deriv_ind), np.max(deriv_ind - deriv_ind_))
    print(np.max(np.abs(gmat - gmat_jacfwd), axis=(1, 2)))
    print(np.max(np.abs(pseudo - pseudo_jacfwd)))
