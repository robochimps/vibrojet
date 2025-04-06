# vibrojet

Python package for constructing molecular rovibrational kinetic and potential energy operators using Taylor-mode automatic differentiation.

Vibrojet enables the computation of the rovibrational kinetic energy $G$-matrix and pseudopotential for arbitrary molecule and user-defined internal coordinates and molecular frame embeddings.
Kinetic energy operators can be evaluated efficiently either on a grid of coordinate values or as Taylor-series expansions around a reference geometry.
The same expansion framework can be applied to other molecular properties, such as potential energy and dipole moment surfaces.

Expansion coefficients can be exported to files and used with variational rovibrational codes, such as [TROVE](https://github.com/Trovemaster/TROVE), for computing matrix elements, energy levels, and spectra.

Vibrojet builds on `jax.experimental.jet` to perform Taylor-mode automatic differentiation and extends it with additional functions for building multivariate Taylor polynomials of rovibrational kinetic energy operators.

Installation
-----
```bash
pip install --upgrade git+https://github.com/robochimps/vibrojet.git
```

Quick example
----
Taylor-series expansion of the rovibrational kinetic energy $G$-matrix for a triatomic molecule in the Eckart molecular frame, expressed in terms of the valence coordinates: two bond lengths $r_1$, $r_2$, and the bond angle $\alpha$:

$$
G_{\lambda,\mu}(r_1,r_2,\alpha)=\sum_\mathbf{t} g_\mathbf{t}^{(\lambda,\mu)}(r_1-r_1^{(eq)})^{t_1}(r_2-r_2^{(eq)})^{t_2}(\alpha-\alpha^{(eq)})^{t_3},
$$

where $\lambda,\mu=\{r_1,r_2,\alpha,\phi,\theta,\chi,X,Y,Z\}$ correspond to the $3N-6$ vibrational, three rotational, and three translational coordinate indices.
```py
import itertools
import jax
import jax.numpy as jnp
from vibrojet.eckart import eckart
from vibrojet.keo import Gmat, batch_Gmat, batch_pseudo, pseudo
from vibrojet.taylor import deriv_list
jax.config.update("jax_enable_x64", True)

# Masses of O, H, H atoms
masses = [15.9994, 1.00782505, 1.00782505]

# Equilibrium values of valence coordinates
r1, r2, alpha = 0.958, 0.958, 1.824
q0 = [r1, r2, alpha]

# Valence-to-Cartesian coordinate transformation
#   input: array of three valence coordinates
#   output: array of shape (number of atoms, 3)
#           containing Cartesian coordinates of atoms

@eckart(q0, masses)
def valence_to_cartesian(q):
    r1, r2, a = q
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [r1 * jnp.sin(a / 2), 0.0, r1 * jnp.cos(a / 2)],
            [-r2 * jnp.sin(a / 2), 0.0, r2 * jnp.cos(a / 2)],
        ]
    )


# Generate list of multi-indices specifying the integer exponents
# for each coordinate in the Taylor series expansion

max_order = 4  # max total expansion order
deriv_ind = [
    elem
    for elem in itertools.product(*[range(0, max_order + 1) for _ in range(len(q0))])
    if sum(elem) <= max_order
]
print("max expansion order:", max_order)
print("number of expansion terms:", len(deriv_ind))

# Function for computing kinetic G-matrix for given masses of atoms
# and internal coordinates
func = lambda x: Gmat(x, masses, valence_to_cartesian)

# Compute Taylor series expansion coefficients
Gmat_coefs = deriv_list(func, deriv_ind, q0, if_taylor=True)
```
The array `Gmat_coefs` contains Taylor-series expansion coefficients $g_\mathbf{t}^{(\lambda,\mu)}$, where the first dimension indexes multi-index $\mathbf{t}$, and second and third dimensions correspond to $\lambda$ and $\mu$, respectively. The associated derivative multi-indices $\mathbf{t}$ are stored in `deriv_ind`.

Examples
---
More examples and user contributions for various molecules can be found in the [examples](examples) folder.

Citation
---
If you use this code in your research, please cite:
> A. Yachmenev, E. Vogt, A. F. Corral, Y. Saleh, "Taylor-mode automatic differentiation for constructing molecular rovibrational Hamiltonian operators" (2025) submitted

```bibtex
@article{Yachmenev2025,
  author  = {A. Yachmenev, E. Vogt, A. F. Corral, Y. Saleh},
  title   = {Taylor-mode automatic differentiation for constructing molecular rovibrational Hamiltonian operators},
  year    = {2025},
  journal = {Submitted},
  archiveprefix = {arXiv},
  arxivid = {},
  eprint = {},
  primaryclass = {physics},
  arxiv = {},
}
```


Contact
---
For questions or feedback, feel free to open an issue or reach out to the authors directly via andrey.yachmenev@robochimps.com
