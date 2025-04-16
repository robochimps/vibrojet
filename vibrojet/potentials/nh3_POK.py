"""
Potential energy surface of ammonia NH₃.

This module provides a Python implementation of the potential energy surface
originally defined in the TROVE function `MLpoten_xy3_morbid_Roman_10`,
https://github.com/Trovemaster/TROVE/blob/bdfbb1c066943f1e44fd88abbe35858ce46ea7d2/pot_NH3_Roman.f90#L94C12-L94C39

The parameters of the function are included in the supplementary material
of the manuscript:

O. L. Polyansky, R. I. Ovsyannikov, A. A. Kyuberis, L. Lodi, J. Tennyson,
A. Yachmenev, S. N. Yurchenko, N. F. Zobov,
"Calculation of rotation-vibration energy levels of the ammonia molecule
based on an ab initio potential energy surface",
J. Mol. Spectrosc. 327, 21-30 (2016).
https://doi.org/10.1016/j.jms.2016.08.003

**Units:**
- Distances must be provided in Ångströms.
- Angles must be provided in radians.
- The output energy is given in inverse centimeters (cm⁻¹)

**Example energy levels (in cm⁻¹):**
...
"""
import jax
from jax import numpy as jnp
from jax import config
import numpy as np


jax.config.update("jax_enable_x64", True)


#  ! Missing parameters
PARAMS = []


def poten_xy3_morbid_DBOC(q):
    r1, r2, r3, s4, s5, delta = q

    r_eq = PARAMS[0]
    alpha_eq = PARAMS[1] * np.pi / 180
    rhoe = np.pi - np.arcsin(2 * np.sin(alpha_eq / 2) / np.sqrt(3))
    aa1 = PARAMS[2]

    alpha1, alpha2, alpha3 = _find_alpha_from_sindelta(s4, s5, jnp.sin(delta))

    alpha = (alpha1 + alpha2 + alpha3) / 3
    sinrho = 2 * jnp.sin(alpha / 2) / jnp.sqrt(3)

    assert jnp.abs(sinrho) < 1, "|sin(rho)| > 1.0"

    y1 = 1 - jnp.exp(-aa1 * (r1 - r_eq))
    y2 = 1 - jnp.exp(-aa1 * (r2 - r_eq))
    y3 = 1 - jnp.exp(-aa1 * (r3 - r_eq))
    y4 = s4
    y5 = s5
    cosrho = jnp.sin(rhoe) - sinrho

    force = PARAMS[3:]
    v = pot_v2(cosrho, y1, y2, y3, y4, y5, force[:35])
    if len(force) > 35:
        v += pot_v3(cosrho, y1, y2, y3, y4, y5, force[35:67])
    if len(force) > 67:
        v += pot_v4(cosrho, y1, y2, y3, y4, y5, force[67:109])
    if len(force) > 109:
        v += pot_v5(cosrho, y1, y2, y3, y4, y5, force[109:181])
    if len(force) > 181:
        v += pot_v6(cosrho, y1, y2, y3, y4, y5, force[181:])
    return v


def _find_alpha_from_sindelta(s4, s5, sindelta, no_iter: int = 20):

    def calc_s2sindelta2(s6, s4, s5):
        alpha1 = (jnp.sqrt(2.0) * s6 + 2.0 * s4) / jnp.sqrt(6.0)
        alpha2 = (jnp.sqrt(2.0) * s6 - s4 + jnp.sqrt(3.0) * s5) / jnp.sqrt(6.0)
        alpha3 = (jnp.sqrt(2.0) * s6 - s4 - jnp.sqrt(3.0) * s5) / jnp.sqrt(6.0)
        tau_2 = (
            1.0
            - jnp.cos(alpha1) ** 2
            - jnp.cos(alpha2) ** 2
            - jnp.cos(alpha3) ** 2
            + 2.0 * jnp.cos(alpha1) * jnp.cos(alpha2) * jnp.cos(alpha3)
        )
        norm_2 = (
            jnp.sin(alpha3) ** 2
            + jnp.sin(alpha2) ** 2
            + jnp.sin(alpha1) ** 2
            + 2.0 * jnp.cos(alpha3) * jnp.cos(alpha1)
            - 2.0 * jnp.cos(alpha2)
            + 2.0 * jnp.cos(alpha2) * jnp.cos(alpha3)
            - 2.0 * jnp.cos(alpha1)
            + 2.0 * jnp.cos(alpha2) * jnp.cos(alpha1)
            - 2.0 * jnp.cos(alpha3)
        )
        return tau_2 / norm_2

    # initial values for alpha1 and s6
    alpha1 = 2.0 * jnp.pi / 3
    s6 = alpha1 * jnp.sqrt(3)

    for _ in range(no_iter):
        f = calc_s2sindelta2(s6, s4, s5)
        eps = f - sindelta**2
        grad = jax.grad(calc_s2sindelta2)(s6, s4, s5)
        dx = eps / grad
        dx0 = dx
        s6 = s6 - dx0
        print(dx0)

    alpha1 = (jnp.sqrt(2.0) * s6 + 2.0 * s4) / jnp.sqrt(6)
    alpha2 = (jnp.sqrt(2.0) * s6 - s4 + jnp.sqrt(3.0) * s5) / jnp.sqrt(6.0)
    alpha3 = (jnp.sqrt(2.0) * s6 - s4 - jnp.sqrt(3.0) * s5) / jnp.sqrt(6.0)

    return alpha1, alpha2, alpha3


def pot_v2(coro, y1, y2, y3, y4, y5, force):
    sqrt3 = jnp.sqrt(3)
    (
        ve,
        f1a,
        f2a,
        f3a,
        f4a,
        f5a,
        f6a,
        f7a,
        f0a1,
        f1a1,
        f2a1,
        f3a1,
        f4a1,
        f5a1,
        f6a1,
        f0a11,
        f1a11,
        f2a11,
        f3a11,
        f4a11,
        f0a12,
        f1a12,
        f2a12,
        f3a12,
        f4a12,
        f0a14,
        f1a14,
        f2a14,
        f3a14,
        f4a14,
        f0a44,
        f1a44,
        f2a44,
        f3a44,
        f4a44,
    ) = force

    v0 = (
        ve
        + f1a * coro
        + f2a * coro**2
        + f3a * coro**3
        + f4a * coro**4
        + f5a * coro**5
        + f6a * coro**6
        + f7a * coro**7
    )  #  +f8a*coro**8

    fea1 = (
        f0a1
        + f1a1 * coro
        + f2a1 * coro**2
        + f3a1 * coro**3
        + f4a1 * coro**4
        + f5a1 * coro**5
        + f6a1 * coro**6
    )
    fea11 = f0a11 + f1a11 * coro + f2a11 * coro**2 + f3a11 * coro**3 + f4a11 * coro**4
    fea12 = f0a12 + f1a12 * coro + f2a12 * coro**2 + f3a12 * coro**3 + f4a12 * coro**4
    fea14 = f0a14 + f1a14 * coro + f2a14 * coro**2 + f3a14 * coro**3 + f4a14 * coro**4
    fea44 = f0a44 + f1a44 * coro + f2a44 * coro**2 + f3a44 * coro**3 + f4a44 * coro**4
    v1 = (y3 + y2 + y1) * fea1
    v2 = (
        (y2 * y3 + y1 * y3 + y1 * y2) * fea12
        + (y2**2 + y3**2 + y1**2) * fea11
        + (
            -sqrt3 * y3 * y5 / 2.0
            - y3 * y4 / 2.0
            + y1 * y4
            + sqrt3 * y2 * y5 / 2.0
            - y2 * y4 / 2.0
        )
        * fea14
        + (y5**2 + y4**2) * fea44
    )
    return v0 + v1 + v2


def pot_v3(coro, y1, y2, y3, y4, y5, force):
    sqrt3 = jnp.sqrt(3)
    (
        f0a111,
        f1a111,
        f2a111,
        f3a111,
        f0a112,
        f1a112,
        f2a112,
        f3a112,
        f0a114,
        f1a114,
        f2a114,
        f3a114,
        f0a123,
        f1a123,
        f2a123,
        f3a123,
        f0a124,
        f1a124,
        f2a124,
        f3a124,
        f0a144,
        f1a144,
        f2a144,
        f3a144,
        f0a155,
        f1a155,
        f2a155,
        f3a155,
        f0a455,
        f1a455,
        f2a455,
        f3a455,
    ) = force

    fea111 = f0a111 + f1a111 * coro + f2a111 * coro**2 + f3a111 * coro**3
    fea112 = f0a112 + f1a112 * coro + f2a112 * coro**2 + f3a112 * coro**3
    fea114 = f0a114 + f1a114 * coro + f2a114 * coro**2 + f3a114 * coro**3
    fea123 = f0a123 + f1a123 * coro + f2a123 * coro**2 + f3a123 * coro**3
    fea124 = f0a124 + f1a124 * coro + f2a124 * coro**2 + f3a124 * coro**3
    fea144 = f0a144 + f1a144 * coro + f2a144 * coro**2 + f3a144 * coro**3
    fea155 = f0a155 + f1a155 * coro + f2a155 * coro**2 + f3a155 * coro**3
    fea455 = f0a455 + f1a455 * coro + f2a455 * coro**2 + f3a455 * coro**3

    v3 = (
        (
            y1 * y3 * y4
            + y1 * y2 * y4
            - 2.0 * y2 * y3 * y4
            + sqrt3 * y1 * y2 * y5
            - sqrt3 * y1 * y3 * y5
        )
        * fea124
        + (
            3.0 / 4.0 * y3 * y4**2
            - sqrt3 * y3 * y4 * y5 / 2.0
            + y1 * y5**2
            + y2 * y5**2 / 4.0
            + 3.0 / 4.0 * y2 * y4**2
            + sqrt3 * y2 * y4 * y5 / 2.0
            + y3 * y5**2 / 4.0
        )
        * fea155
        + (y2 * y3**2 + y1 * y3**2 + y1**2 * y3 + y1 * y2**2 + y2**2 * y3 + y1**2 * y2)
        * fea112
        + (-(y4**3) / 3.0 + y4 * y5**2) * fea455
        + fea123 * y1 * y2 * y3
        + (
            y1 * y4**2
            + 3.0 / 4.0 * y3 * y5**2
            + 3.0 / 4.0 * y2 * y5**2
            + y2 * y4**2 / 4.0
            - sqrt3 * y2 * y4 * y5 / 2.0
            + sqrt3 * y3 * y4 * y5 / 2.0
            + y3 * y4**2 / 4.0
        )
        * fea144
        + (y3**3 + y2**3 + y1**3) * fea111
        + (
            -(y2**2) * y4 / 2.0
            - y3**2 * y4 / 2.0
            + sqrt3 * y2**2 * y5 / 2.0
            + y1**2 * y4
            - sqrt3 * y3**2 * y5 / 2.0
        )
        * fea114
    )
    return v3


def pot_v4(coro, y1, y2, y3, y4, y5, force):
    sqrt3 = jnp.sqrt(3)
    (
        f0a1111,
        f1a1111,
        f2a1111,
        f0a1112,
        f1a1112,
        f2a1112,
        f0a1114,
        f1a1114,
        f2a1114,
        f0a1122,
        f1a1122,
        f2a1122,
        f0a1123,
        f1a1123,
        f2a1123,
        f0a1124,
        f1a1124,
        f2a1124,
        f0a1125,
        f1a1125,
        f2a1125,
        f0a1144,
        f1a1144,
        f2a1144,
        f0a1155,
        f1a1155,
        f2a1155,
        f0a1244,
        f1a1244,
        f2a1244,
        f0a1255,
        f1a1255,
        f2a1255,
        f0a1444,
        f1a1444,
        f2a1444,
        f0a1455,
        f1a1455,
        f2a1455,
        f0a4444,
        f1a4444,
        f2a4444,
    ) = force

    fea1111 = f0a1111 + f1a1111 * coro + f2a1111 * coro**2
    fea1112 = f0a1112 + f1a1112 * coro + f2a1112 * coro**2
    fea1114 = f0a1114 + f1a1114 * coro + f2a1114 * coro**2
    fea1122 = f0a1122 + f1a1122 * coro + f2a1122 * coro**2
    fea1123 = f0a1123 + f1a1123 * coro + f2a1123 * coro**2
    fea1124 = f0a1124 + f1a1124 * coro + f2a1124 * coro**2
    fea1125 = f0a1125 + f1a1125 * coro + f2a1125 * coro**2
    fea1144 = f0a1144 + f1a1144 * coro + f2a1144 * coro**2
    fea1155 = f0a1155 + f1a1155 * coro + f2a1155 * coro**2
    fea1244 = f0a1244 + f1a1244 * coro + f2a1244 * coro**2
    fea1255 = f0a1255 + f1a1255 * coro + f2a1255 * coro**2
    fea1444 = f0a1444 + f1a1444 * coro + f2a1444 * coro**2
    fea1455 = f0a1455 + f1a1455 * coro + f2a1455 * coro**2
    fea4444 = f0a4444 + f1a4444 * coro + f2a4444 * coro**2

    s2 = (
        (y4**4 + y5**4 + 2.0 * y4**2 * y5**2) * fea4444
        + (
            3.0 / 8.0 * sqrt3 * y2 * y5**3
            - 3.0 / 8.0 * sqrt3 * y3 * y4**2 * y5
            - 3.0 / 8.0 * sqrt3 * y3 * y5**3
            - 9.0 / 8.0 * y2 * y4 * y5**2
            - y3 * y4**3 / 8.0
            - y2 * y4**3 / 8.0
            - 9.0 / 8.0 * y3 * y4 * y5**2
            + y1 * y4**3
            + 3.0 / 8.0 * sqrt3 * y2 * y4**2 * y5
        )
        * fea1444
        + (
            3.0 / 4.0 * y2**2 * y4**2
            + 3.0 / 4.0 * y3**2 * y4**2
            + y1**2 * y5**2
            + y3**2 * y5**2 / 4.0
            - sqrt3 * y3**2 * y4 * y5 / 2.0
            + sqrt3 * y2**2 * y4 * y5 / 2.0
            + y2**2 * y5**2 / 4.0
        )
        * fea1155
    )

    s1 = (
        s2
        + (
            y3**2 * y4**2 / 4.0
            + 3.0 / 4.0 * y3**2 * y5**2
            + y1**2 * y4**2
            + y2**2 * y4**2 / 4.0
            + sqrt3 * y3**2 * y4 * y5 / 2.0
            - sqrt3 * y2**2 * y4 * y5 / 2.0
            + 3.0 / 4.0 * y2**2 * y5**2
        )
        * fea1144
        + (
            y1**3 * y4
            + sqrt3 * y2**3 * y5 / 2.0
            - sqrt3 * y3**3 * y5 / 2.0
            - y2**3 * y4 / 2.0
            - y3**3 * y4 / 2.0
        )
        * fea1114
        + (y2**4 + y1**4 + y3**4) * fea1111
        + (
            sqrt3 * y1 * y3 * y4 * y5
            + 3.0 / 2.0 * y2 * y3 * y5**2
            - y2 * y3 * y4**2 / 2.0
            + y1 * y2 * y4**2
            - sqrt3 * y1 * y2 * y4 * y5
            + y1 * y3 * y4**2
        )
        * fea1244
    )

    s2 = (
        s1
        + (
            y1 * y3 * y5**2
            + y1 * y2 * y5**2
            - sqrt3 * y1 * y3 * y4 * y5
            - y2 * y3 * y5**2 / 2.0
            + 3.0 / 2.0 * y2 * y3 * y4**2
            + sqrt3 * y1 * y2 * y4 * y5
        )
        * fea1255
        + (
            -y1 * y3**2 * y4 / 2.0
            + y1**2 * y3 * y4
            - sqrt3 * y1 * y3**2 * y5 / 2.0
            - sqrt3 * y2 * y3**2 * y5 / 2.0
            + y1**2 * y2 * y4
            + sqrt3 * y2**2 * y3 * y5 / 2.0
            - y2**2 * y3 * y4 / 2.0
            + sqrt3 * y1 * y2**2 * y5 / 2.0
            - y2 * y3**2 * y4 / 2.0
            - y1 * y2**2 * y4 / 2.0
        )
        * fea1124
        + (
            y1**2 * y2 * y5
            + sqrt3 * y1 * y3**2 * y4 / 2.0
            + sqrt3 * y1 * y2**2 * y4 / 2.0
            - sqrt3 * y2 * y3**2 * y4 / 2.0
            - sqrt3 * y2**2 * y3 * y4 / 2.0
            - y2**2 * y3 * y5 / 2.0
            + y2 * y3**2 * y5 / 2.0
            - y1 * y3**2 * y5 / 2.0
            + y1 * y2**2 * y5 / 2.0
            - y1**2 * y3 * y5
        )
        * fea1125
    )

    v4 = (
        s2
        + (y2 * y3**3 + y1**3 * y3 + y1**3 * y2 + y1 * y2**3 + y1 * y3**3 + y2**3 * y3)
        * fea1112
        + (y2**2 * y3**2 + y1**2 * y3**2 + y1**2 * y2**2) * fea1122
        + (y1 * y2**2 * y3 + y1**2 * y2 * y3 + y1 * y2 * y3**2) * fea1123
        + (
            5.0 / 8.0 * y2 * y4 * y5**2
            + sqrt3 * y2 * y5**3 / 8.0
            - sqrt3 * y3 * y4**2 * y5 / 8.0
            + sqrt3 * y2 * y4**2 * y5 / 8.0
            - 3.0 / 8.0 * y2 * y4**3
            + y1 * y4 * y5**2
            - sqrt3 * y3 * y5**3 / 8.0
            + 5.0 / 8.0 * y3 * y4 * y5**2
            - 3.0 / 8.0 * y3 * y4**3
        )
        * fea1455
    )
    return v4


def pot_v5(coro, y1, y2, y3, y4, y5, force):
    sqrt3 = jnp.sqrt(3)
    (
        f0a44444,
        f1a44444,
        f2a44444,
        f0a33455,
        f1a33455,
        f2a33455,
        f0a33445,
        f1a33445,
        f2a33445,
        f0a33345,
        f1a33345,
        f2a33345,
        f0a33344,
        f1a33344,
        f2a33344,
        f0a33334,
        f1a33334,
        f2a33334,
        f0a33333,
        f1a33333,
        f2a33333,
        f0a25555,
        f1a25555,
        f2a25555,
        f0a24455,
        f1a24455,
        f2a24455,
        f0a24445,
        f1a24445,
        f2a24445,
        f0a23333,
        f1a23333,
        f2a23333,
        f0a13455,
        f1a13455,
        f2a13455,
        f0a13445,
        f1a13445,
        f2a13445,
        f0a13345,
        f1a13345,
        f2a13345,
        f0a12355,
        f1a12355,
        f2a12355,
        f0a11334,
        f1a11334,
        f2a11334,
        f0a11333,
        f1a11333,
        f2a11333,
        f0a11255,
        f1a11255,
        f2a11255,
        f0a11245,
        f1a11245,
        f2a11245,
        f0a11234,
        f1a11234,
        f2a11234,
        f0a11233,
        f1a11233,
        f2a11233,
        f0a11135,
        f1a11135,
        f2a11135,
        f0a11134,
        f1a11134,
        f2a11134,
        f0a11123,
        f1a11123,
        f2a11123,
    ) = force

    fea44444 = f0a44444 + f1a44444 * coro + f2a44444 * coro**2
    fea33455 = f0a33455 + f1a33455 * coro + f2a33455 * coro**2
    fea33445 = f0a33445 + f1a33445 * coro + f2a33445 * coro**2
    fea33345 = f0a33345 + f1a33345 * coro + f2a33345 * coro**2
    fea33344 = f0a33344 + f1a33344 * coro + f2a33344 * coro**2
    fea33334 = f0a33334 + f1a33334 * coro + f2a33334 * coro**2
    fea33333 = f0a33333 + f1a33333 * coro + f2a33333 * coro**2
    fea25555 = f0a25555 + f1a25555 * coro + f2a25555 * coro**2
    fea24455 = f0a24455 + f1a24455 * coro + f2a24455 * coro**2
    fea24445 = f0a24445 + f1a24445 * coro + f2a24445 * coro**2
    fea23333 = f0a23333 + f1a23333 * coro + f2a23333 * coro**2
    fea13455 = f0a13455 + f1a13455 * coro + f2a13455 * coro**2
    fea13445 = f0a13445 + f1a13445 * coro + f2a13445 * coro**2
    fea13345 = f0a13345 + f1a13345 * coro + f2a13345 * coro**2
    fea12355 = f0a12355 + f1a12355 * coro + f2a12355 * coro**2
    fea11334 = f0a11334 + f1a11334 * coro + f2a11334 * coro**2
    fea11333 = f0a11333 + f1a11333 * coro + f2a11333 * coro**2
    fea11255 = f0a11255 + f1a11255 * coro + f2a11255 * coro**2
    fea11245 = f0a11245 + f1a11245 * coro + f2a11245 * coro**2
    fea11234 = f0a11234 + f1a11234 * coro + f2a11234 * coro**2
    fea11233 = f0a11233 + f1a11233 * coro + f2a11233 * coro**2
    fea11135 = f0a11135 + f1a11135 * coro + f2a11135 * coro**2
    fea11134 = f0a11134 + f1a11134 * coro + f2a11134 * coro**2
    fea11123 = f0a11123 + f1a11123 * coro + f2a11123 * coro**2

    s3 = (
        (y4**5 - 2.0 * y4**3 * y5**2 - 3.0 * y4 * y5**4) * fea44444
        + (
            -4.0 * y3 * y4 * y5**3 * sqrt3
            + 9.0 * y1 * y4**2 * y5**2
            - 3.0 / 2.0 * y1 * y4**4
            + 4.0 * y2 * y4 * y5**3 * sqrt3
            + 3.0 * y2 * y4**4
            + 5.0 / 2.0 * y1 * y5**4
            + 3.0 * y3 * y4**4
            + y2 * y5**4
            + y3 * y5**4
        )
        * fea25555
        + (
            -y2 * y4**4
            + y3 * y4**2 * y5**2
            - 2.0 * y2 * y4 * y5**3 * sqrt3
            - y3 * y4**4
            - 7.0 / 2.0 * y1 * y4**2 * y5**2
            - 3.0 / 4.0 * y1 * y5**4
            + 2.0 * y3 * y4 * y5**3 * sqrt3
            + y2 * y4**2 * y5**2
            + 5.0 / 4.0 * y1 * y4**4
        )
        * fea24455
    )

    s2 = (
        s3
        + (
            y2 * y4**3 * y5
            - 3.0 * y3 * y4 * y5**3
            + 2.0 / 3.0 * y3 * y4**4 * sqrt3
            + 3.0 / 4.0 * y1 * y5**4 * sqrt3
            + 3.0 * y2 * y4 * y5**3
            - 7.0 / 12.0 * y1 * y4**4 * sqrt3
            + 3.0 / 2.0 * y1 * y4**2 * y5**2 * sqrt3
            - y3 * y4**3 * y5
            + 2.0 / 3.0 * y2 * y4**4 * sqrt3
        )
        * fea24445
        + (
            -(y2**2) * y5**3
            + y3**2 * y4**2 * y5
            + y3**2 * y5**3
            + 4.0 / 9.0 * y2**2 * y4**3 * sqrt3
            - 5.0 / 9.0 * y1**2 * y4**3 * sqrt3
            + 4.0 / 9.0 * y3**2 * y4**3 * sqrt3
            - y2**2 * y4**2 * y5
            - y1**2 * y4 * y5**2 * sqrt3
        )
        * fea33445
        + (
            y3**2 * y4 * y5**2
            - y1**2 * y4**3 / 3.0
            - y3**2 * y4**3 / 3.0
            + y1**2 * y4 * y5**2
            + y2**2 * y4 * y5**2
            - y2**2 * y4**3 / 3.0
        )
        * fea33455
    )

    s1 = (
        s2
        + (
            -(y2**3) * y4 * y5
            + y3**3 * y4 * y5
            + y2**3 * y5**2 * sqrt3 / 3.0
            + y1**3 * y4**2 * sqrt3 / 2.0
            + y3**3 * y5**2 * sqrt3 / 3.0
            - y1**3 * y5**2 * sqrt3 / 6.0
        )
        * fea33345
        + (
            y3**3 * y4**2
            + y3**3 * y5**2
            + y2**3 * y4**2
            + y2**3 * y5**2
            + y1**3 * y5**2
            + y1**3 * y4**2
        )
        * fea33344
        + (
            y3**4 * y4
            + sqrt3 * y3**4 * y5
            + y2**4 * y4
            - 2.0 * y1**4 * y4
            - sqrt3 * y2**4 * y5
        )
        * fea33334
        + (y2**5 + y3**5 + y1**5) * fea33333
        + (
            -4.0 / 9.0 * y1 * y2 * y4**3 * sqrt3
            - y1 * y2 * y5**3
            + y1 * y3 * y4**2 * y5
            + y2 * y3 * y4 * y5**2 * sqrt3
            - y1 * y2 * y4**2 * y5
            + 5.0 / 9.0 * y2 * y3 * y4**3 * sqrt3
            - 4.0 / 9.0 * y1 * y3 * y4**3 * sqrt3
            + y1 * y3 * y5**3
        )
        * fea13445
        + (
            y2 * y3 * y4 * y5**2
            + y1 * y2 * y4 * y5**2
            - y2 * y3 * y4**3 / 3.0
            - y1 * y2 * y4**3 / 3.0
            - y1 * y3 * y4**3 / 3.0
            + y1 * y3 * y4 * y5**2
        )
        * fea13455
    )

    s3 = (
        s1
        + (
            y1**2 * y3 * y5**2
            + y2**2 * y3 * y4**2
            + y2**2 * y3 * y5**2
            + y1 * y2**2 * y5**2
            + y1**2 * y2 * y5**2
            + y1 * y2**2 * y4**2
            + y2 * y3**2 * y4**2
            + y1 * y3**2 * y4**2
            + y1**2 * y3 * y4**2
            + y1**2 * y2 * y4**2
            + y1 * y3**2 * y5**2
            + y2 * y3**2 * y5**2
        )
        * fea11255
        + (
            2.0 / 3.0 * y1**2 * y3 * y4**2 * sqrt3
            + y1 * y3**2 * y5**2 * sqrt3 / 2.0
            + y1 * y2**2 * y5**2 * sqrt3 / 2.0
            + y2**2 * y3 * y5**2 * sqrt3 / 2.0
            - y1 * y2**2 * y4 * y5
            + y2 * y3**2 * y4 * y5
            + y1 * y3**2 * y4 * y5
            - y2**2 * y3 * y4 * y5
            + y2 * y3**2 * y4**2 * sqrt3 / 6.0
            + y1 * y3**2 * y4**2 * sqrt3 / 6.0
            + y1 * y2**2 * y4**2 * sqrt3 / 6.0
            + 2.0 / 3.0 * y1**2 * y2 * y4**2 * sqrt3
            + y2 * y3**2 * y5**2 * sqrt3 / 2.0
            + y2**2 * y3 * y4**2 * sqrt3 / 6.0
        )
        * fea13345
    )
    s4 = (
        s3
        + (
            y1**2 * y2 * y4 * y5
            + y1**2 * y3 * y4**2 * sqrt3 / 3.0
            + y1**2 * y2 * y4**2 * sqrt3 / 3.0
            - y1 * y2**2 * y4**2 * sqrt3 / 6.0
            + y2 * y3**2 * y4 * y5
            - y2**2 * y3 * y4 * y5
            - y1**2 * y3 * y4 * y5
            + y2 * y3**2 * y4**2 * sqrt3 / 3.0
            + y1 * y2**2 * y5**2 * sqrt3 / 2.0
            - y1 * y3**2 * y4**2 * sqrt3 / 6.0
            + y2**2 * y3 * y4**2 * sqrt3 / 3.0
            + y1 * y3**2 * y5**2 * sqrt3 / 2.0
        )
        * fea11245
    )
    s2 = (
        s4
        + (
            -(y1**3) * y2 * y5
            + y1**3 * y3 * y5
            + y2**3 * y3 * y5 / 2.0
            - y1 * y2**3 * y4 * sqrt3 / 2.0
            - y1 * y2**3 * y5 / 2.0
            - y2 * y3**3 * y5 / 2.0
            + y1 * y3**3 * y5 / 2.0
            + y2**3 * y3 * y4 * sqrt3 / 2.0
            + y2 * y3**3 * y4 * sqrt3 / 2.0
            - y1 * y3**3 * y4 * sqrt3 / 2.0
        )
        * fea11135
        + (
            y1**3 * y3 * y4
            - y2**3 * y3 * y4 / 2.0
            + y1**3 * y2 * y4
            - y2 * y3**3 * y4 / 2.0
            - y1 * y3**3 * y4 / 2.0
            + y1 * y2**3 * y5 * sqrt3 / 2.0
            + y2**3 * y3 * y5 * sqrt3 / 2.0
            - y2 * y3**3 * y5 * sqrt3 / 2.0
            - y1 * y2**3 * y4 / 2.0
            - y1 * y3**3 * y5 * sqrt3 / 2.0
        )
        * fea11134
    )

    v5 = (
        s2
        + (y1 * y2**4 + y1**4 * y3 + y1**4 * y2 + y2**4 * y3 + y2 * y3**4 + y1 * y3**4)
        * fea23333
        + (
            -2.0 * y2**2 * y3**2 * y4
            + y1**2 * y2**2 * y4
            - sqrt3 * y1**2 * y3**2 * y5
            + sqrt3 * y1**2 * y2**2 * y5
            + y1**2 * y3**2 * y4
        )
        * fea11334
        + (
            y1**2 * y3**3
            + y1**3 * y3**2
            + y2**2 * y3**3
            + y1**2 * y2**3
            + y1**3 * y2**2
            + y2**3 * y3**2
        )
        * fea11333
        + (y1 * y2 * y3 * y4**2 + y1 * y2 * y3 * y5**2) * fea12355
        + (
            -y1 * y2 * y3**2 * y4 / 2.0
            - y1 * y2**2 * y3 * y4 / 2.0
            - sqrt3 * y1 * y2 * y3**2 * y5 / 2.0
            + y1**2 * y2 * y3 * y4
            + sqrt3 * y1 * y2**2 * y3 * y5 / 2.0
        )
        * fea11234
        + (y1 * y2**3 * y3 + y1 * y2 * y3**3 + y1**3 * y2 * y3) * fea11123
        + (y1**2 * y2**2 * y3 + y1 * y2**2 * y3**2 + y1**2 * y2 * y3**2) * fea11233
    )
    return v5


def pot_v6(coro, y1, y2, y3, y4, y5, force):
    sqrt3 = jnp.sqrt(3)
    f0a555555
    f1a555555
    f2a555555
    f0a444444
    f1a444444
    f2a444444
    f0a335555
    f1a335555
    f2a335555
    f0a334455
    f1a334455
    f2a334455
    f0a334445
    f1a334445
    f2a334445
    f0a333555
    f1a333555
    f2a333555
    f0a333333
    f1a333333
    f2a333333
    f0a244555
    f1a244555
    f2a244555
    f0a244455
    f1a244455
    f2a244455
    f0a233445
    f1a233445
    f2a233445
    f0a233444
    f1a233444
    f2a233444
    f0a233345
    f1a233345
    f2a233345
    f0a233344
    f1a233344
    f2a233344
    f0a233335
    f1a233335
    f2a233335
    f0a223355
    f1a223355
    f2a223355
    f0a222335
    f1a222335
    f2a222335
    f0a222334
    f1a222334
    f2a222334
    f0a222333
    f1a222333
    f2a222333
    f0a222255
    f1a222255
    f2a222255
    f0a222245
    f1a222245
    f2a222245
    f0a222233
    f1a222233
    f2a222233
    f0a222224
    f1a222224
    f2a222224
    f0a145555
    f1a145555
    f2a145555
    f0a134444
    f1a134444
    f2a134444
    f0a133444
    f1a133444
    f2a133444
    f0a133345
    f1a133345
    f2a133345
    f0a133334
    f1a133334
    f2a133334
    f0a133333
    f1a133333
    f2a133333
    f0a124555
    f1a124555
    f2a124555
    f0a124455
    f1a124455
    f2a124455
    f0a123455
    f1a123455
    f2a123455
    f0a123345
    f1a123345
    f2a123345
    f0a113555
    f1a113555
    f2a113555
    f0a113345
    f1a113345
    f2a113345
    f0a112355
    f1a112355
    f2a112355
    f0a112335
    f1a112335
    f2a112335
    f0a112233
    f1a112233
    f2a112233
    f0a111444
    f1a111444
    f2a111444
    f0a111234
    f1a111234
    f2a111234
    f0a111233
    f1a111233
    f2a111233
    f0a111123
    f1a111123
    f2a111123 = force

    fea555555 = f0a555555 + f1a555555 * coro + f2a555555 * coro**2
    fea444444 = f0a444444 + f1a444444 * coro + f2a444444 * coro**2
    fea335555 = f0a335555 + f1a335555 * coro + f2a335555 * coro**2
    fea334455 = f0a334455 + f1a334455 * coro + f2a334455 * coro**2
    fea334445 = f0a334445 + f1a334445 * coro + f2a334445 * coro**2
    fea333555 = f0a333555 + f1a333555 * coro + f2a333555 * coro**2
    fea333333 = f0a333333 + f1a333333 * coro + f2a333333 * coro**2
    fea244555 = f0a244555 + f1a244555 * coro + f2a244555 * coro**2
    fea244455 = f0a244455 + f1a244455 * coro + f2a244455 * coro**2
    fea233445 = f0a233445 + f1a233445 * coro + f2a233445 * coro**2
    fea233444 = f0a233444 + f1a233444 * coro + f2a233444 * coro**2
    fea233345 = f0a233345 + f1a233345 * coro + f2a233345 * coro**2
    fea233344 = f0a233344 + f1a233344 * coro + f2a233344 * coro**2
    fea233335 = f0a233335 + f1a233335 * coro + f2a233335 * coro**2
    fea223355 = f0a223355 + f1a223355 * coro + f2a223355 * coro**2
    fea222335 = f0a222335 + f1a222335 * coro + f2a222335 * coro**2
    fea222334 = f0a222334 + f1a222334 * coro + f2a222334 * coro**2
    fea222333 = f0a222333 + f1a222333 * coro + f2a222333 * coro**2
    fea222255 = f0a222255 + f1a222255 * coro + f2a222255 * coro**2
    fea222245 = f0a222245 + f1a222245 * coro + f2a222245 * coro**2
    fea222233 = f0a222233 + f1a222233 * coro + f2a222233 * coro**2
    fea222224 = f0a222224 + f1a222224 * coro + f2a222224 * coro**2
    fea145555 = f0a145555 + f1a145555 * coro + f2a145555 * coro**2
    fea134444 = f0a134444 + f1a134444 * coro + f2a134444 * coro**2
    fea133444 = f0a133444 + f1a133444 * coro + f2a133444 * coro**2
    fea133345 = f0a133345 + f1a133345 * coro + f2a133345 * coro**2
    fea133334 = f0a133334 + f1a133334 * coro + f2a133334 * coro**2
    fea133333 = f0a133333 + f1a133333 * coro + f2a133333 * coro**2
    fea124555 = f0a124555 + f1a124555 * coro + f2a124555 * coro**2
    fea124455 = f0a124455 + f1a124455 * coro + f2a124455 * coro**2
    fea123455 = f0a123455 + f1a123455 * coro + f2a123455 * coro**2
    fea123345 = f0a123345 + f1a123345 * coro + f2a123345 * coro**2
    fea113555 = f0a113555 + f1a113555 * coro + f2a113555 * coro**2
    fea113345 = f0a113345 + f1a113345 * coro + f2a113345 * coro**2
    fea112355 = f0a112355 + f1a112355 * coro + f2a112355 * coro**2
    fea112335 = f0a112335 + f1a112335 * coro + f2a112335 * coro**2
    fea112233 = f0a112233 + f1a112233 * coro + f2a112233 * coro**2
    fea111444 = f0a111444 + f1a111444 * coro + f2a111444 * coro**2
    fea111234 = f0a111234 + f1a111234 * coro + f2a111234 * coro**2
    fea111233 = f0a111233 + f1a111233 * coro + f2a111233 * coro**2
    fea111123 = f0a111123 + f1a111123 * coro + f2a111123 * coro**2

    s3 = (
        (
            y2**3 * y4**3 * sqrt3
            - y2**3 * y4**2 * y5
            + y3**3 * y4**2 * y5
            - 5.0 / 3.0 * y2**3 * y4 * y5**2 * sqrt3
            + y3**3 * y4**3 * sqrt3
            - 5.0 / 3.0 * y3**3 * y4 * y5**2 * sqrt3
            - y2**3 * y5**3
            + y3**3 * y5**3
            - 8.0 / 3.0 * y1**3 * y4 * y5**2 * sqrt3
        )
        * fea333555
        + (
            y1**4 * y5**2 * sqrt3 / 2.0
            + y2**4 * y4 * y5
            + y2**4 * y4**2 * sqrt3 / 3.0
            + y3**4 * y4**2 * sqrt3 / 3.0
            - y3**4 * y4 * y5
            - y1**4 * y4**2 * sqrt3 / 6.0
        )
        * fea222245
        + (y1 * y3**5 + y1 * y2**5 + y2**5 * y3 + y1**5 * y3 + y1**5 * y2 + y2 * y3**5)
        * fea133333
        + (
            y1**4 * y3 * y4
            - 2.0 * y2**4 * y3 * y4
            + y1**4 * y2 * y4
            + y1 * y2**4 * y5 * sqrt3
            + y1 * y3**4 * y4
            - 2.0 * y2 * y3**4 * y4
            + y1**4 * y2 * y5 * sqrt3
            - y1 * y3**4 * y5 * sqrt3
            - y1**4 * y3 * y5 * sqrt3
            + y1 * y2**4 * y4
        )
        * fea133334
        + (-y1 * y2 * y3 * y4**3 / 3.0 + y1 * y2 * y3 * y4 * y5**2) * fea123455
    )

    s4 = (
        s3
        + (
            2.0 / 3.0 * sqrt3 * y1 * y2**2 * y3**2 * y4
            - y1**2 * y2**2 * y3 * y5
            - sqrt3 * y1**2 * y2**2 * y3 * y4 / 3.0
            + y1**2 * y2 * y3**2 * y5
            - sqrt3 * y1**2 * y2 * y3**2 * y4 / 3.0
        )
        * fea112335
        + (
            y1 * y2**2 * y3 * y5**2
            + y1 * y2 * y3**2 * y5**2
            + y1 * y2 * y3**2 * y4**2
            + y1 * y2**2 * y3 * y4**2
            + y1**2 * y2 * y3 * y4**2
            + y1**2 * y2 * y3 * y5**2
        )
        * fea112355
    )

    s2 = (
        s4
        + (
            y2**3 * y3**2 * y5
            - y1**3 * y2**2 * y5 / 2.0
            - y1**2 * y3**3 * y5 / 2.0
            - y2**2 * y3**3 * y5
            + y1**3 * y2**2 * y4 * sqrt3 / 2.0
            - y1**2 * y2**3 * y4 * sqrt3 / 2.0
            + y1**3 * y3**2 * y5 / 2.0
            + y1**2 * y2**3 * y5 / 2.0
            + y1**3 * y3**2 * y4 * sqrt3 / 2.0
            - y1**2 * y3**3 * y4 * sqrt3 / 2.0
        )
        * fea222335
        + (
            -(y1**2) * y2**2 * y5**2 * sqrt3 / 2.0
            - y1**2 * y3**2 * y5**2 * sqrt3 / 2.0
            - y1**2 * y2**2 * y4**2 * sqrt3 / 6.0
            - y1**2 * y2**2 * y4 * y5
            - 2.0 / 3.0 * y2**2 * y3**2 * y4**2 * sqrt3
            + y1**2 * y3**2 * y4 * y5
            - y1**2 * y3**2 * y4**2 * sqrt3 / 6.0
        )
        * fea113345
        + (
            y2**2 * y3**2 * y5**2
            + y2**2 * y3**2 * y4**2
            + y1**2 * y2**2 * y5**2
            + y1**2 * y3**2 * y4**2
            + y1**2 * y3**2 * y5**2
            + y1**2 * y2**2 * y4**2
        )
        * fea223355
    )

    s3 = (
        s2
        + (
            y1 * y2 * y3**2 * y4**2 * sqrt3 / 6.0
            + y1 * y2 * y3**2 * y4 * y5
            + y1 * y2 * y3**2 * y5**2 * sqrt3 / 2.0
            + 2.0 / 3.0 * y1**2 * y2 * y3 * y4**2 * sqrt3
            - y1 * y2**2 * y3 * y4 * y5
            + y1 * y2**2 * y3 * y4**2 * sqrt3 / 6.0
            + y1 * y2**2 * y3 * y5**2 * sqrt3 / 2.0
        )
        * fea123345
        + (
            -(y1**3) * y2**2 * y5 * sqrt3 / 2.0
            - y1**3 * y2**2 * y4 / 2.0
            - y1**3 * y3**2 * y4 / 2.0
            - y1**2 * y2**3 * y4 / 2.0
            + y1**3 * y3**2 * y5 * sqrt3 / 2.0
            - y1**2 * y3**3 * y4 / 2.0
            + y2**3 * y3**2 * y4
            - y1**2 * y2**3 * y5 * sqrt3 / 2.0
            + y2**2 * y3**3 * y4
            + y1**2 * y3**3 * y5 * sqrt3 / 2.0
        )
        * fea222334
        + (
            3.0 * y3**2 * y4**4
            + 5.0 / 2.0 * y1**2 * y5**4
            + y2**2 * y5**4
            + 3.0 * y2**2 * y4**4
            - 4.0 * y3**2 * y4 * y5**3 * sqrt3
            + y3**2 * y5**4
            + 9.0 * y1**2 * y4**2 * y5**2
            - 3.0 / 2.0 * y1**2 * y4**4
            + 4.0 * y2**2 * y4 * y5**3 * sqrt3
        )
        * fea335555
        + (y1**3 * y2**3 + y1**3 * y3**3 + y2**3 * y3**3) * fea222333
    )

    s4 = (
        s3
        + (
            y3 * y4**5 / 5.0
            - y2 * y4**4 * y5 * sqrt3 / 2.0
            - 2.0 / 5.0 * y1 * y4**5
            - 2.0 * y1 * y4**3 * y5**2
            - 3.0 / 10.0 * y2 * y5**5 * sqrt3
            + y3 * y4**3 * y5**2
            + y3 * y4**4 * y5 * sqrt3 / 2.0
            + y2 * y4**3 * y5**2
            + 3.0 / 10.0 * y3 * y5**5 * sqrt3
            + y2 * y4**5 / 5.0
        )
        * fea244455
        + (
            y2**5 * y4
            - 2.0 * y1**5 * y4
            - sqrt3 * y2**5 * y5
            + y3**5 * y4
            + sqrt3 * y3**5 * y5
        )
        * fea222224
    )

    s5 = (
        s4
        + (
            -y3 * y5**5 * sqrt3 / 5.0
            + y2 * y5**5 * sqrt3 / 5.0
            + y1 * y4 * y5**4
            - 7.0 / 15.0 * y2 * y4**5
            + y2 * y4**4 * y5 * sqrt3 / 3.0
            - y3 * y4**4 * y5 * sqrt3 / 3.0
            + y3 * y4 * y5**4
            + y2 * y4 * y5**4
            + 2.0 * y1 * y4**3 * y5**2
            - 7.0 / 15.0 * y3 * y4**5
            - y1 * y4**5 / 15.0
        )
        * fea145555
    )

    s1 = (
        s5
        + (
            -sqrt3 * y1 * y2 * y3**3 * y5 / 2.0
            + y1**3 * y2 * y3 * y4
            + sqrt3 * y1 * y2**3 * y3 * y5 / 2.0
            - y1 * y2**3 * y3 * y4 / 2.0
            - y1 * y2 * y3**3 * y4 / 2.0
        )
        * fea111234
        + (
            y3 * y4**4 * y5 / 3.0
            + y3 * y4**5 * sqrt3 / 18.0
            - y2 * y4**4 * y5 / 3.0
            - y2 * y4 * y5**4 * sqrt3 / 2.0
            - y3 * y4**2 * y5**3
            + 2.0 / 9.0 * y1 * y4**5 * sqrt3
            + y2 * y4**5 * sqrt3 / 18.0
            + y2 * y4**2 * y5**3
            - 2.0 / 3.0 * y1 * y4**3 * y5**2 * sqrt3
            - y3 * y4 * y5**4 * sqrt3 / 2.0
        )
        * fea244555
        + (
            y1 * y2 * y4**2 * y5**2
            - 3.0 / 4.0 * y2 * y3 * y4**4
            - y1 * y2 * y5**4
            - y1 * y3 * y5**4
            + 5.0 / 4.0 * y2 * y3 * y5**4
            + y1 * y3 * y4**2 * y5**2
            - 7.0 / 2.0 * y2 * y3 * y4**2 * y5**2
            - 2.0 * y1 * y2 * y4**3 * y5 * sqrt3
            + 2.0 * y1 * y3 * y4**3 * y5 * sqrt3
        )
        * fea124455
    )

    s3 = (
        s1
        + (y2**6 + y1**6 + y3**6) * fea333333
        + (y1 * y2**4 * y3 + y1**4 * y2 * y3 + y1 * y2 * y3**4) * fea111123
        + fea112233 * y1**2 * y2**2 * y3**2
        + (
            y1**4 * y4**2
            + y2**4 * y4**2
            + y2**4 * y5**2
            + y3**4 * y4**2
            + y1**4 * y5**2
            + y3**4 * y5**2
        )
        * fea222255
    )
    s4 = (
        s3
        + (
            3.0 * y1 * y3 * y5**4
            + y1 * y3 * y4**4
            + 9.0 * y2 * y3 * y4**2 * y5**2
            - 3.0 / 2.0 * y2 * y3 * y5**4
            - 4.0 * y1 * y3 * y4**3 * y5 * sqrt3
            + y1 * y2 * y4**4
            + 4.0 * y1 * y2 * y4**3 * y5 * sqrt3
            + 3.0 * y1 * y2 * y5**4
            + 5.0 / 2.0 * y2 * y3 * y4**4
        )
        * fea134444
        + (
            -y1 * y3**2 * y5**3 * sqrt3 / 3.0
            - 7.0 / 3.0 * y1**2 * y3 * y4 * y5**2
            + 5.0 / 3.0 * y1 * y2**2 * y4**2 * y5 * sqrt3
            - 13.0 / 3.0 * y2**2 * y3 * y4 * y5**2
            - 4.0 / 3.0 * y2 * y3**2 * y5**3 * sqrt3
            - 7.0 / 3.0 * y1**2 * y2 * y4 * y5**2
            - 16.0 / 3.0 * y1 * y3**2 * y4 * y5**2
            + 4.0 / 3.0 * y1**2 * y3 * y4**2 * y5 * sqrt3
            + 4.0 / 3.0 * y2**2 * y3 * y5**3 * sqrt3
            + 3.0 * y1**2 * y2 * y4**3
            + y2 * y3**2 * y4**3
            + y1 * y2**2 * y5**3 * sqrt3 / 3.0
            + y2**2 * y3 * y4**3
            - 13.0 / 3.0 * y2 * y3**2 * y4 * y5**2
            - 5.0 / 3.0 * y1 * y3**2 * y4**2 * y5 * sqrt3
            - 4.0 / 3.0 * y1**2 * y2 * y4**2 * y5 * sqrt3
            + 3.0 * y1**2 * y3 * y4**3
            - 16.0 / 3.0 * y1 * y2**2 * y4 * y5**2
        )
        * fea233444
    )

    s5 = (
        s4
        + (
            2.0 * y1 * y3**2 * y5**3
            + 4.0 * y2 * y3**2 * y5**3
            + 4.0 * y2**2 * y3 * y4 * y5**2 * sqrt3
            - 2.0 * y1 * y2**2 * y5**3
            + y1**2 * y3 * y4 * y5**2 * sqrt3
            + 6.0 * y1 * y3**2 * y4**2 * y5
            - 6.0 * y1 * y2**2 * y4**2 * y5
            - 3.0 * y1**2 * y3 * y4**2 * y5
            + y1**2 * y2 * y4 * y5**2 * sqrt3
            + 4.0 * y1 * y3**2 * y4 * y5**2 * sqrt3
            - 3.0 * y1**2 * y2 * y4**3 * sqrt3
            - 4.0 * y2**2 * y3 * y5**3
            + 3.0 * y1**2 * y2 * y4**2 * y5
            - y1**2 * y2 * y5**3
            + y1**2 * y3 * y5**3
            - 3.0 * y1**2 * y3 * y4**3 * sqrt3
            + 4.0 * y2 * y3**2 * y4 * y5**2 * sqrt3
            + 4.0 * y1 * y2**2 * y4 * y5**2 * sqrt3
        )
        * fea113555
    )

    s2 = (
        s5
        + (
            -2.0 / 3.0 * y3**2 * y4**4 * sqrt3
            - 3.0 / 2.0 * y1**2 * y4**2 * y5**2 * sqrt3
            - 3.0 / 4.0 * y1**2 * y5**4 * sqrt3
            - y2**2 * y4**3 * y5
            + 7.0 / 12.0 * y1**2 * y4**4 * sqrt3
            + y3**2 * y4**3 * y5
            + 3.0 * y3**2 * y4 * y5**3
            - 2.0 / 3.0 * y2**2 * y4**4 * sqrt3
            - 3.0 * y2**2 * y4 * y5**3
        )
        * fea334445
        + (
            -3.0 * y1 * y3 * y4**3 * y5
            + 2.0 / 3.0 * y1 * y2 * y5**4 * sqrt3
            - y1 * y3 * y4 * y5**3
            + 2.0 / 3.0 * y1 * y3 * y5**4 * sqrt3
            + 3.0 * y1 * y2 * y4**3 * y5
            - 7.0 / 12.0 * y2 * y3 * y5**4 * sqrt3
            + 3.0 / 2.0 * y2 * y3 * y4**2 * y5**2 * sqrt3
            + y1 * y2 * y4 * y5**3
            + 3.0 / 4.0 * y2 * y3 * y4**4 * sqrt3
        )
        * fea124555
        + (
            2.0 * y3**2 * y4 * y5**3 * sqrt3
            - 7.0 / 2.0 * y1**2 * y4**2 * y5**2
            + y2**2 * y4**2 * y5**2
            - y2**2 * y4**4
            - y3**2 * y4**4
            - 2.0 * y2**2 * y4 * y5**3 * sqrt3
            - 3.0 / 4.0 * y1**2 * y5**4
            + 5.0 / 4.0 * y1**2 * y4**4
            + y3**2 * y4**2 * y5**2
        )
        * fea334455
    )
    s3 = (
        s2
        + (-6.0 * y4**2 * y5**4 + 9.0 * y4**4 * y5**2 + y5**6) * fea555555
        + (
            y2 * y3**3 * y4**2
            + y2 * y3**3 * y5**2
            + y1 * y3**3 * y4**2
            + y1 * y2**3 * y4**2
            + y1**3 * y2 * y4**2
            + y1 * y2**3 * y5**2
            + y1**3 * y3 * y5**2
            + y1**3 * y3 * y4**2
            + y1**3 * y2 * y5**2
            + y2**3 * y3 * y4**2
            + y1 * y3**3 * y5**2
            + y2**3 * y3 * y5**2
        )
        * fea233344
        + (
            y1 * y2**3 * y5**2 * sqrt3 / 6.0
            - y2**3 * y3 * y5**2 * sqrt3 / 3.0
            - y2 * y3**3 * y5**2 * sqrt3 / 3.0
            + y1**3 * y2 * y4 * y5
            - y1**3 * y2 * y5**2 * sqrt3 / 3.0
            - y1**3 * y3 * y4 * y5
            - y1**3 * y3 * y5**2 * sqrt3 / 3.0
            - y1 * y3**3 * y4**2 * sqrt3 / 2.0
            + y1 * y3**3 * y5**2 * sqrt3 / 6.0
            - y2**3 * y3 * y4 * y5
            + y2 * y3**3 * y4 * y5
            - y1 * y2**3 * y4**2 * sqrt3 / 2.0
        )
        * fea233345
        + (
            -3.0 * y2**3 * y4 * y5**2
            + y3**3 * y4**3
            - 3.0 * y3**3 * y4 * y5**2
            - 3.0 * y1**3 * y4 * y5**2
            + y2**3 * y4**3
            + y1**3 * y4**3
        )
        * fea111444
        + (
            y1 * y2**3 * y3**2
            + y1**3 * y2**2 * y3
            + y1**2 * y2**3 * y3
            + y1 * y2**2 * y3**3
            + y1**2 * y2 * y3**3
            + y1**3 * y2 * y3**2
        )
        * fea111233
    )

    s4 = (
        s3
        + (9.0 * y4**2 * y5**4 - 6.0 * y4**4 * y5**2 + y4**6) * fea444444
        + (
            -5.0 / 3.0 * y1 * y2**2 * y4**2 * y5 * sqrt3
            + y1 * y2**2 * y4**3
            - 4.0 / 3.0 * y1**2 * y3 * y4**2 * y5 * sqrt3
            - 2.0 * y1**2 * y2 * y4**3
            - y1 * y2**2 * y5**3 * sqrt3 / 3.0
            + 4.0 / 3.0 * y2**2 * y3 * y4 * y5**2
            - 4.0 / 3.0 * y2**2 * y3 * y5**3 * sqrt3
            - 2.0 * y1**2 * y3 * y4**3
            + 7.0 / 3.0 * y1 * y2**2 * y4 * y5**2
            - 2.0 / 3.0 * y1**2 * y3 * y4 * y5**2
            + y1 * y3**2 * y4**3
            + 4.0 / 3.0 * y2 * y3**2 * y5**3 * sqrt3
            + y1 * y3**2 * y5**3 * sqrt3 / 3.0
            + 4.0 / 3.0 * y1**2 * y2 * y4**2 * y5 * sqrt3
            + 4.0 / 3.0 * y2 * y3**2 * y4 * y5**2
            + 5.0 / 3.0 * y1 * y3**2 * y4**2 * y5 * sqrt3
            - 2.0 / 3.0 * y1**2 * y2 * y4 * y5**2
            + 7.0 / 3.0 * y1 * y3**2 * y4 * y5**2
        )
        * fea133444
    )

    s5 = (
        s4
        + (
            -(y1**3) * y2 * y4 * y5
            + 2.0 / 3.0 * y2**3 * y3 * y5**2 * sqrt3
            + y1 * y3**3 * y4**2 * sqrt3 / 2.0
            + y1**3 * y3 * y4**2 * sqrt3 / 2.0
            + y1**3 * y3 * y5**2 * sqrt3 / 6.0
            + y1**3 * y2 * y5**2 * sqrt3 / 6.0
            + y1**3 * y3 * y4 * y5
            + y1 * y2**3 * y5**2 * sqrt3 / 6.0
            + y1**3 * y2 * y4**2 * sqrt3 / 2.0
            + 2.0 / 3.0 * y2 * y3**3 * y5**2 * sqrt3
            - y1 * y2**3 * y4 * y5
            + y1 * y2**3 * y4**2 * sqrt3 / 2.0
            + y1 * y3**3 * y5**2 * sqrt3 / 6.0
            + y1 * y3**3 * y4 * y5
        )
        * fea133345
    )

    v6 = (
        s5
        + (
            -(y2**2) * y3 * y4**2 * y5
            + y1**2 * y3 * y4 * y5**2 * sqrt3 / 3.0
            + y2 * y3**2 * y4**2 * y5
            + y2 * y3**2 * y5**3
            - y1 * y2**2 * y5**3
            + 4.0 / 3.0 * y2**2 * y3 * y4 * y5**2 * sqrt3
            + 4.0 / 3.0 * y2 * y3**2 * y4 * y5**2 * sqrt3
            - y1 * y2**2 * y4**2 * y5
            + 4.0 / 3.0 * y1 * y3**2 * y4 * y5**2 * sqrt3
            - y2**2 * y3 * y5**3
            + y1 * y3**2 * y5**3
            + y1**2 * y2 * y4 * y5**2 * sqrt3 / 3.0
            - y1**2 * y2 * y4**3 * sqrt3
            + y1 * y3**2 * y4**2 * y5
            - y1**2 * y3 * y4**3 * sqrt3
            + 4.0 / 3.0 * y1 * y2**2 * y4 * y5**2 * sqrt3
        )
        * fea233445
        + (
            y2 * y3**4 * y4 * sqrt3
            - y1**4 * y2 * y5
            + y2**4 * y3 * y4 * sqrt3
            - y1**4 * y3 * y4 * sqrt3
            + y2 * y3**4 * y5
            - 2.0 * y1 * y2**4 * y5
            + 2.0 * y1 * y3**4 * y5
            - y1**4 * y2 * y4 * sqrt3
            + y1**4 * y3 * y5
            - y2**4 * y3 * y5
        )
        * fea233335
        + (
            y2**2 * y3**4
            + y1**4 * y3**2
            + y1**2 * y2**4
            + y2**4 * y3**2
            + y1**2 * y3**4
            + y1**4 * y2**2
        )
        * fea222233
    )
    return v6


if __name__ == "__main__":
    s4 = 0.5
    s5 = 1
    sindelta = 1.2
    a1, a2, a3 = _find_alpha_from_sindelta(s4, s5, sindelta)
    print(a1, a2, a3)
