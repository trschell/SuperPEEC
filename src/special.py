# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Special functions used by the multipole expansion.

Provides the FMM-normalized spherical harmonics and their building blocks
(the associated Legendre function and the A_n^m normalization coefficient),
plus Gauss-Laguerre quadrature nodes/weights.

Split out of multipole.py; depends only on numpy.
"""
import math

import numpy as np


def a_nm(n, m):
    """Return the FMM normalization coefficient A_n^m.

        A_n^m = (-1)**n / sqrt((n - m)! * (n + m)!)

    This is the scaling coefficient that appears in the Greengard-Rokhlin
    multipole/local expansion and its translation operators (M2M/M2L/L2L).

    Parameters
    ----------
    n : int
        Degree of the expansion term (n >= 0).
    m : int
        Order, with -n <= m <= n.

    Returns
    -------
    float
        The coefficient A_n^m.
    """
    nmbigint = math.factorial(n - m) * math.factorial(n + m)
    return ((-1)**n)/np.sqrt(float(nmbigint))


def sph_harm_of_cos(n, m, theta, phi):
    """Evaluate the FMM-normalized spherical harmonic Y_n^m(theta, phi).

        Y_n^m = sign(m)**m * sqrt((n - m)! / (n + m)!)
                * P_n^m(cos theta) * exp(i * m * phi)

    where P_n^m is the associated Legendre function (:func:`legendreP`).
    This follows the FMM convention: the (2n + 1) / (4 pi) factor of the
    physics-normalized harmonic is omitted (consistent with :func:`a_nm`),
    and sign(m)**m supplies the Condon-Shortley-like phase for m < 0.

    Parameters
    ----------
    n : int
        Degree (n >= 0).
    m : int
        Order, with -n <= m <= n.
    theta : float or ndarray
        Polar angle(s); the Legendre part is evaluated at cos(theta).
    phi : float or ndarray
        Azimuthal angle(s), broadcast against `theta`.

    Returns
    -------
    complex or ndarray
        Y_n^m evaluated at (theta, phi).
    """
    factorial = math.factorial
    k = int(np.sign(m))**m * np.sqrt(factorial(n-m) / factorial(n+m))
    Ynm = k*legendreP(n, m, np.cos(theta)) * np.exp(1j*m*phi)
    return Ynm


def legendreP(l, m, x):
    """Associated Legendre function P_l^m(x) by direct coefficient expansion.

    The polynomial coefficients are formed explicitly (closed form for the
    leading coefficient, then a downward recurrence on the coefficients)
    rather than by the usual recurrence on the functions themselves. This is
    fast and vectorizes over `x`, but the coefficients grow quickly with
    degree, so accuracy degrades for large `l`; prefer a recurrence-based
    routine when high degrees are needed.

    Conventions
    -----------
    * ``abs(m) > abs(l)`` : the function is identically zero.
    * ``l < 0``          : mapped to ``-l - 1`` (since P_{-l-1} = P_l).
    * ``m < 0``          : obtained from the m > 0 result via the factor
      ``(-1)**m * (l - m)! / (l + m)!``.

    Parameters
    ----------
    l : int
        Degree.
    m : int
        Order. For backward compatibility, if `x` is None the value passed as
        `m` is treated as `x` and the order defaults to 0.
    x : float or ndarray
        Evaluation point(s), intended for |x| <= 1 (typically x = cos theta).

    Returns
    -------
    ndarray
        P_l^m(x), same shape as `x`.

    Notes
    -----
    The `err` / two-return-value behavior described in the original MATLAB
    port has been removed; only the polynomial value is returned.
    """
    factorial = math.factorial
    if x is None:
        x = m
        del(m)
        m = 0

    # Some basic error checking of parameters

    # Definition states that if |m|>l, the polynomial is 0; so set to 0 rather
    # than return an error. Some algorithms depend on this behaviour. Note that
    # the original definition requires 0<=m<=l
    if(abs(m) > abs(l)):
        # P_l^m is identically zero when |m| > l.
        return np.zeros(np.shape(x))

    # Definition for l<0
    if (l < 0):
        l = -l - 1

    # For m<0, polynomials are proportional to those with m>0, cfnm is the
    # proportionality coefficient
    cfnm = 1
    if (m < 0):
        m = -m
        cfnm = (-1)**m*factorial(l-m)/factorial(l+m)

    # Calculate coef of maximum degree in x from the explicit analytical
    # formula
    cl = (-1)**m*cfnm*factorial(2*l)/((2**l)*factorial(l)*factorial(l-m))
    maxcf = np.abs(cl)
    # fprintf('Coef is #.16f\n',cl);
    px = l - m

    # Power of x changes from one term to the next by 2. Also needed for
    # sqrt(1-x^2).
    x2 = x*x

    # Calculate efficiently P_l^m (x)/sqrt(1-x^2)^(m/2) - that is, only the
    # polynomial part. At least one coefficient is guaranteed to exist - there
    # is no null Legendre polynomial.
    p = cl*np.ones(np.shape(x))

    for j in range(l-1, -1, -1):
        # Check the exponent of x for current coefficient, px. If it is 0 or 1,
        # just exit the loop
        if (px < 2):
            break
        # If current exponent is >=2, there is a "next" coefficient; multiply p
        # by x2 and add it. Calculate the current coefficient
        cl = -(j+j+2-l-m)*(j+j+1-l-m)/(2*(j+j+1)*(l-j))*cl

        if (maxcf < np.abs(cl)):
            maxcf = np.abs(cl)
        # fprintf('Coef is #.16f\n',cl);
        # ...and add to the polynomial
        p = p*x2 + cl
        # Decrease the exponent of x - this is the exponent of x corresponding
        # to the newly added coefficient
        px = px - 2
    # Estimate the error
    # err = maxcf*np.spacing(1)
    # print("Coef is {0:1.7e}, err {1:1.7e}\n".format(maxcf, err))

    # Now we're done adding coefficients. However, if the exponent of x
    # corresponding to the last added coefficient is 1 (polynomial is odd),
    # multiply the polynomial by x
    if (px == 1):
        p = p*x

    # All that's left is to multiply the whole thing with sqrt(1-x^2)^(m/2). No
    # further calculations are needed if m=0.
    if (m == 0):
        return p

    x2 = 1 - x2
    # First, multiply by the integer part of m/2
    for j in range(1, int(np.floor(m/2)+1)):
        p = p*x2
    # If m is odd, there is an additional factor sqrt(1-x^2)
    if (m != 2*np.floor(m/2)):
        p = p*np.sqrt(x2)

    # Finally, the polynomials are not defined for |x|>1. If you do not need
    # this behaviour, comment the following line
    # p[np.abs(x) > 1] = np.nan

    return p


def laguerre_roots(n):
    """Return the n roots (nodes) of the degree-n Laguerre polynomial L_n.

    These are the abscissae for n-point Gauss-Laguerre quadrature on
    [0, inf) with weight exp(-x). Built by finding the roots of the
    coefficient vector [0, ..., 0, 1] (degree n).

    Parameters
    ----------
    n : int
        Number of nodes / polynomial degree.

    Returns
    -------
    ndarray
        The n real roots of L_n, in ascending order.
    """
    coef = np.append(np.zeros((n,)), 1)
    return np.polynomial.laguerre.lagroots(coef)


def laguerre_weights(n):
    """Return the n weights for n-point Gauss-Laguerre quadrature.

    Uses the standard formula at each node x_i (a root of L_n):

        w_i = x_i / ((n + 1)**2 * L_{n+1}(x_i)**2)

    Pairs with :func:`laguerre_roots` to approximate
    ``integral_0^inf exp(-x) f(x) dx ~= sum_i w_i * f(x_i)``.

    Parameters
    ----------
    n : int
        Number of quadrature nodes.

    Returns
    -------
    ndarray
        The n quadrature weights, ordered to match ``laguerre_roots(n)``.
    """
    roots = laguerre_roots(n)
    coef = np.append(np.zeros((n+1,)), 1)
    val = np.polynomial.laguerre.lagval(roots, coef)
    return roots/((n+1)**2*val**2)
