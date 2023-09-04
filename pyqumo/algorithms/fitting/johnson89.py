"""
=====================================
Fitting MEn(2) matching three moments
=====================================

Module: :py:mod:`pyqumo.fitting.johnson89`

Implementation of the PH moments matching fitting algorithm using a mixture of
two Erlang distributions with common order N using three moments.
This method was proposed by Johnson & Taaffe in paper [JoTa89]_.

Functions
=========

.. autosummary::
    :toctree: generated/

    fit_mern2
    get_mern2_props


Overview
========

Method proposed by Johnson & Taaffe in [JoTa89]_ allows to find :math:`PH`-
distribution in the form of a mixture of two Erlang distributions with the
same shape :math:`n` and different rates (:math:`ME_n(2)`):

.. graphviz::

   digraph foo {
        rankdir=LR;
        node [shape=point label=""]; 0;
        node [shape=circle label="A1"]; 1 3;
        node [shape=circle label="A2"]; 4 6;
        node [shape=plaintext label="..."] 2 5;
        node [shape=doublecircle label="" fixedsize=true width=.2 height=.2];42;
      0 -> 1 [label="p"];
      1 -> 2;
      2 -> 3;
      3 -> 42;
      0 -> 4 [label="1-p"];
      4 -> 5;
      5 -> 6;
      6 -> 42;
   }

Fitting algorithm
-----------------

Firstly, we need to estimate the minimal order of Erlang chains :math:`n^*` as

.. math::
    n^* = \\lceil \\max\\{
        \\frac{1}{c_v^2},
        \\frac{-\\gamma + 1/c_v^3 + 1/c_v + 2c_v}{\\gamma - (c_v - 1/c_v)}
    \\} \\rceil

Any order :math:`n \\geqslant n^*` is eligible, so choose one. Later we may
need to increase :math:`n` if :math:`\\lambda_1 / \\lambda_2` is too large or
too small, or if :math:`p` or :math:`1-p` is too close to zero.
See [JoTa89]_ for details.

Then, we find :math:`\\lambda_i` and :math:`p`:

.. math::
    \\lambda_{1,2}^{-1} &= \\frac{-B \\pm \\sqrt{B^2 - 4AC}}{2A} \\\\
    p &= \\frac{m_1/n - \\lambda_2^{-1}}{\\lambda_1^{-1} - \\lambda_2^{-1}}

where

.. math::

    A &= n (n+2) m_1 y \\\\
    B &= -(nx + \\frac{n(n+2)}{n+1}y^2 + (n+2)m_1^2y) \\\\
    C &= m_1 x \\\\
    x &= m_1 m_3 - \\frac{n+2}{n+1}m_2^2 \\\\
    y &= m_2 - \\frac{n+1}{n}m_1^2


Area of MEn(2) existence
------------------------

The method allows to find a distribution for any feasible :math:`m_1, m_2, m_3`.
Consider values :math:`m_1, m_2, m_3`, such that there exists a :math:`PH`-
distribution :math:`\\xi` with :math:`E[\\xi^k] = m_k`,
:math:`k = 1,2,3`. Then, this method will find a :math:`ME_n(2)`
distribution :math:`\\xi'`, such that :math:`E[\\xi'^k] = m_k`,
:math:`k = 1,2,3`.

The feasibility area can be easily defined on :math:`(c_v - 1/c_v, \\gamma)`
plane as an area :math:`c_v - 1/c_v < \\gamma`, where

.. math::
    \\gamma &= E[(\\frac{X - m_1}{\\sigma})^3] \\\\
    \\sigma &= \\sqrt{m_2 - m_1^2}

(see :ref:`acph-men2-area` for illustration)

However, the number of states in the Erlang chains may be huge, especially
when :math:`c_v < 1`. See [JoTa89]_ for details. The figure below illustrates
growth of :math:`n`, :math:`\\max\\{p, 1 - p\\}` and
:math:`\\max\\{\\lambda_1/\\lambda_2, \\lambda_2/\\lambda_1\\}`.

.. plot::

    >>> from itertools import product
    >>> import matplotlib as mpl
    >>> from matplotlib import pyplot as plt
    >>> from pyqumo.fitting.johnson89 import fit_mern2
    >>> from pyqumo.stats import get_noncentral_m2, get_noncentral_m3
    >>>
    >>> x2cv = lambda x: (x + pow(x**2 + 4, 0.5)) / 2
    >>>
    >>> GRID_SIZE = 200
    >>> X = np.linspace(-10, 10, GRID_SIZE)  # c - 1/c values
    >>> Y = np.linspace(-10, 10, GRID_SIZE)  # skewness values, not all feasible
    >>> CV = np.asarray([x2cv(x) for x in X])
    >>> PLANE_SHAPE = (GRID_SIZE, GRID_SIZE)
    >>> COORDS = list(product(range(GRID_SIZE), range(GRID_SIZE)))
    >>> MEAN = 1.0
    >>>
    >>> distributions = [[
    ...     fit_mern2([MEAN, get_noncentral_m2(MEAN, cv),
    ...                get_noncentral_m3(MEAN, cv, y)])[0]
    ...     if y > x else None for (x, cv) in zip(X, CV)] for y in Y]
    >>>
    >>> # Define three square matrices, with (i,j)-th element corresponding to:
    >>> # 1. orders: order of Erlang chain in (i,j)-th MEn(2) distribution
    >>> # 2. min_p: minimal probability of Erlang chain in (i,j)-th distribution
    >>> # 3. max_ratio: max(A1/A2, A2/A1), where Ak - k-th Erlang's param
    >>> orders = np.empty(shape=PLANE_SHAPE)
    >>> min_p = np.empty(shape=PLANE_SHAPE)
    >>> max_ratio = np.empty(shape=PLANE_SHAPE)
    >>> orders[:] = np.nan
    >>> min_p[:] = np.nan
    >>> max_ratio[:] = np.nan
    >>>
    >>> for (i, j) in COORDS:
    ...     if (dist := distributions[i][j]) is not None:
    ...         orders[i][j] = min(dist.order/2, 100)
    ...         min_p[i][j] = min(min([p for p in dist.probs if p > 0]), 0.05)
    ...         max_ratio[i][j] = min(max(dist.params) / min(dist.params), 50)
    >>>
    >>> plt.rcParams.update({
    ...     'figure.titlesize': 20,
    ...     'axes.titlesize': 18,
    ...     'axes.labelsize': 16,
    ... })
    >>> CMAP = plt.cm.get_cmap('viridis')
    >>> fig, axes = plt.subplots(figsize=(13, 5), ncols=3, nrows=1, sharey=True)
    >>> im1 = axes[0].pcolormesh(X, Y, orders, cmap=CMAP)
    >>> im2 = axes[1].pcolormesh(X, Y, max_ratio, cmap=CMAP)
    >>> im3 = axes[2].pcolormesh(X, Y, min_p, cmap=CMAP.reversed())
    >>> fig.suptitle("MEn(2) properties without optimization")
    >>> axes[0].set_title('Order of Erlang\\n(trunc. at 100)')
    >>> axes[1].set_title(
    >>>     r"$\\max\\{\\lambda_1/\\lambda_2, \\lambda_2/\\lambda_1\\}$" +
    >>>     '\\n(trunc. at 50)')
    >>> axes[2].set_title(r'$\\min\\{p, 1-p\\}$' + '\\n(trunc. at 0.05)')
    >>> axes[0].set_ylabel(r"Skewness ($\\gamma$)")
    >>> for im, ax in zip((im1, im2, im3), axes):
    ...     fig.colorbar(im, ax=ax)
    ...     ax.set_xlabel('c - 1/c')
    ...     ax.grid()
    >>> plt.tight_layout()

Generally, "better" distributions are in dark areas on the plots.

References
==========

.. [JoTa89] Mary A. Johnson & Michael R. Taaffe (1989) Matching moments to
            phase distributions: Mixtures of erlang distributions of common
            order, Communications in Statistics. Stochastic Models, 5:4,
            711-743, https://doi.org/10.1080/15326348908807131
"""
from typing import Sequence, Tuple
import numpy as np

from pyqumo.randoms import HyperErlang
from pyqumo.stats import get_cv, get_skewness, get_noncentral_m3
from pyqumo.errors import BoundsError


def fit_mern2(
    moments: Sequence[float],
    strict: bool = True,
    max_shape_inc: int = 0
) -> Tuple[HyperErlang, np.ndarray]:
    """
    Fit moments with a mixture of two Erlang distributions with common order.

    The algorithm is defined in [JoTa89]_.

    In strict mode (``strict = True``) requires at least three moments to fit.
    Note, that if more moments provided, function will compute errors in
    their estimation, while not taking them into account. If the first three
    moments do not fit into feasible area, function will raise ``BoundsError``
    exception.

    In non-strict mode (``strict=False``) user can provide two or even one
    moment, and M3 may go beyond the feasible area. In this case,
    M3 will be selected using the rule:

    - if :math:`c_v > 1`, then skewness will be equal to
      :math:`6/5 (c_v - 1/c_v)`;
    - if :math:`c_v = 1`, then skewness will be equal 2
      (exponential distribution);
    - if :math:`0 < c_v < 1`, then skewness will be equal to
      :math:`4/5 (c_v - 1/c_v)`.

    User can provide `max_shape_inc` parameter. If so, the algorithm will
    try to improve the ratio between Erlang distribution parameters and
    minimum probability as described in section 7.3 of [JoTa89]_ by increasing
    the Erlang distributions shape (up to ``max_shape_inc``).

    Parameters
    ----------
    moments : sequence of float
        Only the first three moments are taken into account
    strict : bool, optional (default: ``True``)
        If ``True``, require at least three moments explicitly defined,
        and do not perform any attempt to adjust moments if they are out of
        bounds. Otherwise, try to fit even for bad or incomplete data.
    max_shape_inc : int, optional (default: 0)
        if non-zero, maximum increase in shape when attempting to build
        a more stable distribution (refer to 7.3 "Numerical Stability" section
        of [JoTa89]_ for details)

    Raises
    ------
    BoundsError
        raise this if moments provided are out of bounds (in strict mode),
        or can not be recovered (non-strict mode).
    ValueError
        raise this in strict mode if number of moments provided is less than
        three, or if no moments provided at all (also in non-strict mode).

    Returns
    -------
    dist : :py:class:`pyqumo.random.HyperErlang`
        an instance of HyperErlang distribution fitted
    errors : tuple of float
        errors computed for each moment provided
    """
    if (num_moments := len(moments)) == 3:
        m1, m2, m3 = moments[:3]
        cv = get_cv(m1, m2)
    else:
        if (strict and num_moments < 3) or num_moments == 0:
            raise ValueError(f"Expected 3 moments, but {num_moments} found")
        m1 = moments[0]
        m2 = moments[1] if num_moments > 1 else 2*pow(m1, 2)
        cv = get_cv(m1, m2)
        if cv < 1 - 1e-5:
            m3 = get_noncentral_m3(m1, cv, (cv - 1/cv) * 0.8)
        elif abs(cv - 1) <= 1e-4:
            m3 = moments[2] if num_moments > 2 else 6*pow(m1, 3)
        else:
            m3 = get_noncentral_m3(m1, cv, (cv - 1/cv) * 1.2)

    gamma = get_skewness(m1, m2, m3)

    # Check boundaries and raise BoundsError if fail:
    if (min_skew := cv - 1/cv) >= gamma:
        if strict:
            raise BoundsError(
                f"Skewness = {gamma:g} is too small for CV = {cv:g}\n"
                f"\tmin. skewness = {min_skew:g}\n"
                f"\tm1 = {m1:g}, m2 = {m2:g}, m3 = {m3:g}")
        else:
            if cv < 1 - 1e-5:
                m3 = get_noncentral_m3(m1, cv, (cv - 1/cv) * 0.8)
            elif abs(cv - 1) <= 1e-4:
                m3 = 6 * pow(m1, 3)
            else:
                m3 = get_noncentral_m3(m1, cv, (cv - 1/cv) * 1.2)
            # print('previous gamma: ', gamma)
            gamma = get_skewness(m1, m2, m3)
            # print('new gamma: ', gamma)

    # Compute minimal shape for Erlang distributions:
    shape = int(max(
        np.ceil(1 / cv**2),
        np.ceil((-gamma + 1/cv**3 + 1/cv + 2*cv) / (gamma - (cv - 1/cv)))
    )) + (2 if cv <= 1 else 0)

    # If allowed to use higher order of Erlang to make results more stable,
    # try to optimize shape. Otherwise, just get the parameters for
    # the shape found above:
    shape, l1, l2, p = _optimize_stability(m1, m2, m3, shape, max_shape_inc)

    # Build hyper-Erlang distribution:
    dist = HyperErlang([l1, l2], [shape, shape], [p, 1 - p])

    # Estimate errors:
    errors = np.asarray([
        abs(m - dist.moment(i+1)) / abs(m) for i, m in enumerate(moments)
    ])

    return dist, errors


def get_mern2_props(
        m1: float,
        m2: float,
        m3: float,
        n: int
) -> Tuple[float, float, float]:
    """
    Helper function to estimate Erlang distributions rates and
    probabilities from the given moments and Erlang shape (:math:`n`).

    See theorem 3 in [JoTa89]_ for details about :math:`A`, :math:`B`,
    :math:`C`, :math:`p`, :math:`x`, :math:`y` and :math:`\\lambda_{1,2}`
    computation.

    Parameters
    ----------
    m1 : float
        mean value
    m2 : float
        second non-central moment
    m3 : float
        third non-central moment
    n : int
        shape of the Erlang distributions

    Returns
    -------
    l1 : float
        parameter of the first Erlang distribution
    l2 : float
        parameter of the second Erlang distribution
    n : int
        shape of the Erlang distributions

    Raises
    ------
    BoundsError
        raise this if skewness is below CV - 1/CV (CV - coef. of variation)
    """
    # Check boundaries:
    cv = get_cv(m1, m2)
    gamma = get_skewness(m1, m2, m3)
    if (min_skew := cv - 1/cv) >= gamma:
        raise BoundsError(
            f"Skewness = {gamma:g} is too small for CV = {cv:g}\n"
            f"\tmin. skewness = {min_skew:g}\n"
            f"\tm1 = {m1:g}, m2 = {m2:g}, m3 = {m3:g}")

    # Compute auxiliary variables:
    x = m1 * m3 - (n + 2) / (n + 1) * pow(m2, 2)
    y = m2 - (n + 1) / n * pow(m1, 2)
    c = m1 * x
    b = -(
        n * x +
        n * (n + 2) / (n + 1) * pow(y, 2) +
        (n + 2) * pow(m1, 2) * y
    )
    a = n * (n + 2) * m1 * y
    d = pow(b**2 - 4 * a * c, 0.5)

    # Compute Erlang mixture parameters:
    em1, em2 = (-b - d) / (2*a), (-b + d) / (2*a)
    p1 = (m1 / n - em2) / (em1 - em2)
    l1, l2 = 1 / em1, 1 / em2
    return l1, l2, p1


def _optimize_stability(
        m1: float,
        m2: float,
        m3: float,
        shape_base: int,
        max_shape_inc: int) -> Tuple[int, float, float, float]:
    """
    Optimize stability of the resulting hyper-Erlang distribution.

    Try to slightly increase shape of Erlang distributions to make
    ratio between Erlang parameters (`r = l2 / l1` if `l2 > l1`) less,
    as well as to increase the minimum probability.

    Parameters
    ----------
    m1: float
    m2: float
    m3: float
    shape_base: int
    max_shape_inc: int

    Returns
    -------
    shape: int
    l1: float
    l2: float
    p: float
    """
    shape = shape_base
    l1, l2, p = get_mern2_props(m1, m2, m3, shape)

    def get_ratio(l1_, l2_):
        """
        Helper to get ratio between Erlang rates. Always return value >= 1.
        """
        if l2_ >= l1_ > 0:
            return l2_ / l1_
        return l1_ / l2_ if l1_ > l2_ > 0 else np.inf

    def get_min_prob(p_):
        return p_ if p_ < 0.5 else 1 - p_

    r_max_prev = get_ratio(l1, l2)
    p_min_prev = get_min_prob(p)
    inc = 1
    while inc < max_shape_inc:
        shape = shape_base + inc
        l1_new, l2_new, p_new = get_mern2_props(m1, m2, m3, shape)
        r_max_curr = get_ratio(l1_new, l2_new)
        p_min_curr = get_min_prob(p_new)
        # If shape increase doesn't provide sufficient improvement,
        # stop iteration and abondon changes:
        if r_max_prev / r_max_curr < 1.1 and p_min_curr / p_min_prev < 1.1:
            shape = shape - 1
            break
        # Otherwise remember current values of L1, L2, P and go to the
        # next iteration:
        l1, l2, p = l1_new, l2_new, p_new
        inc += 1
    return shape, l1, l2, p
