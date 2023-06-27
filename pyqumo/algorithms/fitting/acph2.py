"""
================================================
Fitting ACPH(2) matching the first three moments
================================================

Implementation of ACPH(2) moments matching fitting algorithm defined in
[TeHe03]_.

.. currentmodule:: pyqumo.fitting.acph2

Functions
=========

.. autosummary::
    :toctree: generated/

    fit_acph2
    get_acph2_m2_min
    get_acph2_m3_bounds


Overview
========

Definition
----------

Canonical acyclic PH-distribution with two states have the form:

.. math::
    S = \\left( \\begin{matrix}
            -\\lambda_1 & \\lambda_1 \\\\
            0 & -\\lambda_2
        \\end{matrix} \\right)
    \\qquad
    \\overline{\\tau} =
        \\left(\\begin{matrix} p & 1 - p \\end{matrix}\\right),

where :math:`S \\in \\mathbb{R}^{N \\times N}` is an infinitesimal generator
and :math:`\\tau \\in \\mathbb{R}^N` is a vector of initial probabilities.

Fitting algorithm
-----------------

These values are computed from the equations:

.. math::
    p = \\frac{-B + 6 m_1 D + \\sqrt{A}}{B + \\sqrt{A}}, \\\\
    \\lambda_1 = \\frac{B - \\sqrt{A}}{C}, \\\\
    \\lambda_2 = \\frac{B + \\sqrt{A}}{C},

where

.. math::
    A = B^2 - 6CD, \\\\
    B = 3 m_1 m_2 - m_3, \\\\
    C = 3 m_2^2 - 2 m_1 m_3 \\\\
    D = 2 m_1^2 - m_2.


Area of existence
-----------------

The main drawback of fitting random data using ACPH(2) is a very limited area
of ACPH(2) existence. The image below shows an area of existence on
:math:`(X,Y)` plane, where X-axis represents values of :math:`c_v - 1/c_v`,
:math:`c_v` is a coefficient of variation, and Y-axis represents skewness.

.. note::
    For exponential distribution :math:`c_v = 1` and :math:`\\gamma=2`,
    so it corresponds to :math:`(0, 2)` point on the plane below.

.. _acph-men2-area:

ACPH(2) and MEn(2) areas plot
-----------------------------

.. plot::

    >>> import matplotlib as mpl
    >>> from matplotlib import pyplot as plt
    >>> from pyqumo.stats import get_cv, get_noncentral_m2, get_noncentral_m3
    >>> from pyqumo.fitting.acph2 import get_acph2_m2_min, get_acph2_m3_bounds
    >>>
    >>> BG_CMAP = mpl.cm.get_cmap('Pastel1')
    >>> FG_CMAP = mpl.cm.get_cmap('tab20b')
    >>> get_color = lambda x: BG_CMAP(x)
    >>>
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> x_min, x_max, y_min, y_max, step = -2.0, 3, -0.5, 6, 0.04
    >>> x2cv = lambda x_: (x_ + pow(x_**2 + 4, 0.5)) / 2  # CV (c) from x=(c-1/c)
    >>> X = np.arange(x_min, x_max, step)  # cv - 1/cv
    >>> Y = np.arange(y_min, y_max, step)  # skew
    >>> CV = np.asarray([x2cv(x) for x in X])
    >>> F = np.zeros((len(Y), len(X)))
    >>> mean = 1  # assume mean value of all distributions is 1 (doesn't matter)
    >>>
    >>> m2_min = get_acph2_m2_min(mean)
    >>> cv_min = get_cv(mean, m2_min)
    >>>
    >>> Xs = np.asarray([x for x in X if (cv_min - 1/cv_min) <= x <= 0])
    >>> Cs = np.asarray([x2cv(x) for x in Xs])
    >>> G1 = (6*Cs**2 - 4 + 3 * 2**0.5 * (1 - Cs**2)**1.5) / Cs**3
    >>> G2 = 3/Cs - 1/Cs**3
    >>>
    >>> Xl = np.asarray([x for x in X if x >= 0])
    >>> Cl = np.asarray([x2cv(x) for x in Xl])
    >>> G3 = 1.5*Cl + 0.5/Cl**3
    >>>
    >>> Xv = [0, 0]
    >>> G4 = [2, y_max]  # constant: lowest point states for Exp dist.
    >>>
    >>> for i, x in enumerate(X):
    ...     cv = CV[i]
    ...     m2 = get_noncentral_m2(mean, cv)
    ...     if m2 < m2_min:
    ...         for j in range(len(Y)):
    ...             F[j, i] = np.nan
    ...         continue
    ...     min_m3, max_m3 = get_acph2_m3_bounds(mean, m2)
    ...     for j, y in enumerate(Y):
    ...         m3 = get_noncentral_m3(mean, cv, y)
    ...         F[j, i] = 1 if min_m3 <= m3 <= max_m3 else np.nan
    >>>
    >>> ax.pcolormesh(X, Y, F, cmap=BG_CMAP)
    >>> ax.plot(Xs, G1, color=FG_CMAP(.4))
    >>> ax.plot(Xs, G2, color=FG_CMAP(.4))
    >>> ax.plot(Xl, G3, color=FG_CMAP(.4))
    >>> ax.plot(Xv, G4, color=FG_CMAP(.4))
    >>>
    >>> boundary = [max(X[0], Y[0]), min(X[-1], Y[-1])]
    >>> ax.plot(boundary, boundary, color=get_color(0.5))
    >>> ax.fill(
    ...     [x_min, min(boundary), max(boundary), x_max, x_min],
    ...     [y_min, min(boundary), max(boundary), y_max, y_max],
    ...     color=BG_CMAP(0.2), alpha=0.5)
    >>>
    >>> scale = 0.7
    >>> ax.text(-2.5 * scale, 0.9,
    ...         r'$\\gamma = \\frac{6c^2 - 4 + 3\\sqrt{2}(1-c^2)^{3/2}}{c^3}$',
    ...         fontsize=23, rotation=0)
    >>> ax.text(-1.7 * scale, 1.7, r'$\\gamma = \\frac{3}{c} - \\frac{1}{c^3}$',
    ...         fontsize=22, rotation=37)
    >>> ax.text(-0.25, 4.0, rotation=90, fontsize=24, s=r'$c = 0$')
    >>> ax.text(1.0, 2.1,  r'$\\gamma = \\frac{1}{2}(3c + \\frac{1}{c^3})$',
    ...         fontsize=22, rotation=37,)
    >>> ax.text(1.5, 1.2, r'$\\gamma = c - \\frac{1}{c}$',
    ...         fontsize=22, rotation=37)
    >>> ax.text(x_min + 0.1, y_min + 0.1, 'Area of\\nPH existence',
    ...         fontsize=18, color="black")
    >>> ax.text(x_max - 2.0 * scale, y_min + 0.1, 'No PH exist',
    ...         fontsize=18, color="black")
    >>> ax.text(x_min + 0.3, y_max - 1.0, r'$ME_n(2)$', fontsize=26,
    ...         color="black")
    >>> ax.text(x_max - 2.5 * scale, y_max - 1.0, r'$ACPH(2)$', fontsize=26,
    ...         color="black")
    >>>
    >>> ax.grid()
    >>> ax.set_xlim((x_min, x_max))
    >>> ax.set_ylim((y_min, y_max))
    >>> ax.set_xlabel(r'$c - 1/c$', fontsize=18)
    >>> ax.set_ylabel(r'Skewness ($\\gamma)$', fontsize=18)
    >>> ax.tick_params(labelsize=16)



References
==========

.. [TeHe03] Telek, Miklós & Heindl, Armin. (2003). Matching Moments For Acyclic
            Discrete And Continuous Phase-Type Distributions Of Second Order.
            International Journal of Simulation Systems, Science & Technology.
            3.
"""
from typing import Sequence, Tuple
import numpy as np
# \begin
# {equation}
# S = \left(\begin
# {matrix}
# -\lambda_1 & \lambda_1 \ \
#     0 & -\lambda_2
# \end
# {matrix}\right),
# \qquad
# \overline
# {\tau} = \left(\begin
# {matrix}
# p_1 & 1 - p_1\end
# {matrix}\right).
# \end
# {equation}

from pyqumo.errors import BoundsError
from pyqumo.randoms import PhaseType


def fit_acph2(
    moments: Sequence[float],
    strict: bool = True
) -> Tuple[PhaseType, np.ndarray]:
    """
    Fit ACPH(2) distribution by :math:`m_1, m_2, m_3`

    Algorithm and moments values bounds are provided in [TeHe03]_. See
    Table 1 inside the paper for boundaries on :math:`m_2` and :math:`m_3`,
    Figure 1 for graphical display of the boundaries and Table 3 for formulas
    to compute PH distribution parameters from valid :math:`m_1, m_2, m_3`.

    If algorithm fails to fit due to values :math:`m_1, m_2, m_3` laying out of
    bounds, raise BoundsError exception when `strict = True`. Otherwise, try to
    find the closest ACPH. In the latter case, select moments with these rules:

    1. If :math:`c_v^2 = (m_1^2 / m_2 - 1) <= 0.5`, then :math:`m_2` and
       :math:`m_3` are set equal to :math:`m_2 = 1.5 * m_1^2` and
       :math:`m_3 = 3 * m_1^3` respectively as for Erlang-2.

    2. If :math:`0.5 < c_v^2 < 1.0`, then :math:`m_2` is OK.
       However, if `m3` is out of bounds (region BII or BIII, see Figure 2
       in [TeHe03]_), then :math:`m_3` is selected to be
       :math:`m_3 = 0.5 * (\\underline{m_3} + \\overline{m_3})`, where
       :math:`\\underline{m_3}` and :math:`\\overline{m_3}` are boundaries of
       BII and BIII, see Table 1 [TeHe03]_ and
       ``get_acph2_m3_bounds()``.

    3. If :math:`c_v == 1`, then :math:`m_3` is set as for exponential
       distribution: :math:`m_3 = 6 * m_1^3`

    4. If `c_v > 1`, then `m_3` is set as :math:`10/9 * \\underline{m_3}`,
       where :math:`\\underline{m_3}` value is defined as the boundary of BI
       (see Figure 2 in [TeHe03]_).

    The same rules for selecting moments :math:`m_2` and :math:`m_3` apply
    if less than three moments were provided and `strict = False`.

    If more than three moments are provided to the algorithm, 4-th and higher
    order moments are not used in estimation. However, the algorithm computes
    relative errors for these moments from the fitted ACPH(2).

    If any moment is less or equal to zero, or if computed :math:`c_v^2 <= 0`,
    the ``ValueError`` exception is raised.

    Parameters
    ----------
    moments : sequence of floats
        First moments. If ``strict = True``, then this sequence MUST contain at
        least three moments. If ``strict = False``, then missing moments will
        be selected by the algorithm.
    strict : bool, optional
        Flag indicating whether raise an error if ACPH(2) can not be found
        due to moment values laying out of bounds.

    Raises
    ------
    ValueError
        raise this when moments are less or equal to 0, or if pow(CV, 2) <= 0
    BoundsError
        raise if ``strict = True`` and moments values

    Returns
    -------
    ph : :py:class:`pyqumo.random.PhaseType`
        ACPH(2) distribution
    errors : np.ndarray
        tuple containing relative errors for moments of the distribution found.
        The number of errors is equal to the number of moments passed: if
        more than three moments were given, errors will be estimated for all
        of them. If strict = False and one or two moments were passed,
        then errors will be computed only for these one or two moments.
    """
    # First of all, check either three moments are provided, or strict = False:
    # - if strict = True and len(moments) < 3, raise ValueError
    # - if strict = False, however, try to find some m2 and m3 values
    #   to use in estimation.
    #
    # If CV2 falls into region between (BII, BIII), use M3 value in the medium.
    # If CV2 > 1, use M3 from a line that is as 10/9 m3 lower bound.
    #
    # If in non-strict mode only M1 is provided, treat as exponential
    # distribution and set M2 and M3 accordingly.
    if (n := len(moments)) < 3:
        if strict:
            raise ValueError(f"Expected three moments, but {n} found")
        if n == 0:
            raise ValueError(f"At least one moment needed, but 0 found")

        # If not strict and at least one moment provided, try to find m2, m3.
        m1 = moments[0]
        m2 = 2 * m1**2 if n < 2 else moments[1]

        # M3 selection depends on pow(CV, 2)
        cv2 = m2 / m1**2 - 1
        if cv2 <= 0.5:  # tread as Erlang: M3 = k(k+1)(k+2) / pow(k/m1, 3), k=2
            m3 = 3 * m1**3
        elif 0.5 < cv2 <= 1.0:
            m3 = sum(get_acph2_m3_bounds(m1, m2)) / 2
        else:
            m3 = 5/3 * m1**3 * (1 + cv2)**2  # to step from boundary
    else:
        m1 = moments[0]
        m2 = moments[1]
        m3 = moments[2]

    # Validate mandatory moments relations:
    # - each moment must be positive real value
    # - pow(CV, 2) must be positive
    for i, m in enumerate(moments):
        if m <= 0:
            raise ValueError(f"Expected m{i+1} > 0, but m{i+1} = {m}")
    cv2 = m2 / m1**2 - 1
    if cv2 <= 0:
        raise ValueError(f"Expected pow(CV, 2) > 0, but pow(CV, 2) = {cv2}")

    # If strict = True, validate moments and raise error is out of bounds:
    if strict:
        if (m2_min := get_acph2_m2_min(m1)) > m2:
            raise BoundsError(
                f"m2 = {m2} is out of bounds for m1 = {m1}\n"
                f"\tpow(CV, 2) = {cv2}\n"
                f"\tmin. pow(CV, 2) = 0.5\n"
                f"\tmin. M2 = {m2_min}"
            )
        m3_min, m3_max = get_acph2_m3_bounds(m1, m2)
        if not (m3_min <= m3 <= m3_max):
            raise BoundsError(
                f"m3 = {m3} is out of bounds for m1 = {m1}, m2 = {m2}\n"
                f"\tpow(CV, 2) = {cv2}\n"
                f"\tmin. M3 = {m3_min}\n"
                f"\tmax. M3 = {m3_max}"
            )

    # If strict = False, tune moments to put them into the valid bounds:
    if not strict:
        # If pow(CV, 2) < 0.5, then set M2 and M3 as for Erlang-2 distribution:
        if cv2 < 0.5:
            m2 = 1.5 * m1**2
            m3 = 3 * m1**3
        elif cv2 < 1.0:
            m3_min, m3_max = get_acph2_m3_bounds(m1, m2)
            if not (m3_min <= m3 <= m3_max):
                m3 = 0.5 * (m3_min + m3_max)
        elif cv2 == 1.0:
            m3 = 6 * m1**3
        elif cv2 > 1.0:
            m3_min = get_acph2_m3_bounds(m1, m2)[0]
            m3 = m3_min * 10/9

    # Define auxiliary variables
    d = 2 * m1**2 - m2
    c = 3 * m2**2 - 2 * m1 * m3
    b = 3 * m1 * m2 - m3
    a = (b**2 - 6 * c * d) ** 0.5  # in paper no **0.5, but this is useful

    # Define subgenerator and probabilities vector elements
    if c > 0:
        p = (-b + 6 * m1 * d + a) / (b + a)
        l1 = (b - a) / c
        l2 = (b + a) / c
    elif c < 0:
        p = (b - 6 * m1 * d + a) / (-b + a)
        l1 = (b + a) / c
        l2 = (b - a) / c
    else:
        p = 0
        l1 = 1 / m1
        l2 = 1 / m1

    # Build the distribution and compute estimation errors:
    ph = PhaseType(
        sub=np.asarray([[-l1, l1], [0.0, -l2]]),
        p=np.asarray([p, 1 - p]))
    errors = [abs(m - ph.moment(i+1)) / m for i, m in enumerate(moments)]
    return ph, np.asarray(errors)


def get_acph2_m2_min(m1: float) -> float:
    """
    Find minimum :math:`m_2` for the given mean value for any valid ACPH(2).

    According to [TeHe03]_, :math:`m_2` has only lower bound since
    :math:`c_v^2` should be greater or equal to 0.5.

    If :math:`m_1 < 0`, then ``ValueError`` is raised.

    Parameters
    ----------
    m1 : float

    Returns
    -------
    m2_min : float
        Minimum eligible value of the second moment.
    """
    if m1 < 0:
        raise ValueError(f"Expected m1 > 0, but m1 = {m1}")
    return 1.5 * m1**2


def get_acph2_m3_bounds(m1: float, m2: float) -> Tuple[float, float]:
    """
    Find min and max :math:`m_3` for the given :math:`m_1, m_2` for ACPH(2).

    Bounds are specified in Table 1 and Figure 2 in [TeHe03]_.
    When :math:`c_v > 1`, only lowest bound exist. If  :math:`0.5 < c_v^2 < 1`,
    then both lower and upper bounds are defined, and they are very tight.
    When :math:`c_v = 1`, :math:`m_3` is fixed for exponential distribution
    (singular point), so both bounds are equal.

    If arguments are such that :math:`c_v^2 < 0.5`
    (i.e. :math:`m_2 < 1.5 * m_1^2`), then raise ``ValueError``.

    Parameters
    ----------
    m1 : float
    m2 : float

    Returns
    -------
    lower : float
        Lower bound
    upper : float
        Upper bound
    """
    if m1 <= 0:
        raise ValueError(f"Expected m1 > 0, but m1 = {m1}")
    if m2 <= 0:
        raise ValueError(f"Expected m2 > 0, but m2 = {m2}")

    # Find square of coefficient of variation (CV**2):
    cv2 = m2 / m1**2 - 1

    # If CV > 1, then only lower bound exists:
    if cv2 > 1:
        return 3/2 * m1**3 * (1 + cv2)**2, np.inf

    # If CV**2 >= 0.5, but <= 1, both bounds exist:
    if 0.5 <= cv2 <= 1:
        return (
            3 * m1**3 * (3 * cv2 - 1 + 2**0.5 * (1 - cv2)**1.5),
            6 * m1**3 * cv2
        )

    # If CV**2 < 0.5, M3 is undefined:
    raise ValueError(
        f"Expected CV >= sqrt(0.5), but CV = {cv2**0.5} "
        "(CV = coef. of variation)")
