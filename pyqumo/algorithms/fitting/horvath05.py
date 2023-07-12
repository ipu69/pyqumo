"""
====================================================================
Moments matching MAP fitting with independent approximation of lag-k
====================================================================

.. currentmodule:: pyqumo.fitting.horvath05

Algorithm for moments matching MAP fitting based on [HoBuTe05]_.

Functions
=========

.. autosummary::
    :toctree: generated/

    fit_map_horvath05
    optimize_lag1

Overview
=========

According to this approach, we first fit stationary :math:`PH`-distribution of
the :math:`MAP`-process, and then use optimization techniques to find
:math:`D_1` matrix, such that it approximates the given lag-1 autocorrelation.

:math:`MAP(D_0, D_1)` has stationary PH distribution with generator :math:`D_0`
and initial probability distribution :math:`p` that is equal to stationary
distribution of the embedded DTMC with matrix :math:`T=(-D_0^{-1})D_1`.
:math:`MAP` moments are completely defined by this :math:`PH`, so matrix
:math:`D_1` itself defines only lag-k autocorrelation. Thus we can fit
:math:`PH` distribution using any kind of EM or moments matching technique,
and then find :math:`D_1` matrix using the given autocorrelation.
For details, see [HoBuTe05]_.

In this implementation we use `scipy.optimize.minimize` to look for the
:math:`D_1` matrix and find lag-1 boundaries.

References
==========

.. [HoBuTe05] Horvath, G., Buchholz, P., & Telek, M. (2005). A MAP fitting
              approach with independent approximation of the inter-arrival time
              distribution and the lag correlation.
              Second International Conference on the Quantitative Evaluation of
              Systems (QEST’05), 124–133. https://doi.org/10.1109/QEST.2005.1
"""
import warnings
from typing import Optional, Tuple

import scipy.optimize

from pyqumo.matrix import cbdiag
from pyqumo.randoms import PhaseType, MarkovArrival
import numpy as np


def fit_map_horvath05(
        ph: PhaseType,
        lag1: float = 0.,
        n_iters: int = 3,
        tol: float = .01) -> Tuple[MarkovArrival, np.ndarray]:
    """
    Fit Markov arrival process using approach from [HoBuTe05]_.

    Internally, calls :py:func:`optimize_lag1` function to find :math:`D_1`
    matrix that defines a :math:`MAP(D_0, D_1)`, where :math:`D_0 = PH.S`
    (generator of the :math:`PH`-distribution).

    See :py:class:`pyqumo.random.PhaseType` for
    details.

    Parameters
    ----------
    ph : :py:class:`pyqumo.random.PhaseType`
        stationary distribution of the
    lag1 : float
        log-1 autocorrelation (optional, default: 0.0)
    n_iters : int,
        optional (default: 3)
    tol: float
        this is the tolerance that is used to validate MAP D0 and D1
        (optional, default: 0.01)

    Returns
    -------
    arrival: :py:class:`pyqumo.arrivals.MarkovArrival`
        fitted :math:`MAP`
    errors: ndarray
        relative fitting error for lag-1, lag-2, etc. (right now, only lag-1)
    """
    m = ph.order
    opt_ret = optimize_lag1(ph, optype='opt', lag1=lag1, n_iters=n_iters)
    d1 = opt_ret.x.reshape((m, m)).T
    markov_arrival = MarkovArrival(ph.s, d1, tol=tol)
    lags = [lag1]
    abs_lags = [abs(lag) for lag in lags]
    errors = [
        abs(markov_arrival.lag(i+1) - lag) / abs_lag
        if abs_lag > 1e-3
        else abs(markov_arrival.lag(i+1) - lag)
        for i, (lag, abs_lag) in enumerate(zip(lags, abs_lags))
    ]
    return markov_arrival, np.asarray(errors)


# noinspection PyPep8Naming
def optimize_lag1(
        ph: PhaseType,
        optype: str = 'opt',
        lag1: Optional[float] = None,
        n_iters: int = 3) -> scipy.optimize.OptimizeResult:
    """
    Solve optimization problem of either fitting :math:`D_1` matrix, or search
    for the minimum or maximum possible lag-1 autocorrelation for a given
    :math:`PH`.

    :math:`PH`-distribution has generator :math:`S=D_0`, its initial probability
    distribution is equal to stationary distribution of the embedded DTMC
    stationary distribution.

    This function can solve three tasks:

    1. Find :math:`D_1` such that its lag-1 autocorrelation is close to the
       given value (``optype = "opt"``).
    2. Find minimum possible lag-1 autocorrelation value (``optype = "min"``).
    3. Find maximum possible lag-1 autocorrelation value (``optype = "max"``).

    In the second and the third cases the return value will hold the boundary
    value in the field ``fun``. In the first case `fun` field will hold
    a square error of the approximated lag-1.

    Algorithm will make several attempts to find a solution using different
    initial :math:`D_1` values. For the first time it will start from a matrix
    :math:`D_1` that appears in MAP representation of the given :math:`PH`
    (i.e., when lag-1 is zero). Number of other attempts is given in ``n_iters``
    argument. Initial points will be selected as possible solutions of simplex
    method when solving a dummy linear problem on the same linear feasible
    region.

    FIXME: in fact, errors are very high. Consider other optimization methods.

    Parameters
    ----------
    ph : :py:class:`pyqumo.random.PhaseType``
        stationary phase-type distribution of MAP
    optype : ``"opt"``, ``"min"`` or ``"max"``
        type of optimization problem
    lag1 : float
        used when ``optype = "opt"``, lag-1 autocorrelation to approximate
        (optional, default: 0)
    n_iters : int
        number of attempts to solve the optimization problem with random
        initial vectors (optional, default: 3)

    Returns
    -------
    results : scipy.optimize.OptimizeResult
        result is returned by ``scipy.optimize.minimize()`` function.
    """
    # Extract props:
    m = ph.order
    rate = 1 / ph.mean

    # Build constraints matrix and right side (without lag here!)
    A_eq = np.vstack([
        np.hstack([np.eye(m) for _ in range(m)]),
        cbdiag(m, [(0, ph.init_probs.reshape((1, m)) @ ph.sni)])
    ])
    b_eq = np.vstack([
        -ph.s @ np.ones((m, 1)),
        ph.init_probs.reshape((m, 1))
    ])

    # Build helper matrices for lag-1 autocorrelation computation.
    delta = (rate ** 2) * ph.init_probs.reshape((1, m)) @ ph.sni @ ph.sni
    f = ph.sni @ np.ones((m, 1))

    def get_lag1(x_: np.ndarray) -> float:
        return ((delta @ x_.reshape((m, m)).T @ f - 1) /
                (2 * delta @ np.ones((m, 1)) - 1)).item()

    # Find initial guess iterating through solutions of a linear problem
    # with linear constraints using simplex method.

    xs = []  # Here we store points simplex algorithm visited

    def callback(opt_res):
        xs.append(opt_res.x[:-2 * m])  # remove auxiliary variables

    # Simplex method here may warn about linear dependency, as well as
    # that some methods are not applicable. Since we are not interested
    # in linear problem solution (it is redundant) and we need only
    # boundary points of the feasibility region, ignore warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scipy.optimize.linprog(
            np.hstack([
                np.zeros(m ** 2),
                np.ones(2 * m) + np.random.normal(0, 0.5, 2 * m)]
            ),
            A_eq=np.hstack([A_eq, np.eye(2 * m)]),
            b_eq=b_eq,
            options={
                'maxiter': 100,  # at most 100 points will be put to xs
                'cholesky': False,
                'sym_pos': False,
            },
            method='simplex',  # here slower method is better :)
            callback=callback
        )

    # Define optimization function depending on the `optype` value:
    if optype == 'opt':
        # if we are interested in D1 for a MAP that has lag-1 autocorrelation
        # as close as possible to the given lag-1 value:
        if lag1 is None:
            raise ValueError("expected lag1 when optype = 'opt'")
        # use traditional MSE as loss function that we minimize:
        obj_fun = lambda x_: (get_lag1(x_) - lag1) ** 2
    elif optype == 'min':
        # if we are interested in minimum lag-1 value, then just use
        # lag-1 autocorrelation function as the objective function
        obj_fun = get_lag1
    elif optype == 'max':
        # if we are interested in maximum lag-1 value, then use negated
        # lag-1 autocorrelation:
        obj_fun = lambda x_: -get_lag1(x_)
    else:
        raise ValueError("expected 'opt', 'min' or 'max'")

    # Define a function that will be used as a constraint in optimization:
    def constraint(x_: np.ndarray) -> float:
        diff = A_eq @ x_.reshape((m ** 2, 1)) - b_eq
        return (diff.T @ diff).item()

    best_solution: Optional[scipy.optimize.OptimizeResult] = None

    # Make several attempts to find the best solution using different
    # initial points. Result with the lowest `fun` value is considered
    # to be the best.
    for it in range(n_iters + 1):  # always at least 1 iteration
        if it == 0:
            # In the first iteration we always use D1 matrix that is taken
            # from MAP representation of the PH distribution, it has
            # lag-1 autocorrelation 0.0
            markov_arrival = MarkovArrival.phase_type(ph.s, ph.init_probs)
            x_init = markov_arrival.d1.T.reshape((m ** 2,))
        else:
            # Take some point visited by the simplex method as initial
            # assumption. To randomize, we always start from the first point
            # and add random number of points, taken internally (also at
            # random distances) on the lines between current point and
            # other points, visited by the simplex algorithm.
            x_init = xs[0]
            for x in xs[1:]:
                if np.random.rand() <= 0.5:
                    continue
                alpha = np.random.rand()
                x_init = alpha * x_init + (1 - alpha) * x

        # Solve a problem with linear boundaries.
        new_solution = scipy.optimize.minimize(
            obj_fun,
            x0=x_init,
            bounds=[(0, None) for _ in range(m ** 2)],
            constraints={'type': 'eq', 'fun': constraint},
        )
        # Update best solution if needed.
        if best_solution is None or new_solution.fun < best_solution.fun:
            best_solution = new_solution

    if optype == 'max':
        # Invert result, since in maximization we used -get_lag1(x):
        best_solution.fun = -best_solution.fun

    return best_solution
