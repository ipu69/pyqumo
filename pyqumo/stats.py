"""
************
pyqumo.stats
************

Statistical utility functions.
"""
from collections import namedtuple
from typing import List, Sequence

import numpy as np

from .matrix import str_array


def rel_err(expected, actual):
    """Get relative error between of two values.
    """
    if abs(expected) < 1e-10:
        return abs(actual)
    return abs(expected - actual) / abs(expected)


def get_cv(m1: float, m2: float) -> float:
    """Compute coefficient of variation.
    """
    return (m2 - m1**2)**0.5 / m1


def get_skewness(m1: float, m2: float, m3: float) -> float:
    """Compute skewness.
    """
    var = m2 - m1**2
    std = var**0.5
    return (m3 - 3*m1*var - m1**3) / (var * std)


def get_noncentral_m3(mean: float, cv: float, skew: float) -> float:
    """Compute non-central third moment if mean, CV and skew provided.
    """
    m1 = mean
    std = cv * mean
    var = std**2
    return skew * var * std + 3 * m1 * var + m1**3


def get_noncentral_m2(mean: float, cv: float) -> float:
    """Compute non-central from the mean value and coef. of variation.
    """
    return (mean * cv)**2 + mean**2


def moment(source, maxn=1, minn=1):
    """Computes moments from a given source.

    Args:
        source (array-like, distribution or arrival):
            data to compute moments for. Treated as a set of samples
            if an array-like, otherwise is expected to have a `moment(k)`
            method (e.g. distribution or arrival process)
        maxn (int): maximum number of moment to compute.
        minn (int): minimum number of moment to compute

    Returns:
        ndarray containing computed moments
    """
    if hasattr(source, 'moment'):
        return np.asarray([source.moment(k) for k in range(1, maxn + 1)])
    if not isinstance(source, np.ndarray):
        source = np.asarray(source)
    ret = [np.power(source, k).mean() for k in range(minn, maxn + 1)]
    return np.asarray(ret)


def lag(source, maxn=1):
    """Computes lags from a given source.

    Args:
        source (array-like or arrival):
            data to compute moments for. Treated as a set of samples
            if an array-like, otherwise is expected to have a `lag(k)`
            method (e.g. arrival process)
        maxn (int): a number of lags to compute.

    Returns:
        ndarray containing computed lags
    """
    if hasattr(source, 'lag'):
        return np.asarray([source.lag(k) for k in range(1, maxn + 1)])
    else:
        n = len(list(source))
        maxn = min(maxn, n - 1)  # need at least k + 1 samples for k-th lag
        moments = moment(source, 2)
        m1 = moments[0]
        sigma2 = moments[1] - moments[0] ** 2
        ret = []
        for k in range(1, maxn + 1):
            values = [(source[i] - m1) * (source[i + k] - m1)
                      for i in range(n - k)]
            ret.append(sum(values) / ((n - k) * sigma2))
        return np.asarray(ret)


def normalize_moments(moments, k=None):
    """
    Normalizes moments using a given or computed coefficient.

    Parameters
    ----------
    moments : array-like
        an array of the first moments

    k : scalar, callable or None
        a coefficient. If `None`, it will be :math:`m_1` if only one
        moment is given, or :math:`\\frac{m_2}{m_1}`.
        If scalar, this scalar will be used. If callable, it will be called
        as ``k(moments)`` to get the coefficient.

    Returns
    -------
        normalized_moments : ndarray
            normalized moments vector,
        mu : float
            the coefficient being used.
    """
    # TODO: add unit test
    if k is None:
        try:
            mu = moments[1] / moments[0]
        except IndexError:
            mu = moments[0]
    else:
        try:
            mu = k(moments)
        except TypeError:
            mu = k
    ret = list(moments)
    for i in range(len(ret)):
        ret[i] /= pow(mu, i + 1)
    return np.asarray(ret), mu


Statistics = namedtuple('Statistics', ['avg', 'var', 'std', 'count'])


class TimeSizeRecords:
    """
    Recorder for time-size statistics.

    Key feature of time-size records is that size varies on a natural numbers
    axis, so it is not float, negative, or something else. Thus, we
    store durations of each size in an array. If new value is larger then
    this array, the array is extended.

    Array of durations is easily converted into PMF just dividing it on the
    total time. Assuming initial time is always zero, total time is
    the time when the last update was recorded.

    One more thing to mention is that when recording values, actually previous
    value is recorded: when we call `add(ti, vi)`, it means that at `ti`
    new value became `vi`. However, we store information that the _previous_
    value `v{i-1}` was kept for `t{i} - t{i-1}` interval. Thus, we store
    the previous value (by default, zero) and previous update time.
    """
    def __init__(self, init_value: int = 0, init_time: float = 0.0):
        self._durations: List[float] = [0.0]
        self._updated_at: float = init_time
        self._curr_value: int = init_value
        self._init_time = init_time

    def add(self, time: float, value: int):
        """
        Record new value at the given time.

        When called, duration of the _previous_ value is actually incremented:
        if, say, system size at time `t2` became equal `v2`, then we need
        to store information, that for interval `t2 - t1` value _was_ `v1`.

        New value and update time are stored, so in the next `add()` call
        they will be used to save interval of current value.

        Parameters
        ----------
        time : float
        value : int
        """
        prev_value = self._curr_value
        self._curr_value = value
        num_cells = len(self._durations)
        if prev_value >= num_cells:
            num_new_cells = prev_value - num_cells + 1
            self._durations.extend([0.0] * num_new_cells)
        self._durations[prev_value] += time - self._updated_at
        self._updated_at = time

    @property
    def pmf(self) -> np.ndarray:
        """
        Normalize durations to get PMF.
        """
        return (np.asarray(self._durations) /
                (self._updated_at - self._init_time))

    def __repr__(self):
        return f"(TimeSizeRecords: durations={str_array(self._durations)})"


def build_statistics(intervals: Sequence[float]) -> Statistics:
    """
    Build Statistics object from the given sequence of intervals.

    Parameters
    ----------
    intervals : 1D array_like

    Returns
    -------
    statistics : Statistics
    """
    if len(intervals) > 0:
        avg = np.mean(intervals)
        var = np.var(intervals, ddof=1)  # unbiased estimate
        std = var**0.5
    else:
        avg = 0.0
        var = 0.0
        std = 0.0
    return Statistics(avg=avg, var=var, std=std, count=len(intervals))
