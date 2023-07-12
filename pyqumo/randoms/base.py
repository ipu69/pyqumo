from abc import ABC
from functools import cached_property, lru_cache
from typing import Union

import numpy as np

from pyqumo.stats import get_skewness
from .variables import VariablesFactory, Variable


default_randoms_factory = VariablesFactory()


class Distribution:
    """
    Base class for all distributions.

    Class :class:`Distribution` defines basic properties that should implement
    any real distribution like :attr:`mean` or :attr:`cv`.
    """
    def __init__(self, factory: VariablesFactory = None):
        self._factory = factory or default_randoms_factory

    @property
    def factory(self) -> VariablesFactory:
        return self._factory

    """
    Base class for all continuous distributions.
    """
    @cached_property
    def mean(self) -> float:
        """
        Return mean value.
        """
        return self._moment(1)

    @cached_property
    def rate(self) -> float:
        """
        Return rate (1/mean)
        """
        return 1 / self.mean

    @cached_property
    def var(self) -> float:
        """
        Return variance (dispersion) of the random variable.

        .. math::
            Var(X) = E[(X - E[x])^2]
        """
        return self._moment(2) - self._moment(1)**2

    @cached_property
    def std(self) -> float:
        """
        Return standard deviation of the random variable.
        """
        return self.var ** 0.5

    @cached_property
    def cv(self) -> float:
        """
        Get coefficient of variation (relation of std.dev. to mean value)
        """
        return self.std / self.mean

    @cached_property
    def skewness(self) -> float:
        """
        Get skewness.
        """
        return get_skewness(self.mean, self._moment(2), self._moment(3))

    def moment(self, n: int) -> float:
        """
        Get n-th moment of the random variable.

        Parameters
        ----------
        n : int
            moment degree, for n=1 it is mean value

        Returns
        -------
        value : float

        Raises
        ------
        ValueError
            raised if n is not an integer or is non-positive
        """
        if n < 0 or (n - np.floor(n)) > 0:
            raise ValueError(f'positive integer expected, but {n} found')
        if n == 0:
            return 1
        return self._moment(n)

    def _moment(self, n: int) -> float:
        """
        Compute n-th moment.
        """
        raise NotImplementedError

    def __call__(self, size: int = 1) -> Union[float, np.ndarray]:
        """
        Generate random samples of the random variable with this distribution.

        Parameters
        ----------
        size : int, optional
            number of values to generate (default: 1)

        Returns
        -------
        value : float or ndarray
            if size > 1, then returns a 1D array, otherwise a float scalar
        """
        if size == 1:
            return self.rnd.eval()
        return np.asarray([self.rnd.eval() for _ in range(size)])

    @property
    def order(self) -> int:
        """
        Get distribution order.

        By default, distribution has order 1 (e.g., normal, exponential,
        uniform, etc.). However, if the distribution is a kind of a
        sequence or mixture of distributions, it can have greater order.
        For instance, Erlang distribution shape is its order.
        """
        return 1

    @property
    def rnd(self) -> Variable:
        raise NotImplementedError

    def copy(self) -> 'Distribution':
        raise NotImplementedError

    def as_ph(self, **kwargs) -> 'PhaseType':
        """
        Get distribution representation in the form of a PH distribution.

        By default, raise `RuntimeError`, but can be overridden.
        """
        raise RuntimeError(f"{repr(self)} can not be casted to PhaseType")


class RandomProcess(ABC, Distribution):
    """
    Abstract base class for any random process.
    """

    @lru_cache
    def lag(self, n: int) -> float:
        """
        Return auto-correlation coefficient with lag n.

        Value of lag-n autocorrelation is defined as:

        .. math::
            r_k = (E[X_{t+n} - m_1][X_{t} - m_1]) / s^2,

        where :math:`m_1` - mean value and :math:`s^2` - variance (dispersion).

        Parameters
        ----------
        n : int
            Time lag - number of steps between intervals

        Returns
        -------
        value : float
            Auto-correlation coefficient

        Raises
        ------
        ValueError
            raised if n is not an integer or is non-positive
        """
        if n < 0 or (n - np.floor(n)) > 0:
            raise ValueError(f'positive integer expected, but {n} found')
        if n == 0:
            return 1
        return self._lag(n)

    def _lag(self, n: int) -> float:
        """
        Get lag-n autocorrelation. In this method it can be assumed that
        `n` is a non-zero positive integer.

        This method must be implemented in inherited classes.
        """
        raise NotImplementedError

    def copy(self) -> 'RandomProcess':
        """
        Return a deep copy of the object.

        This method must be implemented in inherited classes.
        """
        raise NotImplementedError

