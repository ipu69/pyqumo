from functools import cached_property

import numpy as np

from .cont_dist import Exponential
from .base import RandomProcess, Distribution
from .variables import VariablesFactory, Variable


class GIProcess(RandomProcess):
    """
    General independent (GI) random process model.

    Samples of this kind of process are built from a known distribution,
    and they don't change over lifetime.

    Poisson process is an example of GI-process with exponential arrivals.
    """
    def __init__(self, dist: Distribution, factory: VariablesFactory = None):
        """
        Constructor.

        Parameters
        ----------
        dist : random distribution
        """
        super().__init__(factory)
        if dist is None:
            raise ValueError('distribution required')
        self._dist = dist

    @property
    def dist(self) -> Distribution:
        """
        Get random distribution.
        """
        return self._dist

    @cached_property
    def mean(self) -> float:
        return self._dist.mean

    @cached_property
    def var(self) -> float:
        return self._dist.var

    @cached_property
    def std(self) -> float:
        return self._dist.std

    @cached_property
    def cv(self) -> float:
        return self._dist.std / self._dist.mean

    @property
    def rnd(self) -> Variable:
        return self._dist.rnd

    def _moment(self, n: int) -> float:
        return self._dist.moment(n)

    def _lag(self, k: int) -> float:
        return 0.0  # always 0.0

    def _eval(self, size: int) -> np.ndarray:
        return self._dist(size)

    def copy(self) -> 'GIProcess':
        return GIProcess(self._dist.copy())

    def __repr__(self):
        return f'(GI: f={self.dist})'


class Poisson(GIProcess):
    """
    Custom case of GI-process with exponential arrivals.
    """
    def __init__(self, rate: float, factory: VariablesFactory = None):
        """
        Constructor.

        Parameters
        ----------
        rate : float
            Rate (1 / mean) of exponential distribution.
        """
        if rate <= 0.0:
            raise ValueError(f"positive rate expected, {rate} found")
        super().__init__(Exponential(rate), factory)

    def __repr__(self):
        return f'(Poisson: r={self.rate:.3g})'
