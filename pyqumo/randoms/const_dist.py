from functools import cached_property, lru_cache
from typing import Callable, Iterator, Tuple

import numpy as np

from .base import Distribution
from .mixins import ContinuousDistributionMixin, DiscreteDistributionMixin, \
    AbstractCdfMixin
from .variables import VariablesFactory, Variable


class Const(ContinuousDistributionMixin, DiscreteDistributionMixin,
            AbstractCdfMixin, Distribution):
    """
    Constant distribution that always results in a given constant value.
    """
    def __init__(self, value: float, factory: VariablesFactory = None):
        super().__init__(factory)
        self._value = value
        self._next = None

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        return lambda x: np.inf if x == self._value else 0

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        return lambda x: 0 if x < self._value else 1

    @cached_property
    def pmf(self) -> Callable[[float], float]:
        return lambda x: 1 if x == self._value else 0

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        yield self._value, 1.0

    @lru_cache
    def _moment(self, n: int) -> float:
        return self._value ** n

    @cached_property
    def rnd(self) -> Variable:
        return self.factory.constant(self._value)

    def __repr__(self):
        return f'(Const: value={self._value:g})'

    def copy(self) -> 'Const':
        return Const(self._value)

