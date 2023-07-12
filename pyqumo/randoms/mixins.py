from functools import lru_cache, cached_property
from typing import Callable, Iterator, Tuple, Sequence, Optional

import numpy as np
import scipy
from scipy.special import ndtr

from .variables import Variable, VariablesFactory
from .. import stats


class AbstractCdfMixin:
    """
    Mixin adds cumulative distribution function property prototype.
    """
    @property
    def cdf(self) -> Callable[[float], float]:
        """
        Get cumulative distribution function (CDF).
        """
        raise NotImplementedError


class ContinuousDistributionMixin:
    """
    Base mixin for continuous distributions, provides `pdf` callable property.
    """
    @property
    def pdf(self) -> Callable[[float], float]:
        """
        Get probability density function (PDF).
        """
        raise NotImplementedError


class DiscreteDistributionMixin:
    """
    Base mixin for discrete distributions, provides `pmf` property and iterator.
    """
    @property
    def pmf(self) -> Callable[[float], float]:
        """
        Get probability mass function (PMF).
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        """
        Iterate over (value, prob) pairs.
        """
        raise NotImplementedError


class AbsorbMarkovPhasedEvalMixin:
    """
    Mixin for RND for phased distributions with Markovian transitions.

    To use this mixin, distribution need to implement:

    - ``states`` (property): returns an iterable. Calling an item should
      return a sample value of the time spent in the state, size ``N``.
      Signature of each item ``__call__()`` method should support ``size``
      parameter.
    - ``init_probs`` (property): initial probability distribution, should have
      the same shape as ``time`` sequence (``N``).
    - ``trans_probs`` (property): matrix of transitions probabilities, should
      have shape ``(N+1)x(N+1)``, where the last state is an absorbing state.
    - ``order`` (property): should return the number of transient states (``N``)

    """
    @property
    def order(self) -> int:
        raise NotImplementedError

    @property
    def states(self) -> Sequence[Callable[[Optional[int]], np.ndarray]]:
        raise NotImplementedError

    @property
    def init_probs(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def trans_probs(self) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, size: int = 1):
        if size == 1:
            return self.rnd.eval()
        return np.asarray([self.rnd.eval() for _ in range(size)])

    @cached_property
    def rnd(self) -> Variable:
        variables = [state.rnd for state in self.states]
        return self.factory.absorb_semi_markov(
            variables,
            self.init_probs,
            self.trans_probs,
            self.order)


class EstStatsMixin:
    """
    Mixin for distributions without analytics form for moments computation.

    This mixin estimates moments, variance and standard deviation based on the
    sampled data.

    The derived class must provide ``num_stats_samples`` property, that returns
    the number of samples those should to be used in moments estimation.
    """
    @property
    def num_stats_samples(self) -> int:
        raise NotImplementedError

    @cached_property
    def _stats_samples(self) -> np.ndarray:
        """
        Get samples cache to estimate moments and/or other properties.

        If cache doesn't exist, it will be created`.
        """
        if not hasattr(self, '__stats_samples'):
            self.__stats_samples = self.__call__(self.num_stats_samples)
        return self.__stats_samples

    @lru_cache
    def _moment(self, n: int) -> float:
        return stats.moment(self._stats_samples, minn=n, maxn=n)[0]


class KdePdfMixin:
    """
    Mixin for distributions without analytics form for PDF and CDF computation.

    This mixin estimates PDF and CDF functions using Gaussian KDE from
    ``scipy``. Derived class must implement `num_kde_samples` property, that
    returns number of samples should to be used in KDE building.

    ... note::
        ``num_kde-samples`` shouldn't be too large since KDE works VERY slowly.
    """
    @property
    def num_kde_samples(self) -> int:
        raise NotImplementedError

    @cached_property
    def _kde(self) -> scipy.stats.gaussian_kde:
        if not hasattr(self, '__kde'):
            kde_samples = self.__call__(self.num_kde_samples)
            self.__kde = scipy.stats.gaussian_kde(kde_samples)
        return self.__kde

    @property
    def pdf(self):
        return lambda x: self._kde.pdf(x)[0]

    @property
    def cdf(self):
        dataset = self._kde.dataset
        factor = self._kde.factor
        return lambda x: ndtr(np.ravel(x - dataset) / factor).mean()
