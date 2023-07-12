from functools import cached_property, lru_cache
from typing import Sequence, Mapping, Union, Callable, Iterator, Tuple

import numpy as np

from .base import Distribution
from .mixins import DiscreteDistributionMixin, AbstractCdfMixin
from .variables import VariablesFactory, Variable
from ..matrix import fix_stochastic, is_pmf, str_array


class Choice(DiscreteDistributionMixin, AbstractCdfMixin, Distribution):
    """
    Discrete distribution of values with given non-negative weights.
    """
    def __init__(self, values: Sequence[float],
                 weights: Union[Mapping[float, float], Sequence[float]] = None,
                 factory: VariablesFactory = None):
        """
        Discrete distribution constructor.

        Different values probabilities are computed based on weights as
        :math:`p_i = w_i / (w_1 + w_2 + ... + w_N)`.

        Parameters
        ----------
        values : sequence of values
        weights : mapping of values to weights or a sequence of weights, opt.
            if provided as a sequence, then weights length should be equal
            to values length; if not provided, all values are expected to
            have the same weight.
        """
        super().__init__(factory)
        if len(values) == 0:
            raise ValueError('expected non-empty values')
        values_ = []
        try:
            # First we assume that weights is dictionary. In this case we
            # expect that it stores values in pairs like `value: weight` and
            # iterate through it using `items()` method to fill value and
            # weights arrays:
            # noinspection PyUnresolvedReferences
            weights_ = []
            for key, weight in weights.items():
                values_.append(key)
                weights_.append(weight)
            weights_ = np.asarray(weights_)
        except AttributeError:
            # If `values` doesn't have `items()` attribute, we treat as an
            # iterable holding values only. Then we check whether its size
            # matches weights (if provided), and fill weights it they were
            # not provided:
            values_.extend(values)
            if weights is not None and len(weights) > 0:
                if len(values) != len(weights):
                    raise ValueError('values and weights size mismatch')
            else:
                weights = (1. / len(values),) * len(values)
            weights_ = np.asarray(weights)

        # Check that all weights are non-negative and their sum is positive:
        if np.any(weights_ < 0):
            raise ValueError('weights must be non-negative')
        total_weight = sum(weights_)
        if np.allclose(total_weight, 0):
            raise ValueError('weights sum must be positive')

        # Store values and probabilities
        probs_ = weights_ / total_weight
        self._data = [(v, p) for v, p in zip(values_, probs_)]
        self._data.sort(key=lambda item: item[0])

    @cached_property
    def values(self) -> np.ndarray:
        return np.asarray([item[0] for item in self._data])

    @cached_property
    def probs(self) -> np.ndarray:
        return np.asarray([item[1] for item in self._data])

    @lru_cache()
    def __len__(self):
        return len(self._data)

    def find_left(self, value: float) -> int:
        """
        Searches for the value and returns the closest left side index.

        Examples
        --------
        >>> choices = Choice([1, 3, 5], [0.2, 0.5, 0.3])
        >>> choices.find_left(1)
        >>> 0
        >>> choices.find_left(2)  # not in values, return leftmost value index
        >>> 0
        >>> choices.find_left(5)
        >>> 2
        >>> choices.find_left(-1)  # for too small values return -1
        >>> -1

        Parameters
        ----------
        value : float
            value to search for

        Returns
        -------
        index : int
            if `value` is found, return its index; if not, but there is value
            `x < value` and there are no other values `y: x < y < value` in
            data, return index of `x`. If for any `x` in data `x > value`,
            return `-1`.
        """
        def _find(start: int, end: int) -> int:
            delta = end - start
            if delta < 1:
                return -1
            if delta == 1:
                return start if value >= self.values[start] else -1
            middle = start + delta // 2
            middle_value = self.values[middle]
            if np.allclose(value, middle_value):
                return middle
            if value < middle_value:
                return _find(start, middle)
            return _find(middle, end)
        return _find(0, len(self))

    def get_prob(self, value: float) -> float:
        """
        Get probability of a given value.

        Parameters
        ----------
        value : float

        Returns
        -------
        prob : float
        """
        index = self.find_left(value)
        stored_value = self.values[index]
        if index >= 0 and np.allclose(value, stored_value):
            return self.probs[index]
        return 0.0

    @lru_cache
    def _moment(self, n: int) -> float:
        return (self.values**n).dot(self.probs).sum()

    @cached_property
    def rnd(self):
        return self.factory.choice(self.values, self.probs)

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        cum_probs = np.cumsum(self.probs)

        def fn(x):
            index = self.find_left(x)
            return cum_probs[index] if index >= 0 else 0.0

        return fn

    @cached_property
    def pmf(self) -> Callable[[float], float]:
        return lambda x: self.get_prob(x)

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        for value, prob in self._data:
            yield value, prob

    def __repr__(self):
        return f"(Choice: values={self.values.tolist()}, " \
               f"p={self.probs.tolist()})"

    def copy(self) -> 'Choice':
        return Choice(self.values, self.probs)


class CountableDistribution(DiscreteDistributionMixin,
                            AbstractCdfMixin,
                            Distribution):
    """
    Distribution with set of values {0, 1, 2, ...} - all non-negative numbers.

    An example of this kind of distributions is geometric distribution.
    Since a set of values is infinite, we specify this distribution with
    a probability function that takes value = 0, 1, ..., and returns its
    probability.

    To compute properties we will need to find sum of infinite series.
    We specify precision as maximum tail probability, and use only first
    values (till this tail) when estimating sums.
    """
    def __init__(self,
                 prob: Union[Callable[[int], float], Sequence[float]],
                 precision: float = 1e-9,
                 max_value: int = np.inf,
                 moments: Sequence[float] = (),
                 factory: VariablesFactory = None):
        """
        Constructor.

        Parameters
        ----------
        prob : callable (int) -> float, or array_like
            Probability function. If given in functional form,
            should accept arguments 0, 1, 2, ... and return their probability.
            If an array instance of length N, treated as probability mass
            function of values 0, 1, 2, ... N. Length N + 1 is assumed to be
            maximum value.
        precision : float, optional
            Maximum tail probability. If this tail starts at value X=N,
            properties will be estimated over values 0, 1, ..., N only,
            without respect to the tail. Note, that when estimating
            moments of high order the error will grow due to the growth
            of tail weight (for n > N > 1, n**K > n for K > 1).
            If `max_value < np.inf` or `prob` is given in array form,
            this argument is ignored. By default 1e-9.
        max_value : int, optional
            If provided, specifies the maximum possible value. Any value
            above it will have zero probability. If this argument is provided,
            `precision` is ignored. If `prob` is given in array form,
            this argument is ignored, and max value is assigned to
            the length of `prob` array minus one. By default `np.inf`.
        moments : sequence of floats, optional
            Optional explicit moments values. If given, they will be
            used instead of estimating over first 0, 1, ..., N values.
            By default, empty tuple.
        """
        super().__init__(factory)
        self._prob = prob
        self._precision = precision
        self._moments = moments

        try:
            # Treat `prob` as array_like defining a probability mass function:
            self._pmf = np.asarray(list(prob))
            if not is_pmf(self._pmf):
                self._pmf = fix_stochastic(self._pmf, tol=0.1)[0]
            self._max_value = len(self._pmf) - 1
            self._truncated_at = self._max_value
            self._hard_max_value = True
        except TypeError:
            # Not iterable - assume `prob` to be a callable [(x) -> pr.]:
            pmf_ = []
            if max_value >= np.inf:
                # If max_value is not fixed, find number I, such that
                # P[X > I] < precision:
                head_prob = 0.0
                self._hard_max_value = False
                self._max_value = np.inf
                self._truncated_at = -1
                while head_prob < 1 - precision:
                    self._truncated_at += 1
                    p = prob(self._truncated_at)
                    pmf_.append(p)
                    head_prob += p
                self._pmf = np.asarray(pmf_)

            elif max_value >= 0:
                # Max value is not infinite - use it as the truncation point:
                self._pmf = np.asarray([prob(x) for x in range(max_value + 1)])
                self._max_value = max_value
                self._truncated_at = max_value
                self._hard_max_value = True

            else:
                raise ValueError(f"non-negative max_value expected, "
                                 f"but {max_value} found")

        self._trunc_cdf = np.cumsum(self._pmf)
        values = tuple(range(self._truncated_at + 1))
        self._trunc_choice = Choice(values, weights=self._pmf)

    @lru_cache
    def get_prob_at(self, x: int) -> float:
        """
        Get probability of a given value.

        This method is cached, so only first access at a given value may
        be long. For the values 0, 1, ..., `truncated_at` this method is
        fast even at the first run, since these values are computed in
        constructor when computing `truncated_at` value itself.
        For greater values access to `prob(x)` may take time.

        Returns 0.0 for negative arguments, no matter of `prob(x)`.

        Parameters
        ----------
        x : int, non-negative

        Returns
        -------
        probability : float
        """
        if 0 <= x <= self._truncated_at:
            return self._pmf[x]
        if self._hard_max_value:
            return 0.0
        return self._prob(x) if x > self._truncated_at else 0.0

    @property
    def prob(self) -> Callable[[int], float]:
        """
        Returns probability function.
        """
        return self._prob

    @property
    def precision(self) -> float:
        """
        Returns precision of this distribution.
        """
        return self._precision

    @property
    def truncated_at(self) -> int:
        """
        Returns a value, such that tail probability is less then precision.

        If `truncated_at = N`, then total probability of values
        N+1, N+2, ... is less then `precision`.
        """
        return self._truncated_at

    @property
    def max_value(self) -> int:
        """
        Returns maximum value the random value can take.
        """
        return self._max_value

    @lru_cache
    def _moment(self, n: int) -> float:
        """
        Computes n-th moment.

        If moments were provided in the constructor, use them. Otherwise
        estimate moment as the weighted sum of the first N elements, where
        `truncated_at = N`. Note, that for large `n` error may be large,
        and, if needed, distribution should be built with high precision.

        Parameters
        ----------
        n : int
            Order of the moment.

        Returns
        -------
        value : float
        """
        if n <= len(self._moments) and self._moments[n-1] is not None:
            return self._moments[n-1]
        values = np.arange(self._truncated_at + 1)
        degrees = np.power(values, n)
        probs = np.asarray([self.get_prob_at(x) for x in values])
        return probs.dot(degrees)

    @cached_property
    def pmf(self) -> Callable[[float], float]:
        """
        This function returns probability mass function.

        For values, those are NOT non-negative integers, returns 0.0
        Works very fast for values between 0 and `truncated_at` (incl.),
        but requires call to `prob(x)` for greater values.

        Returns
        -------
        pmf : callable (float) -> float
        """
        if self._hard_max_value:
            def hard_fn(x: float) -> float:
                fl, num = np.math.modf(x)
                if (abs(fl) <= 1e-12 and
                        0 <= (num := int(num)) <= self._truncated_at):
                    return self._pmf[num]
                return 0.0
            return hard_fn

        def soft_fn(x: float) -> float:
            fl, num = np.math.modf(x)
            if abs(fl) <= 1e-12 and (num := int(num)) >= 0:
                return self.get_prob_at(num)
            return 0.0
        return soft_fn

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        """
        This function returns cumulative distribution function.

        For values, those are NOT non-negative integers, returns 0.0
        Works very fast for values between 0 and `truncated_at` (incl.),
        but requires call to `prob(x)` for greater values.

        Returns
        -------
        pmf : callable (float) -> float
        """
        if self._hard_max_value:
            def hard_fn(x: float) -> float:
                num = min(int(np.math.modf(x)[1]), self._truncated_at)
                return self._trunc_cdf[num] if num >= 0 else 0.0
            return hard_fn

        def soft_fn(x: float) -> float:
            _, num = np.math.modf(x)
            num = int(num)
            if num < 0:
                return 0.0
            if num <= self._truncated_at:
                return self._trunc_cdf[num]
            p = self._trunc_cdf[self._truncated_at]
            for i in range(self._truncated_at + 1, num + 1):
                p += self.get_prob_at(i)
            return p

        return soft_fn

    def __iter__(self) -> Iterator[Tuple[float, float]]:
        # To avoid infinite loops, we iterate over 10-times max value.
        total_prob = 0.0
        for i in range(10 * (self._truncated_at + 1)):
            if total_prob >= 1 - 1e-12:
                return
            p = self.get_prob_at(i)
            total_prob += p
            yield i, p

    @cached_property
    def rnd(self) -> Variable:
        return self._trunc_choice.rnd

    # def _eval(self, size: int) -> np.ndarray:
    #     """
    #     Generate a random array of the given size.

    #     When generating random values, use `Choice` distribution with values
    #     0, 1, ..., `truncated_at`. Thus, no values from tail (which prob. is
    #     less than precision) will be generated.

    #     Parameters
    #     ----------
    #     size : array size

    #     Returns
    #     -------
    #     array : np.ndarray
    #     """
    #     return self._trunc_choice(size)

    def __repr__(self):
        if not self._hard_max_value:
            values = ', '.join([
                f"{self.get_prob_at(x):.3g}" for x in range(5)
            ])
            return f"(Countable: p=[{values}, ...], "\
                   f"precision={self.precision})"
        return f"(Countable: p={str_array(self._pmf)})"

    def copy(self) -> 'CountableDistribution':
        if self._hard_max_value:
            return CountableDistribution(
                self._prob, self._truncated_at,
                moments=self._moments)
        return CountableDistribution(self._prob, self._precision,
                                     moments=self._moments)


