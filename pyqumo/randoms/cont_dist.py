from functools import cached_property, lru_cache
from typing import Callable, Sequence, Optional

import numpy as np
from scipy import linalg
from scipy import integrate

from .base import Distribution
from .mixins import ContinuousDistributionMixin, AbstractCdfMixin, \
    AbsorbMarkovPhasedEvalMixin
from .variables import VariablesFactory, Variable
from ..matrix import str_array, order_of, MatrixShapeError, \
    is_subinfinitesimal, fix_infinitesimal, is_pmf, fix_stochastic, cbdiag


class Uniform(ContinuousDistributionMixin, AbstractCdfMixin, Distribution):
    """
    Uniform random distribution.

    Notes
    -----
    PDF function :math:`f(x) = 1/(b-a)` anywhere inside ``[a, b]``,
    and zero otherwise. CDF function `F(x)` is equal to 0 for ``x < a``,
    1 for ``x > b`` and :math:`F(x) = (x - a)/(b - a)` anywhere inside
    ``[a, b]``.

    Moment :math:`m_n` for any natural number n is computed as:

    .. math::
        m_n = 1/(n+1) (a^0 b^n + a^1 b^{n-1} + ... + a^n b^0).

    Variance :math:`Var(x) = (b - a)^2 / 12`.
    """
    def __init__(self, a: float = 0, b: float = 1,
                 factory: VariablesFactory = None):
        super().__init__(factory)
        self._a, self._b = a, b

    @property
    def min(self) -> float:
        return self._a if self._a < self._b else self._b

    @property
    def max(self) -> float:
        return self._b if self._a < self._b else self._a

    @lru_cache
    def _moment(self, n: int) -> float:
        a_degrees = np.power(self._a, np.arange(n + 1))
        b_degrees = np.power(self._b, np.arange(n, -1, -1))
        return 1 / (n + 1) * a_degrees.dot(b_degrees)

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        k = 1 / (self.max - self.min)
        return lambda x: k if self.min <= x <= self.max else 0

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        a, b = self.min, self.max
        k = 1 / (b - a)
        return lambda x: 0 if x < a else 1 if x > b else k * (x - a)

    @cached_property
    def rnd(self) -> Variable:
        return self.factory.uniform(self.min, self.max)

    def __repr__(self):
        return f'(Uniform: a={self.min:g}, b={self.max:g})'

    def copy(self) -> 'Uniform':
        return Uniform(self._a, self._b)


class Normal(ContinuousDistributionMixin, AbstractCdfMixin, Distribution):
    """
    Normal random distribution.
    """
    def __init__(self, mean: float, std: float,
                 factory: VariablesFactory = None):
        super().__init__(factory)
        self._mean, self._std = mean, std

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std

    @cached_property
    def var(self) -> float:
        return self._std**2

    @lru_cache
    def _moment(self, n: int) -> float:
        m, s = self._mean, self._std

        if n == 1:
            return m
        elif n == 2:
            return m**2 + s**2
        elif n == 3:
            return m**3 + 3 * m * (s**2)
        elif n == 4:
            return m**4 + 6 * (m**2) * (s**2) + 3 * (s**4)

        # If moment order is too large, try to numerically solve it using
        # `scipy.integrate` module `quad()` routine:

        # noinspection PyTypeChecker
        return integrate.quad(lambda x: x**n * self.pdf(x), -np.inf, np.inf)[0]

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        k = 1 / np.sqrt(2 * np.pi * self.var)
        return lambda x: k * np.exp(-(x - self.mean)**2 / (2 * self.var))

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        k = 1 / (self.std * 2**0.5)
        return lambda x: 0.5 * (1 + np.math.erf(k * (x - self.mean)))

    @cached_property
    def rnd(self) -> Variable:
        return self.factory.normal(self.mean, self.std)

    def __repr__(self):
        return f'(Normal: mean={self._mean:.3g}, std={self._std:.3g})'

    def copy(self) -> 'Normal':
        return Normal(self._mean, self._std)


class Exponential(ContinuousDistributionMixin, AbstractCdfMixin, Distribution):
    """
    Exponential random distribution.
    """
    def __init__(self, rate: float, factory: VariablesFactory = None):
        super().__init__(factory)
        if rate <= 0.0:
            raise ValueError("exponential parameter must be positive")
        self._param = rate

    @property
    def order(self) -> int:
        return 1

    @property
    def param(self):
        return self._param

    @lru_cache
    def _moment(self, n: int) -> float:
        return np.math.factorial(n) / (self.param**n)

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        r = self.rate
        base = np.e ** -r
        return lambda x: r * base**x if x >= 0 else 0.0

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        r = self.rate
        base = np.e ** -r
        return lambda x: 1 - base**x if x >= 0 else 0.0

    def __str__(self):
        return f"(Exp: rate={self.rate:g})"

    def copy(self) -> 'Exponential':
        return Exponential(self._param)

    @cached_property
    def rnd(self) -> Variable:
        return self.factory.exponential(self.rate)

    @staticmethod
    def fit(avg: float) -> 'Exponential':
        """
        Build a distribution for a given average.

        Parameters
        ----------
        avg : float

        Returns
        -------

        """
        return Exponential(1 / avg)

    def as_ph(self, **kwargs):
        return PhaseType.exponential(self.rate)


class Erlang(ContinuousDistributionMixin, AbstractCdfMixin, Distribution):
    """
    Erlang random distribution.

    To create a distribution its shape (k) and rate (lambda) parameters must
    be specified. Its density function is defined as:

    .. math::

        f(x; k, l) = l^k x^(k-1) e^(-l * x) / (k-1)!
    """

    def __init__(self, shape: int, param: float,
                 factory: VariablesFactory = None):
        super().__init__(factory)
        if (shape <= 0 or shape == np.inf or
                np.abs(np.round(shape) - shape) > 0):
            raise ValueError("shape must be positive integer")
        if param <= 0.0:
            raise ValueError("rate must be positive")
        self._shape, self._param = int(np.round(shape)), param

    def as_ph(self, **kwargs):
        """
        Get representation of this Erlang distribution as PH distribution.
        """
        return PhaseType.erlang(self.shape, self.param)

    @property
    def shape(self) -> int:
        return self._shape

    @property
    def order(self) -> int:
        return self._shape

    @property
    def param(self) -> float:
        return self._param

    @lru_cache
    def _moment(self, n: int) -> float:
        """
        Return n-th moment of Erlang distribution.

        N-th moment of Erlang distribution with shape `K` and rate `R` is
        computed as: :math:`k (k+1) ... (k + n - 1) / r^n`
        """
        k, r = self.shape, self.param
        return k / r if n == 1 else (k + n - 1) / r * self._moment(n - 1)

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        r, k = self.param, self.shape
        koef = r**k / np.math.factorial(k - 1)
        base = np.e**(-r)
        return lambda x: 0 if x < 0 else koef * x**(k - 1) * base**x

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        # Prepare data
        r, k = self.param, self.shape
        factorials = np.cumprod(np.concatenate(([1], np.arange(1, k))))
        # Summation coefficients are: r^0 / 0!, r^1 / 1!, ... r^k / k!:
        koefs = np.power(r, np.arange(self.shape)) / factorials
        base = np.e**(-r)

        # CDF is given with:
        #   1 - e^(-r*x) ((r^0/0!) * x^0 + (r^1/1!) x^1 + ... + (r^k/k!) x^k):
        return lambda x: 0 if x < 0 else \
            1 - base**x * koefs.dot(np.power(x, np.arange(k)))

    @cached_property
    def rnd(self) -> Variable:
        return self.factory.erlang(self.shape, self.param)

    def __repr__(self):
        return f"(Erlang: shape={self.shape:.3g}, rate={self.param:.3g})"

    def copy(self) -> 'Erlang':
        return Erlang(self._shape, self._param)

    @staticmethod
    def fit(avg: float, std: float) -> 'Erlang':
        """
        Fit Erlang distribution for a given average and standard deviation.

        Parameters
        ----------
        avg : float
        std : float

        Returns
        -------
        dist : Erlang distribution
        """
        cv = std / avg
        if cv >= 1:
            return Erlang(1, 1/avg)
        rate = avg / std**2
        shape = int(np.round(avg**2 / std**2))
        return Erlang(shape, rate)


class MixtureDistribution(
    ContinuousDistributionMixin,
    AbstractCdfMixin,
    Distribution
):
    """
    Mixture of continuous distributions.

    This is defined by:

    - a sequence of distributions `xi_0, xi_1, ..., xi_{N-1}`
    - a sequence of weights `w_0, w_1, ..., w_{N-1}`

    The resulting probability is a weighted sum of the given distributions:

    :math:`f(x) = w_0 f_{xi_0}(x) + ... + w_{N-1} f_{xi_{N-1}}(x)`
    """
    def __init__(self, states: Sequence[Distribution],
                 weights: Optional[Sequence[float]] = None,
                 factory: VariablesFactory = None):
        super().__init__(factory)
        num_states = len(states)
        if num_states == 0:
            raise ValueError("no distributions provided")
        if weights is not None:
            if len(states) != len(weights):
                raise ValueError(
                    f"expected equal number of states and weights, "
                    f"but {len(states)} and {weights} found")
            weights = np.asarray(weights)
            if (weights < 0).any():
                raise ValueError(f"negative weights disallowed: {weights}")
            self._probs = weights / weights.sum()
        else:
            weights = np.ones(num_states)
            self._probs = 1/num_states * weights
        # Store distributions as a new tuple:
        self._states = tuple(states)
        self._num_states = num_states

    @property
    def states(self) -> Sequence[Distribution]:
        return self._states

    @property
    def probs(self) -> np.ndarray:
        return self._probs

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def order(self) -> int:
        """
        Get the distribution order as the sum of orders of internal
        distributions.
        """
        return sum(state.order for state in self._states)

    @lru_cache
    def _moment(self, n: int) -> float:
        moments = np.asarray([st.moment(n) for st in self._states])
        return self._probs.dot(moments)

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        fns = [state.pdf for state in self._states]
        return lambda x: sum(p * f(x) for (p, f) in zip(self.probs, fns))

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        fns = [state.cdf for state in self._states]
        return lambda x: sum(p * f(x) for (p, f) in zip(self.probs, fns))

    @cached_property
    def rnd(self) -> Variable:
        variables = [state.rnd for state in self.states]
        return self.factory.mixture(variables, self.probs)

    def __repr__(self):
        states_str = "[" + ", ".join(str(state) for state in self.states) + "]"
        probs_str = str_array(self._probs)
        return f"(Mixture: states={states_str}, probs={probs_str})"

    def copy(self) -> 'MixtureDistribution':
        return MixtureDistribution(
            [state.copy() for state in self._states],
            self._probs
        )

    def as_ph(self, **kwargs):
        """
        Get representation in the form of PH distribution.

        If any internal distribution with probability greater or equal to
        `min_prob` can not be represented as PH distribution,
        raise `RuntimeError`.

        Parameters
        ----------
        min_prob : float, optional
            all states with probability below this value will not be included
            into the PH distribution (default: 1e-5)

        Returns
        -------
        ph : PhaseType
        """
        min_prob = kwargs.get('min_prob', 1e-5)
        state_ph = []
        state_probs = []
        for p, state in zip(self.probs, self.states):
            if p >= min_prob:
                state_ph.append(state.as_ph(**kwargs))
                state_probs.append(p)
        order = sum(ph.order for ph in state_ph)
        mat = np.zeros((order, order))
        probs = np.zeros(order)
        base = 0
        for p, ph in zip(state_probs, state_ph):
            n = ph.order
            probs[base] = p
            mat[base:base+n, base:base+n] = ph.s
            base += n
        probs = np.asarray(probs) / sum(probs)
        return PhaseType(mat, probs)


class HyperExponential(MixtureDistribution):
    """Hyper-exponential distribution.

    Hyper-exponential distribution is defined by:

    - a vector of rates (a1, ..., aN)
    - probabilities mass function (p1, ..., pN)

    Then the resulting probability is a weighted sum of exponential
    distributions Exp(ai) with weights pi:

    $X = \\sum_{i=1}^{N}{p_i X_i}$, where $X_i ~ Exp(ai)$
    """
    def __init__(self, rates: Sequence[float], probs: Sequence[float],
                 factory: VariablesFactory = None):
        exponents = [Exponential(rate) for rate in rates]
        super().__init__(exponents, probs, factory)

    @cached_property
    def rates(self) -> np.ndarray:
        return np.asarray([state.rate for state in self.states])

    def __repr__(self):
        return f"(HyperExponential: " \
               f"probs={str_array(self.probs)}, " \
               f"rates={str_array(self.rates)})"

    @staticmethod
    def fit(avg: float, std: float, skew: float = 0) -> 'HyperExponential':
        """
        Fit hyperexponential distribution with average, std and skewness.

        Parameters
        ----------
        avg
        std
        skew
        """
        # TODO: add support for skewness
        cv = std / avg
        if cv <= 1:
            return HyperExponential([1/avg], [1.0])

        a = avg
        b = (std**2 + avg**2) / 2

        r2 = 1/a + 1.0
        r1 = (a*r2 - 1) / (b*r2 - a)
        p1 = (a*r2 - 1)**2 / (b*r2**2 - 2*a*r2 + 1)
        p2 = 1 - p1
        if p1 < 0 or p1 > 1 or r1 < 0:
            raise RuntimeError(f"failed to fit hyperexponential distribution:"
                               f"avg = {avg}, std={std}; resulting p1 = {p1}, "
                               f"r1 = {r1}; selected r2={r2}.")

        return HyperExponential([r1, r2], [p1, p2])


class HyperErlang(MixtureDistribution):
    """Hyper-Erlang distribution.

    Hyper-Erlang distribution is defined by:

    - a vector of parameters (a1, ..., aN)
    - a vector of shapes (n1, ..., nN)
    - probabilities mass function (p1, ..., pN)

    Then the resulting probability is a weighted sum of Erlang
    distributions Erlang(ni, ai) with weights pi:

    $X = \\sum_{i=1}^{N}{p_i X_i}$, where $X_i ~ Er(ni, ai)$
    """
    def __init__(
            self,
            params: Sequence[float],
            shapes: Sequence[int],
            probs: Sequence[float],
            factory: VariablesFactory = None):
        states = [Erlang(shape, param) for shape, param in zip(shapes, params)]
        super().__init__(states, probs, factory)

    @cached_property
    def params(self) -> np.ndarray:
        return np.asarray([state.param for state in self.states])

    @cached_property
    def shapes(self) -> np.ndarray:
        return np.asarray([state.shape for state in self.states])

    def __repr__(self):
        return f"(HyperErlang: " \
               f"probs={str_array(self.probs)}, " \
               f"shapes={str_array(self.shapes)}, " \
               f"params={str_array(self.params)})"


class PhaseType(ContinuousDistributionMixin,
                AbstractCdfMixin,
                AbsorbMarkovPhasedEvalMixin,
                Distribution):
    """
    Phase-type (PH) distribution.

    This distribution is specified with a subinfinitesimal matrix and
    initial states probability distribution.

    PH distribution is a generalization of exponential, Erlang,
    hyperexponential, hypoexponential and hypererlang distributions, so
    they can be defined using PH distribution means. However, PH distribution
    operates with matrices, incl. matrix-exponential operations, so it is
    less efficient then custom implementations.
    """
    def __init__(self, sub: np.ndarray, p: np.ndarray, safe: bool = False,
                 factory: VariablesFactory = None, tol: float = 1e-3):
        super().__init__(factory)
        # Validate and fix data:
        # ----------------------
        if not safe:
            if (sub_order := order_of(sub)) != order_of(p):
                raise MatrixShapeError(f'({sub_order},)', p.shape, 'PMF')
            if not is_subinfinitesimal(sub):
                sub = fix_infinitesimal(sub, sub=True, tol=tol)[0]
            if not is_pmf(p):
                p = fix_stochastic(p, tol=tol)[0]

        # Store data in fields:
        # ---------------------
        self._subgenerator = sub
        self._pmf0 = p
        self._sni = -np.linalg.inv(sub)  # (-S)^{-1} - negated inverse of S

        # Build internal representations for transitions PMFs and rates:
        # --------------------------------------------------------------
        self._order = order_of(self._pmf0)
        self._rates = -self._subgenerator.diagonal()
        self._trans_probs = np.hstack((
            self._subgenerator + np.diag(self._rates),
            -self._subgenerator.sum(axis=1)[:, None]
        )) / self._rates[:, None]
        self._states = [Exponential(r) for r in self._rates]

    @staticmethod
    def exponential(rate: float) -> 'PhaseType':
        sub = np.asarray([[-rate]])
        p = np.asarray([1.0])
        return PhaseType(sub, p, safe=True)

    @staticmethod
    def erlang(shape: int, rate: float) -> 'PhaseType':
        blocks = [
            (0, np.asarray([[-rate]])),
            (1, np.asarray([[rate]]))
        ]
        sub = cbdiag(shape, blocks)
        p = np.zeros(shape)
        p[0] = 1.0
        return PhaseType(sub, p, safe=True)

    @staticmethod
    def hyperexponential(rates: Sequence[float], probs: Sequence[float]):
        order = len(rates)
        sub = np.zeros((order, order))
        for i, rate in enumerate(rates):
            sub[(i, i)] = -rate
        if not isinstance(probs, np.ndarray):
            probs = np.asarray(probs)
        return PhaseType(sub, probs, safe=False)

    @cached_property
    def order(self) -> int:
        return order_of(self._subgenerator)

    @property
    def s(self):
        return self._subgenerator

    @property
    def init_probs(self):
        return self._pmf0

    @property
    def p(self):
        return self._pmf0

    @property
    def trans_probs(self):
        return self._trans_probs

    @property
    def states(self):
        return self._states

    @property
    def sni(self) -> np.ndarray:
        return self._sni

    def scale(self, k: float) -> 'PhaseType':
        """
        Get new PH distribution with mean = (this PH).mean x K.
        """
        return PhaseType(self.s / k, self.p, safe=True)

    @lru_cache
    def _moment(self, n: int) -> float:
        sni_powered = np.linalg.matrix_power(self.sni, n)
        ones = np.ones(shape=(self.order, 1))
        x = np.math.factorial(n) * self.init_probs.dot(sni_powered).dot(ones)
        return x.item()

    @cached_property
    def pdf(self) -> Callable[[float], float]:
        p = np.asarray(self._pmf0)
        s = np.asarray(self._subgenerator)
        tail = -s.dot(np.ones(self.order))
        return lambda x: 0 if x < 0 else p.dot(linalg.expm(x * s)).dot(tail)

    @cached_property
    def cdf(self) -> Callable[[float], float]:
        p = np.asarray(self._pmf0)
        ones = np.ones(self.order)
        s = np.asarray(self._subgenerator)
        return lambda x: 0 if x < 0 else \
            1 - p.dot(linalg.expm(x * s)).dot(ones)

    def __repr__(self):
        return f"(PH: s={str_array(self.s)}, p={str_array(self.init_probs)})"

    def copy(self) -> 'PhaseType':
        return PhaseType(self._subgenerator, self._pmf0, safe=True)

    def as_ph(self, **kwargs) -> 'PhaseType':
        return self.copy()
