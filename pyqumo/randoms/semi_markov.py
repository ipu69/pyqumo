from typing import Union, Sequence, Iterable

import numpy as np

from .base import Distribution
from .mixins import AbsorbMarkovPhasedEvalMixin, EstStatsMixin, \
    KdePdfMixin
from .variables import VariablesFactory

from ..matrix import is_pmf, is_square, order_of, is_substochastic, \
    str_array
from ..errors import MatrixShapeError


class SemiMarkovAbsorb(AbsorbMarkovPhasedEvalMixin,
                       EstStatsMixin,
                       KdePdfMixin,
                       Distribution):
    """
    Semi-Markov process with absorbing states.

    Process with `N` states is specified with three parameters:

    - random distribution of time in each state `T[i], i = 0, 1, ..., N-1`
    - transition strictly sub-stochastic matrix `P` of shape `(N, N)`
    - initial probability distribution `p0` of shape `(N,)`

    Process starts in one of its states as defined with `p0`. It spends time
    in each state as defined by the corresponding distribution `T[i]` and
    after that move to another state selected with probabilities `P'[i]`.
    Here `P'` = `[[P; I - P*1], [0, 1]]` - a stochastic matrix built from
    sub-stochastic matrix `P` by appending a column to the right with missing
    probabilities of transiting to absorbing state:
    `P'[i,N] = 1 - (P[i,0] + P[i,1] + ... + P[i,N-1])`. We require P
    to be strictly sub-stochastic, so at least at one row this value should
    be non-zero.

    To generate random samples, we model the behavior of this process
    multiple times. We don't analyze reachability, so loops are possible.

    Note: this method doesn't try to fix sub-stochastic transition matrix if
    it is not valid, in contrast to phase-type distributions or MAP processes.
    """
    def __init__(self, trans: Union[np.ndarray, Sequence[Sequence[float]]],
                 time_dist: Sequence[Distribution],
                 probs: Union[np.ndarray, Sequence[float], int] = 0,
                 num_samples: int = 10000,
                 num_kde_samples: int = 1000,
                 factory: VariablesFactory = None):
        """
        Constructor.

        Parameters
        ----------
        trans : array_like
            sub-stochastic transition matrix between non-absorbing states
            with shape `(N, N)`.
        time_dist : sequence of distributions
            distribution of time spent in state, should have length `N`.
        probs : array_like or int
            initial distribution of length `N`, or the number of the first
            state. If the state number is provided (int), then process will
            start from this state with probability 1.0.
        num_samples : int, optional
            number of samples that is used for moments estimation.
            As a rule of thumb, should be large enough, especially if
            high order moments needed.
            By default: 10'000
        num_kde_samples : int, optional
            number of samples that is used for Gaussian KDE building
            for PDF and CDF estimation.
            As a rule of thumb, should NOT be too large, since using PDF
            of KDE built from too large number of samples is very slow.
            By default: 1'000
        """
        super().__init__(factory)
        # Convert to ndarray and validate transitions matrix:
        if not isinstance(trans, np.ndarray):
            trans = np.asarray(trans)
        else:
            trans = trans.copy()  # copy to avoid changes from outside

        if not is_square(trans):
            raise MatrixShapeError("(N, N)", trans.shape, "transitions matrix")
        order = order_of(trans)
        if not is_substochastic(trans):
            raise ValueError(f"expected sub-stochastic matrix, "
                             f"but {trans} found")

        # Validate time_dist and p0, convert p0 to ndarray if needed:
        if len(time_dist) != order:
            raise ValueError(f"need {order} time distributions, "
                             f"{len(time_dist)} found")

        if isinstance(probs, Iterable):
            if not isinstance(probs, np.ndarray):
                probs = np.asarray(probs)
            else:
                probs = probs.copy()  # copy to avoid changes from outside
            if not is_pmf(probs):
                raise ValueError(f"PMF expected, but {probs} found")
            if (p0_order := order_of(probs)) != order:
                raise ValueError(f"expected P0 vector with order {order}, "
                                 f"but {p0_order} found")
        else:
            # If here, assume p0 is an integer - number of state. Build
            # initial PMF with 1.0 set for this state.
            if not (0 <= probs < order):
                raise ValueError(f"semi-Markov process of order {order} "
                                 f"doesn't have transient state {probs}")
            p0_ = np.zeros(order)
            p0_[probs] = 1.0
            probs = p0_

        # Store matrices and parameters:
        self._trans = trans
        self._order = order
        self._states = tuple(time_dist)  # copy to avoid changes
        self._init_probs = probs
        self._num_samples = num_samples
        self._num_kde_samples = num_kde_samples

        # Build full transitions stochastic matrix:
        self._trans_probs = np.vstack((
            np.hstack((
                trans,
                np.ones((order, 1)) - trans.sum(axis=1).reshape((order, 1))
            )),
            np.asarray([[0] * order + [1]]),
        ))

        # Define cache that is used for moments estimations:
        # --------------------------------------------------
        self.__samples = None
        self.__kde = None

    @property
    def init_probs(self):
        return self._init_probs

    @property
    def trans_probs(self):
        return self._trans_probs

    @property
    def order(self):
        return self._order

    @property
    def states(self):
        return self._states

    @property
    def num_stats_samples(self):
        return self._num_samples

    @property
    def num_kde_samples(self) -> int:
        return self._num_kde_samples

    def __repr__(self):
        trans = str_array(self._trans_probs)
        time_ = "[" + ', '.join([str(td) for td in self._states]) + "]"
        probs = str_array(self._init_probs)
        return f"(SemiMarkovAbsorb: trans={trans}, time={time_}, p0={probs})"

    def copy(self) -> 'SemiMarkovAbsorb':
        return SemiMarkovAbsorb(
            self._trans,
            [dist.copy() for dist in self._states],
            self._init_probs,
            num_samples=self._num_samples,
            num_kde_samples=self._num_kde_samples
        )
