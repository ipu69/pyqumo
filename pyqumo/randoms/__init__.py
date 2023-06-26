"""
Random distributions and arrival processes (:mod:`pyqumo.randoms`)
==================================================================

Module provides models for random variables distributions and stochastic
processes.

Basic distributions
-------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Const


Continuous distributions
------------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Uniform
    Normal
    Exponential
    Erlang
    MixtureDistribution
    HyperExponential
    HyperErlang
    PhaseType
    SemiMarkovAbsorb


Discrete distributions
----------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Choice
    CountableDistribution


Arrival processes
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   GIProcess
   Poisson
   MarkovArrival


Base classes and mixins
-----------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Distribution
    RandomProcess
    AbstractCdfMixin
    ContinuousDistributionMixin
    DiscreteDistributionMixin
    MixtureDistribution
    AbsorbMarkovPhasedEvalMixin
    EstStatsMixin
    KdePdfMixin


Sampling utilities
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Variable
    VariablesFactory


Details
-------
The common base class is :class:`Distribution`. All distributions are made up
from combinations of various mixins:


Random processes used in common queueing systems model time intervals between
successive packets or batches arrivals, as well as the time taken to serve a
packet.

Consider new packets arrive in the system at timestamps
:math:`t_0, t_1, t_2, \\dots, t_n, t_{n+1}, \\dots`. Then, i-th interval is
:math:`x_i = t_{i} - t_{i-1}`. If all intervals :math:`\\{ x_i \\}` have the
same random distributions and don't depend on previous values, the process
is called *general independent process*, or GI-process. We model it with
:class:`pyqumo.arrivals.GIProcess`. Any distribution from :mod:`pyqumo.random`
may be used for intervals in this type of process.

A typical example of GI-processes is Poisson process - a random process,
in which all inter-arrival intervals have exponential distribution with the
fixed rate :math:`\\lambda`. Since this random process is very important in
queueing theory and used very often, we implement it in
:class:`pyqumo.arrivals.Poisson`.

On the other hand, intervals :math:`x_i` and :math:`x_{i+k}, k > 0`
may be correlated. This is the case when the process have some inner (hidden)
state space and make state transitions after each event. For instance,
consider a process in which odd intervals are always equal :math:`T` and
even intervals are exponentially distributed with parameter :math:`\\lambda`.

In current version, PyQumo supports only one type of correlated
processes - Markovian arrival process (:class:`pyqumo.arrivals.MarkovArrival`).
"""

from .base import Distribution, RandomProcess

from .mixins import AbstractCdfMixin, ContinuousDistributionMixin, \
    DiscreteDistributionMixin, AbsorbMarkovPhasedEvalMixin, EstStatsMixin, \
    KdePdfMixin

from .const_dist import Const

from .cont_dist import Uniform, Normal, Exponential, Erlang, \
    MixtureDistribution, HyperErlang, HyperExponential, PhaseType

from .discr_dist import Choice, CountableDistribution

from .basic_proc import Poisson, GIProcess
from .markov_arrival import MarkovArrival
from .variables import VariablesFactory, Variable
from .semi_markov import SemiMarkovAbsorb
