"""
Random distributions (:mod:`pyqumo.random`)
===========================================

Module provides models for random variables distributions.
These models are used in random arrivals (see :mod:`pyqumo.arrivals`).

Concrete distributions defined in this module are:

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Const
    Uniform
    Normal
    Exponential
    Erlang
    HyperExponential
    HyperErlang
    Choice
    CountableDistribution
    PhaseType
    SemiMarkovAbsorb

The common base class is :class:`Distribution`. All distributions are made up
from combinations of various mixins:

.. autosummary::
    :toctree: generated/
    :nosignatures:

    Distribution
    AbstractCdfMixin
    ContinuousDistributionMixin
    DiscreteDistributionMixin
    MixtureDistribution
    AbsorbMarkovPhasedEvalMixin
    EstStatsMixin
    KdePdfMixin
"""