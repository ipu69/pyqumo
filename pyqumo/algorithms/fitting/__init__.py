"""
**************
pyqumo.fitting
**************

Module :mod:`pyqumo.fitting` provides routines for fitting random distributions
and arrival processes based on real data, sample traces or given moments.

.. autosummary::
    :toctree: generated/
    :recursive:

    acph2
    horvath05
    johnson89
"""

from .acph2 import fit_acph2, get_acph2_m3_bounds, get_acph2_m2_min
from .johnson89 import fit_mern2
from .horvath05 import fit_map_horvath05, optimize_lag1
