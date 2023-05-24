"""
**************
pyqumo.fitting
**************

Module provides routines for fitting random distributions and arrival
processes based on real data, sample traces or given moments.
"""

from .acph2 import fit_acph2
from .johnson89 import fit_mern2
from .horvath05 import fit_map_horvath05, optimize_lag1
