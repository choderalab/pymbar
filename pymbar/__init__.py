##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2010-2017 University of Colorado Boulder, Memorial Sloan-Kettering Cancer Center
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp, Levi Naden
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the MIT License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.
#
# You should have received a copy of the MIT License along with pymbar.
##############################################################################

"""The pymbar package contains the pymbar suite of tools for the analysis of
simulated and experimental data with the multistate Bennett acceptance
ratio (MBAR) estimator.

"""

__author__ = "Michael R. Shirts and John D. Chodera"
__license__ = "MIT"
__maintainer__ = "Levi N. Naden, Jaime Rodríguez-Guerra, Michael R. Shirts and John D. Chodera"
__email__ = "levi.naden@choderalab.org,jaime.rodriguez-guerra@choderalab.org,michael.shirts@colorado.edu,john.chodera@choderalab.org"

from . import timeseries, testsystems, confidenceintervals
from .mbar import MBAR
from .other_estimators import bar, bar_overlap, bar_zero, exp, exp_gauss
from .fes import FES

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "exp",
    "exp_gauss",
    "bar",
    "bar_overlap",
    "bar_zero",
    "MBAR",
    "timeseries",
    "testsystems",
    "confidenceintervals",
    "utils",
    "FES",
]
