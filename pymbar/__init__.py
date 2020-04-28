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
__maintainer__ = "Levi N. Naden, Michael R. Shirts and John D. Chodera"
__email__ = "levi.naden@choderalab.org,michael.shirts@colorado.edu,john.chodera@choderalab.org"

from pymbar.logging_utils import setup_logging

setup_logging()

from pymbar import timeseries, testsystems, confidenceintervals
from pymbar.mbar import MBAR
from pymbar.other_estimators import bar, bar_zero, exp, exp_gauss
from pymbar.pmf import PMF
import pymbar.old_mbar

try:
    from pymbar.version import *
except ImportError:
    # Fill in information manually.
    # TODO: See if we can at least get the git revision info in here.
    version = "dev"
    full_version = "dev"
    git_revision = "dev"
    isrelease = False

__all__ = [
    "exp",
    "exp_gauss",
    "bar",
    "bar_zero",
    "MBAR",
    "timeseries",
    "testsystems",
    "confidenceintervals",
    "utils",
    "PMF",
]
