"""
Package pymbar

This package contains the pymbar suite of tools for the analysis of
simulated and experimental data with the multistate Bennett acceptance
ratio (MBAR) estimator.

"""

__author__ = "Michael R. Shirts and John D. Chodera"
__version__ = "2.0beta"
__license__ = "GPL"
__maintainer__ = "Michael R. Shirts and John D. Chodera"
__email__ = "michael.shirts@virginia.edu,choderaj@mskcc.org"

import pymbar
from pymbar.mbar import MBAR
from pymbar.bar import BAR, BARzero
from pymbar.exponential_averaging import EXP, EXPGauss

import pymbar.timeseries
import pymbar.testsystems
import pymbar.confidenceintervals

__all__ = ['EXP', 'EXPGauss', 'BAR', 'BARzero', 'MBAR', 'timeseries', 'testsystems', 'confidenceintervals', 'utils']

