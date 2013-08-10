#!/usr/bin/env python

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

from pymbar import EXP, EXPgauss, BAR, BARzero, MBAR

import timeseries
import testsystems
import confidenceintervals

__all__ = ['EXP', 'EXPgauss', 'BAR', 'BARzero', 'MBAR', 'timeseries', 'testsystems', 'confidenceintervals']

