##############################################################################
# pymbar: A Python Library for MBAR
#
# Copyright 2010-2014 University of Virginia, Memorial Sloan-Kettering Cancer Center
#
# Authors: Michael Shirts, John Chodera
# Contributors: Kyle Beauchamp
#
# pymbar is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with pymbar. If not, see <http://www.gnu.org/licenses/>.
##############################################################################

"""The pymbar package contains the pymbar suite of tools for the analysis of
simulated and experimental data with the multistate Bennett acceptance
ratio (MBAR) estimator.

"""

__author__ = "Michael R. Shirts and John D. Chodera"
__license__ = "LGPL"
__maintainer__ = "Michael R. Shirts and John D. Chodera"
__email__ = "michael.shirts@virginia.edu,choderaj@mskcc.org"

from pymbar import timeseries, testsystems, confidenceintervals, version
from pymbar.mbar import MBAR
from pymbar.bar import BAR, BARzero
from pymbar.exp import EXP, EXPGauss
import pymbar.old_mbar

try:
    from pymbar import version
except:
    # Fill in information manually.
    # TODO: See if we can at least get the git revision info in here.
    version = 'dev'
    full_version = 'dev'
    git_revision = 'dev'
    isrelease = False

__all__ = ['EXP', 'EXPGauss', 'BAR', 'BARzero', 'MBAR', 'timeseries', 'testsystems', 'confidenceintervals', 'utils']
