# This file is part of sphstat.
#
# sphstat is licensed under the MIT License.
#
# Copyright (c) 2022 Huseyin Hacihabiboglu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
sphstat: A Python package for inferential statistics on vectorial data on the unit sphere.

Spherical data arises in many different fields of science. **sphstat**
provides the necessary tools to apply inferential statistics on
data on the unit sphere. The package implements tests and algorithms for
vectorial datagiven by Fisher, Lewis and Embleton [1]. Note that tests and
methods for axial data have not yet been implemented as of version 0.1.0.

AUTHOR

- Huseyin Hacihabiboglu, METU Spatial Audio Research Group (SPARG) https://www.sparglab.org, Ankara, Turkiye
- Github: https://www.github.com/hacihabiboglu
- E-mail: hhuseyin@metu.edu.tr
"""

import sphstat.distributions
import sphstat.twosample
import sphstat.utils
import sphstat.modelling
import sphstat.singlesample
import sphstat.descriptives
import sphstat.plotting

__version__ = "1.0"
__date__ = "15 December 2022"
__author__ = 'Huseyin Hacihabiboglu'
__credits__ = 'METU Spatial Audio Research Group, Ankara, Turkiye'