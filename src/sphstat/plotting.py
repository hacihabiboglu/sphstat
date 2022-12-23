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
===========================
Functions for plotting data
===========================

- :func:`plotmapping` maps the azimuth from [-pi,pi) to [0, 2 * pi]
- :func:`plotdata` plots the data in a given projection
- :func:`plotdatalist` plots a number of samples

"""

import numpy as np
from matplotlib import pyplot as plt

from .descriptives import mediandir, rotationmatrix, pointsonanellipse
from .utils import polartocart, carttopolar, cart2sph


def plotmapping(input: list | float):
    """
    Utility function to map an angle in [0, 2 * pi] to [-pi, pi]

    :param input: Input angle in [0, 2 * pi] in radians to map to [-pi, pi]
    :type input: list | float
    :return: Mapped angle in radians
    :rtype: float
    """

    output = []
    if type(input)==list:
        output = input.copy()
    else:
        output.append(input)

    for ind in range(len(input)):
        if not (0<= input[ind] < 2 * np.pi):
            input[ind] = np.mod(input[ind], 2 * np.pi)
        if input[ind] > np.pi:
            input[ind] -= 2 * np.pi

    for ind in range(len(output)):
        if 2 * np.pi > output[ind] > np.pi:
             output[ind] -= 2 * np.pi
    if type(input) == float:
        output = output[0]
    return output


def plotdata(sample: dict, proj: str='mollweide', mflag: bool = False) -> bool:
    """
    Plot a sample represented in polar (colat, long) format in Mollweide projection

    :param sample: Sample in rad format (polar)
    :type sample: dict
    :param proj: Projection type (either 'mollweide' or 'lambert')
    :type proj: str
    :param mflag: Flag to plot the median and the 95% cone of confidence
    :type mflag: bool
    :return: bool (True)
    """
    try:
        assert sample['type'] == 'rad'
    except AssertionError:
        raise AssertionError('Sample type should be rad.')

    try:
        assert proj in ['mollweide', 'lambert']
    except AssertionError:
        raise AssertionError('Unknown projection type!')

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=proj)

    phis = plotmapping(sample['phis'])
    ax.scatter(np.array(phis), np.pi / 2 - np.array(sample['tetas']), s=1.5 * plt.rcParams['lines.markersize'] ** 1.5,
               marker='o', alpha=0.5)  # convert degrees to radians
    if mflag:
        samplec = polartocart(sample)
        medi, cuss, cc, W = mediandir(samplec)
        h1 = np.array([1, 0, 0])
        h2 = np.array([0, 1, 0])
        h3 = np.array([0, 0, 1])
        A = rotationmatrix(medi[0], medi[1], 0)
        h1 = A.T @ h1
        h2 = A.T @ h2
        h3 = A.T @ h3

        pts = pointsonanellipse(h1, h2, h3, cc['coeffs'])
        scc = dict()
        scc['points'] = pts
        scc['type'] = 'cart'
        scc['n'] = 360
        scr = carttopolar(scc)
        phis = plotmapping(scr['phis'])
        thes = np.array(scr['tetas'])
        ax.scatter(phis, np.pi / 2 - thes, s=1.5 * plt.rcParams['lines.markersize'], marker='.', color='k', alpha=0.5)
        thm, phm = cart2sph(h3)
        phl = plotmapping([phm])
        ax.scatter(phl, np.pi / 2 - thm, s=1.5 * plt.rcParams['lines.markersize'] ** 2, marker='s', color='k',
                   alpha=0.5)

    tick_labels_x, tick_labels_y = [], []
    if proj == 'mollweide':
        tick_labels_x = [210, 240, 270, 300, 330, 0, 30, 60, 90, 120, 150]
        tick_labels_y = np.array([15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0])
    elif proj == 'lambert':
        tick_labels_x = [None, 240, 270, 300, 330, 0, 30, 60, 90, 120, None]
        tick_labels_y = [] #np.array([15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0])

    ax.set_xticklabels(tick_labels_x)  # we add the scale on the x axis
    ax.set_yticklabels(tick_labels_y)  # we add the scale on the x axis
    ax.title.set_fontsize(15)
    ax.set_xlabel("Longitude [deg]")
    ax.xaxis.label.set_fontsize(12)
    ax.set_ylabel("Colatitude [deg]")
    ax.yaxis.label.set_fontsize(12)
    ax.grid(True)
    plt.show()
    return True


def plotdatalist(samplelist: list, labels: list=None, proj: str='mollweide', mflag: bool = False) -> bool:
    """
    Superimposed plot of a list of samples

    :param samplelist: List containing multiple samples
    :type samplelist: list
    :param labels: List of (string) labels e.g. ['A', 'B']
    :type labels: list
    :param proj: Projection type (either 'mollweide' or 'lambert')
    :type proj: str
    :param mflag: Flag to plot the median and the 95% cone of confidence
    :type mflag: bool
    :return: Return True when completed
    :rtype: bool
    """

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection=proj)

    try:
        assert len(labels) == len(samplelist)
    except AssertionError:
        raise AssertionError('Number of labels should match the number of samples in samplelist')

    ind = -1
    for sample in samplelist:
        ind += 1
        assert sample['type'] == 'rad'
        # phis = sample['phis']
        phis = plotmapping(sample['phis'])
        ax.scatter(np.array(phis), np.pi / 2 - np.array(sample['tetas']), s= 1.5 * plt.rcParams['lines.markersize']**1.5, marker='o', label=labels[ind], alpha=0.5)  # convert degrees to radians
        if mflag:
            samplec = polartocart(sample)
            medi, cuss, cc, W = mediandir(samplec)
            h1 = np.array([1, 0, 0])
            h2 = np.array([0, 1, 0])
            h3 = np.array([0, 0, 1])
            A = rotationmatrix(medi[0], medi[1], 0)
            h1 = A.T @ h1
            h2 = A.T @ h2
            h3 = A.T @ h3

            pts = pointsonanellipse(h1, h2, h3, cc['coeffs'])
            scc = dict()
            scc['points'] = pts
            scc['type'] = 'cart'
            scc['n'] = 360
            scr = carttopolar(scc)
            phis = plotmapping(scr['phis'])
            thes = np.array(scr['tetas'])
            ax.scatter(phis, np.pi / 2 - thes, s=1.5 * plt.rcParams['lines.markersize'], marker='.', color='k', alpha=0.5)
            thm, phm = cart2sph(h3)
            ax.scatter(phm, np.pi / 2 - thm, s=1.5 * plt.rcParams['lines.markersize'] ** 2, marker='s', color='k',
                       alpha=0.5)

    tick_labels_x, tick_labels_y = [], []
    if proj=='mollweide':
        tick_labels_x = [210, 240, 270, 300, 330, 0, 30, 60, 90, 120, 150]
        tick_labels_y = np.array([15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0])
    elif proj=='lambert':
        tick_labels_x = [None , 240, 270, 300, 330, 0, 30, 60, 90, 120, None]
        tick_labels_y = [] # np.array([15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0])

    fdict= {'fontsize': plt.rcParams['axes.titlesize'],
     'fontweight': plt.rcParams['axes.titleweight'],
     'verticalalignment': 'baseline'}

    ax.set_xticklabels(tick_labels_x, fontdict=fdict)  # we add the scale on the x axis
    ax.set_yticklabels(np.flip(tick_labels_y), fontdict=fdict)

    ax.title.set_fontsize(15)
    ax.set_xlabel("Longitude [deg]")
    ax.xaxis.label.set_fontsize(16)
    ax.set_ylabel('Colatitude [deg]')
    ax.yaxis.label.set_fontsize(16)
    ax.grid(True)
    plt.legend(fontsize='large', loc=1)
    plt.show()
    return True
