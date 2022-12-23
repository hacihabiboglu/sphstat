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
========================================================================
Functions to generate random data from different spherical distributions
========================================================================

- :func:`uniform` generates samples from uniform spherical distribution
- :func:`bingham` generates samples from uniform Bingham distribution
- :func:`fisherbingham` generates samples from Fisher/Bingham distribution
- :func:`kent` generates samples from Kent distribution
- :func:`fisher` generates samples from Fisher distribution
- :func:`watson` generates samples from Watson distribution

"""

import numpy as np
from numpy.random import default_rng

from .utils import polartocart, cart2sph, sph2cart, negatesample, poolsamples
from .descriptives import rotatesample, rotationmatrix_withaxis


def uniform(numsamp: int) -> dict:
    """
    Generate uniformly sampled data on the unit sphere

    :param numsamp: Number of samples to generate
    :type numsamp: int
    :return: Data dictionary of type 'cart' containing numsamp uniformly distributed data
    :rtype: dict
    """
    rng = default_rng()
    n1 = rng.standard_normal(numsamp)
    n2 = rng.standard_normal(numsamp)
    n3 = rng.standard_normal(numsamp)
    rd = np.sqrt(n1 ** 2 + n2 ** 2 + n3 ** 2)
    x = n1 / rd
    y = n2 / rd
    z = n3 / rd
    samplecart = dict()
    samplecart['n'] = numsamp
    samplecart['type'] = 'cart'
    samplecart['points'] = []
    for ind in range(numsamp):
        samplecart['points'].append(np.array((x[ind], y[ind], z[ind])))
    return samplecart


def bingham(numsamp: int, lamb: float) -> dict:
    """
    Generate Bingham distributed data on the unit sphere

    :param numsamp: Number of samples
    :type numsamp: int
    :param lamb: Eigenvalyes of the diagpnal symmetric matrix of the Bingham distribution in decreasing order
    :type lamb: np.array
    :return: Data dictionary of type 'cart' containing numsamp Bingham distributed data
    :rtype: dict
    """
    rng = default_rng()
    lam = lamb
    # lam = np.flip( np.sort(lamb) )
    n = 0
    samplecart = dict()
    samplecart['n'] = numsamp
    samplecart['type'] = 'cart'
    samplecart['points'] = []
    lamfull = np.concatenate((lam, [0]))
    qa = len(lamfull)
    mu = np.zeros(qa)
    sigacginv = 1 + 2 * lamfull
    sigacg = np.sqrt(1 / sigacginv)
    while n < numsamp:
        xsamp = False
        while not xsamp:
            yp = rng.normal(mu, sigacg, (1, qa))
            y = yp / np.linalg.norm(yp)
            lratio = -1 * np.sum(y ** 2 * lamfull) - qa / 2 * np.log(qa) \
                + 0.5 * (qa - 1) + qa / 2 * np.log(np.sum(y ** 2 * sigacginv))

            if np.log(rng.uniform(0, 1, 1)) < lratio:
                samplecart['points'].append(y)
                xsamp = True
                n += 1
    return samplecart


def fisherbingham(numsamp: int, alpha: float, beta: float, kappa: float, A: np.ndarray) -> dict:
    """
    Generate Fisher-Bingham distributed data on the unit sphere [1]_, [2]_

    :param numsamp: number of samples
    :type numsamp: int
    :param alpha: Inclination angle of the mean (0 <= alpha < pi)
    :type alpha: float
    :param beta: Azimuth angle of the mean (0 <= beta < 2 * pi)
    :type beta: float
    :param kappa: Concentration parameter
    :type kappa: float
    :param A: Symmetric matrix for the Bingham part
    :return: Data dictionary of type 'cart' containing numsamp FB distributed data
    :rtype: dict

    [1] Kent J.T., Ganeiber A.M. and Mardia K.V. (2013). A new method to simulate the Bingham and related distributions in directional data analysis with applications.

    [2] https://rdrr.io/cran/Directional/man/rfb.html
    """
    rng = default_rng()
    samplecart = dict()
    samplecart['n'] = numsamp
    samplecart['type'] = 'cart'
    samplecart['points'] = []
    pt = sph2cart(alpha, beta)
    mu0 = np.array([0., 1., 0.])
    B = rotationmatrix_withaxis(mu0, pt)
    q = len(mu0)
    A1 = A + kappa / 2 * (np.eye(q) - mu0.reshape((3, 1)) @ mu0.reshape((1, 3)))
    lam, V = np.linalg.eig(A1)
    idx = lam.argsort()[::-1]
    lam = lam[idx]
    V = V[:, idx]
    lam -= lam[q - 1]
    lam = lam[:q - 1]
    sbingham = bingham(numsamp, lam)
    pts = sbingham['points'].copy()
    ptsnew = []
    xn = []
    u = np.log(rng.uniform(0, 1, numsamp))
    for pt in pts:
        ptn = pt.reshape(1, 3) @ V.T
        ptsnew.append(ptn[0])
        xn.append(ptn[0][1])
    xn = np.array(xn)
    pt = np.array(ptsnew)
    ffb = kappa * xn - np.sum(pt @ A @ pt.T, axis=1)
    fb = kappa - np.sum(pt @ A1 @ pt.T, axis=1)
    x1 = pt[np.where(u <= (ffb - fb))[0], :]
    n1 = np.shape(x1)[0]
    while n1 < numsamp:
        sbingham = bingham(numsamp - n1, lam)
        pts = sbingham['points'].copy()
        ptsnew = []
        xn = []
        u = np.log(rng.uniform(0, 1, numsamp - n1))
        for pt in pts:
            ptn = pt.reshape(1, 3) @ V.T
            ptsnew.append(ptn)
            xn.append(ptn[0][1])
        xn = np.array(xn)
        pt = np.array(ptsnew).reshape((numsamp - n1, 3))
        ffb = kappa * xn - np.sum(pt @ A @ pt.T, axis=1)
        fb = kappa - np.sum(pt @ A1 @ pt.T, axis=1)
        x1 = np.concatenate((x1, pt[np.where(u < (ffb - fb))[0], :]), axis=0)
        n1 = np.shape(x1)[0]
    x = x1 @ B.T
    for rw in range(np.shape(x)[0]):
        pt = x[rw, :]
        samplecart['points'].append(pt)
    return samplecart


def kent(numsamp: int, kappa: float, beta: float, mu: np.array, mu0: np.array) -> dict:
    """
    Generate Kent (5-parameter Fisher-Bingham - FB5) distributed data on the unit sphere

    :param numsamp: Number of samples to generate
    :type numsamp: int
    :param kappa: Concentration parameter
    :param beta: Ovalness parameter
    :param mu: Mean vector of Kent distribution
    :type mu: np.array
    :param mu0: Mean vector of the Fisher part
    :type mu0: np.array
    :return: Data dictionary of type 'cart' containing numsamp Kent distributed data
    :rtype: dict
    """

    mu0 /= np.linalg.norm(mu0)
    alph, bet = cart2sph(mu0)
    mu /= np.linalg.norm(mu)
    a = rotationmatrix_withaxis(mu0, mu)
    A = np.diag([-beta, 0, beta])
    samplerbf = fisherbingham(numsamp, alph, bet, kappa, A)
    pts = samplerbf['points'].copy()
    samplekent = dict()
    samplekent['n'] = numsamp
    samplekent['type'] = 'cart'
    samplekent['points'] = []
    for pt in pts:
        ptn = pt.reshape((1, 3)) @ a.T
        samplekent['points'].append(ptn[0] / np.linalg.norm(ptn[0]))
    return samplekent


def fisher(numsamp: int, alpha: float, beta: float, kappa: float) -> dict:
    """
    Generate von Mises-Fisher distributed data on the unit sphere [1]_

    :param numsamp: Number of samples to generate
    :type numsamp: int
    :param alpha: Inclination angle centroid (0<= alpha <=pi)
    :type alpha: float
    :param beta: Azimuth angle centroid (0 <= beta < 2 * pi)
    :type beta: float
    :param kappa: Concentration parameter
    :type kappa: float
    :return: Data dictionary of type 'cart' containing numsamp Fisher distributed data
    :rtype: dict

    [1] Fisher, N. I., Lewis, T. & Willcox, M. E. (1981). Tests of discordancy for samples from Fisher's distribution on the sphere. Appl. Statist. 30, 230-237.
    """
    rng = default_rng()
    r1 = rng.uniform(0, 1, numsamp)
    r2 = rng.uniform(0, 1, numsamp)
    lamb = np.exp(-2 * kappa)
    the = 2 * np.arcsin(np.sqrt(-np.log(r1 * (1 - lamb) + lamb) / (2 * kappa))) - np.pi
    phi = 2 * np.pi * r2
    samplerad_fisher = dict()
    samplerad_fisher['tetas'] = list(the - np.pi)
    samplerad_fisher['phis'] = list(phi)
    samplerad_fisher['n'] = numsamp
    samplerad_fisher['type'] = 'rad'
    samplecart_fisher = polartocart(samplerad_fisher)
    sample = rotatesample(samplecart_fisher, alpha, 0)
    sample = rotatesample(sample, 0, beta)
    return sample


def watson(numsamp: int, lamb: float, mu: float, nu: float, kappa: float) -> dict:
    """
    Generate Watson distributed data on the unit sphere [1]_

    :param numsamp: Number of samples to generate
    :type numsamp: int
    :param lamb: Direction of cosines in the x-axis
    :type lamb: float
    :param mu: Direction of cosines in the y-axis
    :type mu: float
    :param nu: Direction of cosines in the z-axis
    :type nu: float
    :param kappa:
    :return: Data dictionary of type 'cart' containing numsamp Watson distributed data
    :rtype: dict

    [1] Best, D. J. & Fisher, N. I. (1986). Goodness-of-fit and discordancy tests for samples from the Watson distribution on the sphere. Austral. J. Statist. 28, 13-31.
    """
    sample = dict()
    sample['n'] = numsamp
    sample['type'] = 'rad'
    sample['tetas'] = []
    sample['phis'] = []
    rng = default_rng()
    sampleall = dict()
    try:
        assert kappa != 0
    except ValueError:
        raise ValueError('kappa cannot be zero!')

    if kappa > 0:
        for ind in range(numsamp):
            C = 1 / (np.exp(kappa) - 1)
            U = rng.uniform(0, 1, 1)
            V = rng.uniform(0, 1, 1)
            S = 1 / kappa * np.log(U / C + 1)
            while V > np.exp(kappa * S ** 2 - kappa * S):
                U = rng.uniform(0, 1, 1)
                V = rng.uniform(0, 1, 1)
                S = 1 / kappa * np.log(U / C + 1)
            the = np.arccos(S)
            phi = 2 * np.pi * rng.uniform(0, 1, 1)
            sample['tetas'].append(the)
            sample['phis'].append(phi)
        samplecart = polartocart(sample)
        alpha, beta = cart2sph(np.array([lamb, mu, nu]))
        samprot = rotatesample(samplecart, alpha, beta)
        samprot = rotatesample(samprot, 0, beta)
        sampleneg = negatesample(samprot)
        sampleall = poolsamples([samprot, sampleneg], 'cart')
    elif kappa < 0:
        for ind in range(numsamp):
            c1 = np.sqrt(np.abs(kappa))
            c2 = np.arctan(c1)
            U = rng.uniform(0, 1, 1)
            V = rng.uniform(0, 1, 1)
            S = (1 / c1) * np.tan(c2 * U)
            while V > (1 - kappa * S ** 2) * np.exp(kappa * S ** 2):
                U = rng.uniform(0, 1, 1)
                V = rng.uniform(0, 1, 1)
                S = (1 / c1) * np.tan(c2 * U)
            the = np.arccos(S)
            phi = 2 * np.pi * rng.uniform(0, 1, 1)
            sample['tetas'].append(the)
            sample['phis'].append(phi)
        samplecart = polartocart(sample)
        alpha, beta = cart2sph(np.array([lamb, mu, nu]))
        samprot = rotatesample(samplecart, alpha, 0)
        samprot = rotatesample(samplecart, 0, beta)
        sampleneg = negatesample(samprot)
        sampleall = poolsamples([samprot, sampleneg], 'cart')

    return sampleall
