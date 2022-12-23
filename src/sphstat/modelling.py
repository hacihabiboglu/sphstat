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
==================================================================
Functions for correlation, regression, temporal association
==================================================================

- :func:`xcorrrandomsamples` calculates the cross-correlation between two samples
- :func:`samplecorrelation` calculates the sample cross-correlation
- :func:`jackknife_corrci` calculates a jeckknife estimate of the CI for the correlation coefficient
- :func:`xcorrsamplevariable` calculates the correlation of a sample with a variable
- :func:`regresscircular` calculates a regression model for a circular variable
- :func:`isnotseriallyassociated` tests the null hypothesis that the samples are independent as opposed to being temporally associated
"""

import numpy as np
from math import log, sqrt, sin, cos
from scipy.stats import norm, chi2
from scipy.linalg import sqrtm

from .descriptives import resultants
from .utils import randompermutations, excludesample
from .singlesample import isfisher


def xcorrrandomsamples(samplecart1: dict, samplecart2: dict, numperms: int, htype: str = '=', alpha: float = 0.05) -> dict:
    """
    Correlation coefficient of two random unit vectors and hypothesis test given two samples [1]_

    :param samplecart1: Sample 1 to be used in the computations in 'cart' form
    :type samplecart1: dict
    :param samplecart2: Sample 2 to be used in the computations in 'cart' form
    :type samplecart2: dict
    :param numperms: Number of permutations to be used in the permutation test
    :type numperms: int
    :param htype: Null hypothesis type, '!='  for H0: rho!=0 (no correlation), '>' for H0: rho > 0 (positive correlation), '<' for H0: rho < 0 (negative correlation)
    :param htype: str
    :param alpha: (1-alpha)% CI is calculated
    :type alpha: float
    :return: Dictionary containing:
        - rhohat: Correlation coefficient estimate (float)
        - std: Standard deviation from jackknife estimate (float)
        - cval': Critical values (tuple)
        - ci: (1-alpha)% CI for the correlation coefficient (tuple)
        - testresult: Hypothesis test result (bool)
    :rtype: dict

    [1] Fisher, N. I. & Lee, A. J. (1986). Correlation coefficients for random variables on a unit sphere or hypersphere.
    """
    try:
        assert samplecart1['type'] == 'cart'
        assert samplecart2['type'] == 'cart'
    except AssertionError:
        raise AssertionError('Type of both samples should be cart')

    try:
        assert samplecart1['n'] == samplecart2['n']
    except AssertionError:
        raise AssertionError('Samples should have the same size!')

    try:
        assert htype in ['=', '>', '<']
    except AssertionError:
        raise AssertionError('Hypothesis type can be !=, > or < only!')

    n = samplecart1['n']

    perms = randompermutations(n, numperms)

    rhovhat = samplecorrelation(samplecart1, samplecart2)
    rholist = []
    for perm in perms:
        rhoi = samplecorrelation(samplecart1, samplecart2, perm)
        rholist.append(rhoi)
    rhoia = np.array(rholist)

    s1f = isfisher(samplecart1, alpha)['H0']
    s2f = isfisher(samplecart2, alpha)['H0']

    fisherflag = s1f and s2f

    if fisherflag:
        cval1 = -log(alpha) / n
        cval2 = log(alpha) / n
        cval3 = log(2 * alpha) / n
        cval4 = -log(2 * alpha) / n
    else:
        cval1 = np.percentile(rhoia, 1 - alpha / 2)
        cval2 = np.percentile(rhoia, alpha / 2)
        cval3 = np.percentile(rhoia, alpha)
        cval4 = np.percentile(rhoia, 1 - alpha)

    if htype == '=':
        testresult = (cval1 < rhovhat < cval2)
        cval = (cval1, cval2)
    elif htype == '<':
        testresult = (rhovhat < cval3)
        cval = (cval3, None)
    else:
        testresult = (rhovhat > cval4)
        cval = (None, cval4)

    _, st, ci = jackknife_corrci(samplecart1, samplecart2, alpha)
    res = {'rhohat': rhovhat, 'std': st, 'cval': cval, 'ci': ci, 'testresult': testresult}

    return res


def samplecorrelation(samplecart1: dict, samplecart2: dict, indlist: list = None) -> float:
    """
    Correlation of two samples

    :param samplecart1: Sample 1 to be used in the computations in 'cart' form
    :type samplecart1: dict
    :param samplecart2: Sample 2 to be used in the computations in 'cart' form
    :type samplecart2: dict
    :param indlist: Indices to use for the second sample (use only for the permutation test)
    :return: Sample cross correlation
    :rtype: float
    """
    try:
        assert samplecart1['type'] == 'cart'
        assert samplecart2['type'] == 'cart'
    except AssertionError:
        raise AssertionError('Type of both samples should be cart')

    try:
        assert samplecart1['n'] == samplecart2['n']
    except AssertionError:
        raise AssertionError('Samples should have the same size!')

    n = samplecart1['n']
    if indlist is None:
        indlist = list(range(n))

    Cxxs = np.zeros((3, 3))
    Cxx = np.zeros((3, 3))
    Cxsxs = np.zeros((3, 3))

    for ind in range(n):
        Xi = samplecart1['points'][ind].reshape(1, 3)
        Xistar = samplecart2['points'][indlist[ind]].reshape(1, 3)
        Cxxs += Xi.T @ Xistar
        Cxx += Xi.T @ Xi
        Cxsxs += Xistar.T @ Xistar

    Sxx = np.linalg.det(Cxx)
    Sxxs = np.linalg.det(Cxxs)
    Sxsxs = np.linalg.det(Cxsxs)
    rhov = Sxxs / np.sqrt(Sxx * Sxsxs)
    return rhov


def jackknife_corrci(samplecart1: dict, samplecart2: dict, alpha: float = 0.05) -> tuple:
    """
    Jackknife method for calculating an approximate confidence interval for correlation coefficient

    :param samplecart1: Sample 1 to be used in the computations in 'cart' form
    :type samplecart1: dict
    :param samplecart2: Sample 2 to be used in the computations  in 'cart' form
    :type samplecart2: dict
    :param alpha: (1-alpha)% CI is calculated
    :type alpha: float
    :return:
        - psijhat: Unbiased estimate using the jackknife method (float)
        - ci: (1-alpha)% confidence interval (tuple)
    :rtype: tuple
    """
    psilist = []
    n = samplecart1['n']
    psihat = samplecorrelation(samplecart1, samplecart2)

    for ind in range(n):
        samplered1 = excludesample(samplecart1, ind)
        samplered2 = excludesample(samplecart2, ind)
        psii = samplecorrelation(samplered1, samplered2)

        psiihat = n * psihat - (n - 1) * psii
        psilist.append(psiihat)
    psiihata = np.array(psilist)

    psijhat = psihat
    sj2 = 1 / (n * (n - 1)) * np.sum((psijhat - psiihata) ** 2)
    za2 = norm.ppf(1 - alpha / 2)
    ci = (psijhat - sqrt(sj2) * za2, psijhat + sqrt(sj2) * za2)
    return psijhat, sqrt(sj2), ci


def xcorrsamplevariable(samplecart: dict, variable: list, alpha: float = 0.05):
    """
    Correlation between a random unit vector or axis and another random variable
    Tests the null hypothesis (H0) that the spherical sample and the variable are
    uncorrelated [1]_

    :param samplecart: Sample to be used in the computations in 'cart' form
    :type samplecart: dict
    :param variable: List of floats or 1-D numpy arrays representing a variable associated with the points in the sample
    :type variable: list
    :param alpha: Type-I error level to be used in the test
    :type alpha: float
    :return: Dictionary containing...
        - rhohatg: Correlation coefficient (float)
        - teststat: Test statistics (float)
        - cval: Critical value used in the test (float)
        - testresult: Result of the hypothesis test (bool)

    [1] Jupp, P. E. & Mardia, K. V. (1980). A general correlation coefficient for directional data and related regression problems. Biometrika 67, 163-173.
    """
    try:
        n = samplecart['n']
        assert len(variable) == n
    except AssertionError:
        raise AssertionError('The sample size and the variable size should match!')

    try:
        assert n > 25
    except AssertionError:
        raise ValueError('The sample size should be at least 25!')

    arrayflag = None
    p = 0.

    try:
        shp = np.shape(variable[0])
        typ = type(variable[0])
        assert typ in (float, np.ndarray)
        for ind in range(1, n):
            assert type(variable[ind]) == typ
            if typ == np.ndarray:
                assert np.shape(variable[ind]) == shp
                p = shp[0]
                arrayflag = True
            else:
                arrayflag = False
    except AssertionError:
        raise AssertionError('The type and dimensions of each value in the variable should be the same')

    if arrayflag:
        Xbar = np.zeros((3, 1))
        Ybar = np.zeros((p, 1))
        for ind in range(n):
            Xbar += samplecart['points'][ind].reshape(3, 1)
            Ybar += variable[ind].reshape(p, 1)
        Xbar /= n
        Ybar /= n

        sigma11 = np.zeros((3, 3))
        sigma12 = np.zeros((3, p))
        sigma22 = np.zeros((p, p))

        for ind in range(n):
            Xi = samplecart['points'][ind].reshape(3, 1)
            Yi = variable[ind].reshape(p, 1)
            sigma11 += (Xi - Xbar) @ (Xi - Xbar).T
            sigma12 += (Xi - Xbar) @ (Yi - Ybar).T
            sigma22 += (Yi - Ybar) @ (Yi - Ybar).T

        sigmahat = np.linalg.inv(sigma11) @ sigma12 @ np.linalg.inv(sigma22) @ sigma12.T
        q = min(3, p)
        rhohatg = np.sum(np.diag(sigmahat)) / q
    else:
        Xbar = np.zeros((3, 1))
        Ybar = 0.
        for ind in range(n):
            Xbar += samplecart['points'][ind].reshape(3, 1)
            Ybar += variable[ind]
        Xbar /= n
        Ybar /= n

        sigma11 = np.zeros((3, 3))
        sigma12 = np.zeros((3, 1))
        sigma22 = 0.

        for ind in range(n):
            Xi = samplecart['points'][ind].reshape(3, 1)
            Yi = variable[ind]
            sigma11 += (Xi - Xbar) @ (Xi - Xbar).T
            sigma12 += (Xi - Xbar) * (Yi - Ybar)
            sigma22 += (Yi - Ybar) * (Yi - Ybar)

        sigmahat = (np.linalg.inv(sigma11) @ sigma12 @ sigma12.T) * 1 / sigma22
        rhohatg = np.sum(np.diag(sigmahat))
        p, q = 1, 1
    cval = chi2.ppf(1 - alpha, 3 * p)
    teststat = q * n * rhohatg
    result = teststat < cval
    res = {'rhohatg': rhohatg, 'teststat': teststat, 'cval': cval, 'testresult': result}
    return res


def regresscircular(samplecart: dict, thetas: list, alpha0: float = np.pi/2, thr: float = 1e-2) -> tuple:
    """
    Regression of a random unit vector on a circular variable [1]_

    :param samplecart: Sample to be used in the computations in 'cart' form
    :type samplecart: dict
    :param thetas: List containing a circular variable
    :type thetas: list
    :param alpha0: Initial angle for the iterations
    :type alpha0: float
    :param thr: Stopping condition for the iterations (Defaults to 1e-15)
    :type thr: float
    :return:
        - Uhat: Regression matrix (np.array)
        - w: np.array([0, 0, 1])
        - alphahat: The angle to be used in the regression model
        - fitmodel: Function to predict a value for a given variable [Callable]
    :rtype: tuple

    [1] Jupp, P. E. & Mardia, K. V. (1980). A general correlation coefficient for directional data and related regression problems. Biometrika 67, 163-173.
    """
    r = resultants(samplecart)

    try:
        n = samplecart['n']
        assert len(thetas) == n
    except AssertionError:
        raise AssertionError('The circular parameters have to be the same length as the sample size!')

    Xbar = r['Resultant Vector'] / n
    Xbar = np.reshape(Xbar, (3, 1))
    w = np.array([0., 0., 1.]).reshape(3, 1)
    Swx = w @ Xbar.T
    vilist = []

    Svx = np.zeros((3, 3))
    for ind in range(n):
        vi = np.array([cos(thetas[ind]), sin(thetas[ind]), 0]).reshape(3, 1)
        vilist.append(vi)
        Svx += vi @ samplecart['points'][ind].reshape(1, 3)

    Svx /= n

    diff = np.Inf
    U1 = None
    while diff > thr:
        B1 = Svx * sin(alpha0) + Swx * cos(alpha0)
        U1 = np.linalg.inv(sqrtm(B1.T @ B1)) @ B1.T
        alpha1 = np.arctan(np.sum(np.diag(Svx @ U1.T)) / (Xbar.T @ U1 @ w))
        diff = np.abs(alpha1 - alpha0)
        alpha0 = alpha1

    Uhat, alphahat = np.real(U1), np.real(alpha0)

    def fitmodel(theta: float) -> np.array:
        v = np.array([cos(theta), sin(theta), 0]).reshape(3, 1)
        xhat = Uhat @ (v * sin(alphahat) + w * cos(alphahat))
        return xhat

    return Uhat, w, alphahat, fitmodel


def isnotseriallyassociated(samplecart: dict, alpha: float = 0.05):
    """
    Test the null hypothesis that the observations are independent as opposed to having serial association [1]_

    :param samplecart: Sample to be used in the computations in 'cart' form
    :type samplecart: dict
    :param alpha: Type-I error level to be used in the test
    :type alpha: float
    :return: Dictionary containing the keys:
        - Sstar: The test statistic (float)
        - cval: Critical value (float)
        - testresult: Test result (bool), False if the observations are serially associated
    :rtype: dict

    [1] Watson, G. S. & Beran, R. J. (1967). Testing a sequence of unit vectors for randomness. J. Geophys. Res. 72, 5655-5659.
    """

    try:
        n = samplecart['n']
        assert n >= 25
    except AssertionError:
        raise AssertionError("Sample size has to be at least 25!")

    S = 0

    for ind in range(n-1):
        S += np.sum(samplecart['points'][ind] * samplecart['points'][ind + 1])

    r = resultants(samplecart)

    X = r['Resultant Vector']
    R = r['Resultant Length']

    T = np.zeros((3, 3))
    for pt in samplecart['points']:
        ptt = pt.reshape((3, 1))
        T += ptt @ ptt.T

    S1 = np.sum(np.diag(T @ T))
    S2 = X.reshape((3, 1)).T @ T @ X

    SE = R**2 / n - 1
    SV = S1 / n - 2 * S2[0] / (n * (n - 1)) + R**4 / (n**2 * (n - 1)) + 2 * SE / (n - 1) - 2 + n / (n - 1)

    Sstar = (S - SE) / np.sqrt(SV)
    cval = norm.ppf(1 - alpha)
    result = Sstar < cval
    res = {'Sstar': Sstar, 'cval': cval, 'testresult': result}
    return res
