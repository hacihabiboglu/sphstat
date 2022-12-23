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
=======================================================
Functions for inferential statistics on a single sample
=======================================================

- :func:`isuniform` tests the null-hypothesis that the sample is drawn from a uniform spherical distribution
- :func:`testagainstmedian` tests the null-hypothesis that the sample has a given median direction
- :func:`rotationalsymmetry` tests rotational symmetry around the known true mean
- :func:`meanifsymmetric` calculates the mean direction under the assumption tha the sample comes from a symmetric distribution
- :func:`testagainstmean` tests the null-hypothesis that the sample has a given mean direction
- :func:`isaxisymmetric` tests the null hypothesis that the sample comes from a rotationally symmetric distribution
- :func:`isfisher` tests the null hypothesis that the sample comes from a Fisher distribution
- :func:`outliertest` identifies outliers in the sample
- :func:`fisherparams` calculates the parameters of the Fisher distribution that the sample is drawn from
- :func:`meantest` tests against a mean value
- :func:`kappatest` tests against a concentration parameter
- :func:`kentparams` calculates the parameters of the Kent distribution that the sample is drawn from
- :func:`kentmeanccone` calculates the confidence interval for the mean direction for a Kent distributed sample
- :func:`isfishervskent` tests the null hypothesis that the sample is drawn from a Fisher distribution as opposed to a Kent distribution
- :func:`bimodalparams` calculate the parameters of a bimodal (e.g. Wood) distribution

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import iv
from scipy.stats import chi2, norm, probplot, expon, uniform
from sympy import Symbol, nsolve, coth

from .descriptives import resultants, rotatesample, orientationmatrix, rotationmatrix, mediandir
from .utils import carttopolar, sph2cart, cart2sph, excludesample, deepcopy, maptofundamental


def isuniform(sample: dict, alpha: float = 0.05) -> dict:
    """
    Test if the sample comes from a uniform distribution as opposed to a unimodal distribution [1]_

    :param sample: Sample to be tested in 'cart' format
    :param alpha: Type-I error level  (e.g. 0.05)
    :returns: Dictionary containing the keys
        - teststat - Test statistic (float)
        - crange - Critical range (float)
        - testresult - Test result (bool)
    :rtype: dict

    [1] Diggle, P. J., Fisher, N. I. & Lee, A. J. (1985). A comparison of tests of uniformity for spherical data. Austral. J. Statist. 27, 53-59.
    """

    samplesize = sample['n']
    try:
        assert type(alpha) == float
        assert 0 < alpha < 1
        assert sample['type'] == 'cart'
        assert samplesize > 3
    except AssertionError:
        raise AssertionError('Please check the requirements for the arguments.')

    rs = resultants(sample)
    R2 = rs['Resultant Length']**2

    if samplesize < 10:
        try:
            assert alpha in [0.1, 0.05, 0.02, 0.01]
        except AssertionError:
            raise AssertionError('For small sample sizes alpha should be eithre 0.01, 0.02, 0.05, or 0.1')

    if samplesize < 10:
        assert alpha in {0.1, 0.05, 0.02, 0.01}
        # These critical values are from the book (Appendix A7)
        distsmall = dict()
        distsmall['0.1'] = [0, 0, 0, 2.85, 3.19, 3.50, 3.78, 4.05, 4.30]
        distsmall['0.05'] = [0, 0, 0, 3.10, 3.50, 3.85, 4.18, 4.48, 4.76]
        distsmall['0.02'] = [0, 0, 0, 3.35, 3.83, 4.24, 4.61, 4.96, 5.28]
        distsmall['0.01'] = [0, 0, 0, 3.49, 4.02, 4.48, 4.89, 5.26, 5.61]
        crange = distsmall[str(alpha)][samplesize-1]
        teststat = R2
    else:
        teststat = 3 * R2 / samplesize
        crange = chi2.ppf(1-alpha, 3)

    result = teststat < crange
    res = {'teststat': teststat, 'crange': crange, 'testresult': result}
    return res


def testagainstmedian(sample: dict, tmedi: list, alpha: float = 0.05) -> dict:
    """
    Test the null hypothesis that the population has the specified median direction [1]_

    :param sample: Sample to be tested in 'cart' format
    :type sample: dict
    :param tmedi: List containing the polar angles for the specified median
    :type tmedi: array-like
    :param alpha: Type-I error level (e.g. 0.05)
    :type alpha: float
    :return: Test measure (X2, float), critical value (cval, float), test result (bool), and significance (p-value, float)
    :rtype: dict

    [1] Fisher, N. I. (1985). Spherical medians. J.R. Statist. Soc. B47, 342-348.
    """
    # Cross-checked with Appendix_B1 data
    try:
        assert sample['type'] == 'cart'
    except AssertionError:
        raise AssertionError('Sample type should be cart.')
    try:
        assert sample['n'] >= 25
    except AssertionError:
        raise AssertionError('Only works for large sample sizes (i.e. n>=25)')

    medi, _, _, _ = mediandir(sample)

    samprot = rotatesample(sample, medi[0], medi[1], 0.)
    samppol = carttopolar(samprot)
    ph = np.array(samppol['phis'])

    sigma11 = 1 + np.sum(np.cos(2 * ph)) / samppol['n']
    sigma22 = 1 - np.sum(np.cos(2 * ph)) / samppol['n']
    sigma12 = np.sum(np.sin(2 * ph)) / samppol['n']

    Sigma = 0.5 * np.matrix([[sigma11, sigma12], [sigma12, sigma22]])

    samprotttest = rotatesample(sample, tmedi[0], tmedi[1])
    samprottesttpol = carttopolar(samprotttest)
    testph = np.array(samprottesttpol['phis'])
    U = 1 / np.sqrt(sample['n']) * np.array([np.sum(np.cos(testph)), np.sum(np.sin(testph))])
    X2 = np.transpose(U) @ np.linalg.inv(Sigma) @ U
    testval = X2.item(0)
    cval = chi2.ppf(1-alpha, 2)
    result = (testval < cval)
    sig = np.exp(-testval / 2)
    res = {'X2': testval, 'cval': cval, 'testresult': result, 'sig': sig}
    return res


def rotationalsymmetry(samplecart: dict, mdir: list, alpha: float = 0.05) -> dict:
    """
    Kuiper test for rotational symmetry around the known true mean

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param mdir: Mean direction in polar coordinates (theta, phi)
    :type mdir: list
    :param alpha: Type-I error level (e.g. 0.05)
    :type: float
    :return: Dictionary containing the following fields
        - Test measure ('Vnstar', float),
        - Critical value ('cval', float),
        - Test result ('res', bool)
    :rtype: dict
    """
    n = samplecart['n']
    try:
        assert n >= 9
    except AssertionError:
        raise AssertionError('Sample size cannot be less than 9.')
    try:
        assert alpha in [0.15, 0.1, 0.05, 0.01]
    except AssertionError:
        raise AssertionError('Test level (alpha) can be either 0.15, 0.1, 0.05 or 0.01')
    try:
        assert samplecart['type'] == 'cart'
    except AssertionError:
        raise AssertionError('Type of the sample must be cart.')

    samprot = rotatesample(samplecart, mdir[0], mdir[1])
    samppol = carttopolar(samprot)

    ph = samppol['phis']
    ph = np.mod(np.array(ph), 2 * np.pi)
    phnorm = ph / (2 * np.pi)
    phnormsorted = np.sort(phnorm)
    Dnplus = np.max(np.arange(1, n+1) / n - phnormsorted)
    Dnminus = np.max(phnormsorted - np.arange(0, n)/n)
    Vn = Dnplus + Dnminus
    Vnstar = Vn * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n))
    cvals = {0.15: 1.537, 0.1: 1.620, 0.05: 1.747, 0.01: 2.001}
    cval = cvals[alpha]
    testresult = Vnstar < cval
    res = {'Vnstar': Vnstar, 'cval': cval, 'res': testresult}
    return res


def meanifsymmetric(samplecart: dict, alpha: float = 0.05) -> tuple:
    """
    Estimation of the mean direction of a symmetric unimodal distribution [1]_

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param alpha: Type-I error level
    :type alpha: float
    :return: Dictionary containing the following fields
        - Spherical mean direction (theta: float, phi: float)
        - Spherical standard deviation (float)
        - Semi-vertical angle (float)
    :rtype: tuple

    [1] Fisher, N. I. & Lewis, T. (1983). Estimating the common mean direction of several circular or spherical distributions with differing dispersions. Biometrika 70, 333-341.
    """
    # Cross-checked with Appendix_B2 data
    rs = resultants(samplecart)
    xyzhat = rs['Directional Cosines']
    mdir = rs['Mean Direction']
    Rbar = rs['Mean Resultant Length']
    n = samplecart['n']
    dsum = 0
    for pt in samplecart['points']:
        dsum += np.sum(xyzhat * np.array(pt))**2
    d = 1 - dsum / n
    sigmahat = np.sqrt(d / (n * Rbar**2))  # Spherical standard error
    q = np.arcsin(np.sqrt(-np.log(alpha)) * sigmahat)
    return mdir, sigmahat, q


def testagainstmean(samplecart: dict, tmean: list, alpha: float = 0.05) -> dict:
    """
    Test for a specified mean direction of a symmetric distribution [1]_

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param tmean: Mean to be tested against (theta, phi)
    :type tmean: np.array
    :param alpha: Type-I error level
    :type alpha: float
    :return: Dictionary containing the following fields
        - Test measure ('hn', float)
        - Critical value to test against ('cval', float)
        - Test result ('testresult', bool)

    [1] Watson, G. S. (1983). Statistics on Spheres. University of Arkansas Lecture Notes in the Mathematical Sciences, Volume 6. New York: John Wiley.
    """
    _, sigmahat, _ = meanifsymmetric(samplecart, alpha)
    r = resultants(samplecart)
    dircosdata = r['Directional Cosines']
    dircostmean = sph2cart(tmean[0], tmean[1])
    hn = (1 - np.sum(dircostmean * dircosdata)**2) / sigmahat**2
    assert samplecart['n'] >= 25
    tres = (hn < -np.log(alpha))
    res = {'hn': hn, 'cval': -np.log(alpha), 'testresult': tres}
    return res


def isaxisymmetric(samplecart: dict, alpha: float = 0.05) -> dict:
    """
    Test for rotational symmetry about the mean direction

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param alpha: Type-I error level
    :type alpha: float
    :return: Dictionary containing the keys:
        - Pn: Test statistic (float)
        - cval: Critical value to test against (float)
        - pval: Actual p-value (float)
        - testresult: Test result (bool)
    :rtype: dict
    """
    assert samplecart['type'] == 'cart'
    n = samplecart['n']
    try:
        assert n >= 25
    except AssertionError:
        raise AssertionError('The number of samples cannot be less than 25 for a formal test of axisymmetry')
    Tmat = orientationmatrix(samplecart)
    paxis = Tmat['u'][:, 2]
    taubar = Tmat['taubar']
    Gamma = 0
    for pt in samplecart['points']:
        Gamma += np.dot(paxis, pt)**4 / n
    Pn = 2 * n * (taubar[1] - taubar[0])**2 / (1 - 2 * taubar[2] + Gamma)
    cval = -2 * np.log(alpha)
    pval = np.exp(-Pn / 2)

    if Pn > cval:
        result = False
    else:
        result = True
    res = {'Pn': Pn, 'cval': cval, 'pval': pval, 'testresult': result}
    return res


def isfisher(samplecart: dict, alpha: float = 0.05, plotflag: bool = False) -> dict:
    """
    Goodness-of-fit of the data with the Fisher model [1]_

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param alpha: Type-I error level
    :type alpha: float
    :param plotflag: Flag to plot the Q-Q plots
    :type plotflag: bool
    :return: Dictionary containing the results of three tests:
    - 'colatitute': Results of the colatitude test as a nested dictionary
        - 'stat': Test statistic (float)
        - 'crange': Critical range (float)
        - 'H0': Test result (bool)
    - 'longitude': Results of the longitude test as a nested dictionary
        - 'stat': Test statistic (float)
        - 'crange': Critical range (float)
        - 'H0': Test result (bool)
    - 'twovariable': Results of the two-variable test as a nested dictionary
        - 'stat': Test statistic (float)
        - 'crange': Critical range (float)
        - 'H0': Test result (bool)
    - 'H0': All three tests retain H0 then True, otherwise false
    - 'alpha': Type-I error level

    [1] Fisher, N. I. & Best, D. J. (1984). Goodness-of-fit tests for Fisher's distribution on the sphere. Austral. J. Statist. 26, 142-150.
    """

    res = dict()

    try:
        assert alpha in {0.1, 0.05, 0.01}
    except AssertionError:
        raise AssertionError('alpha can be 0.1, 0.05 or 0.01 only')
    mdir, sigmahat, q = meanifsymmetric(samplecart)
    samprot1 = rotatesample(samplecart, mdir[0], mdir[1])
    samprot2 = rotatesample(samplecart, 3 * np.pi / 2 - mdir[0], mdir[1] - np.pi)


    Xi_list = []
    Xi_prime_list = []
    Xi_dprime_list = []
    n = samplecart['n']
    for pt1 in samprot1['points']:
        th_prime, ph_prime = cart2sph(pt1)
        Xi_list.append(1 - np.cos(th_prime))
        Xi_prime_list.append(ph_prime / (2 * np.pi))

    for pt2 in samprot2['points']:
        th_dprime, ph_dprime = cart2sph(pt2)
        # Following is different from the book! if we subtract pi from ph_dprime than the plot will not pass through 0.
        Xi_dprime_list.append(ph_dprime * np.sqrt(np.sin(th_dprime)))

    Xi = np.sort(np.array(Xi_list))
    Xi_prime = np.sort(np.array(Xi_prime_list))
    Xi_dprime = np.sort(np.array(Xi_dprime_list))

    def F_colatitute(x, kappa):
        return 1 - np.exp(-kappa * x)

    def F_longitude(x):
        return x

    def F_twovariable(x):
        return norm.cdf(x)

    def criticals_a8(alphai):
        alphas = np.array([0.1, 0.05, 0.01])
        ind = np.argmin(np.abs(alphas - alphai))
        medni = [0.990, 1.094, 1.308]
        muvni = [1.138, 1.207, 1.347]
        mndni = [0.819, 0.895, 1.035]
        return medni[ind], muvni[ind], mndni[ind]

    medn, muvn, mndn = criticals_a8(alpha)

    def colatitudetest_fisher(Xii, plotflagi):
        rescolatitude = dict()
        if plotflagi:
            ax = plt.figure().gca()
            probplot(Xii, dist=expon, plot=ax)
            plt.show()
        # Calculate Dn for Kolmogorov-Smirnov test for Fisher
        nact = n
        ind = np.arange(nact) + 1
        kappahat = (nact - 1) / np.sum(Xii)
        Dn_plus = np.max(ind / nact - F_colatitute(Xii, kappahat))
        Dn_minus = np.max(F_colatitute(Xii, kappahat) - (ind-1) / nact)
        Dn = np.max([Dn_minus, Dn_plus])
        ME = (Dn - 0.2 / nact) * (np.sqrt(nact) + 0.26 + 0.5 / np.sqrt(nact))
        rescolatitude['stat'] = ME
        rescolatitude['crange'] = medn
        if ME > medn:
            rescolatitude['H0'] = False
        else:
            rescolatitude['H0'] = True
        return rescolatitude

    def longitudetest_fisher(Xi_primei, plotflagi):
        reslongitude = dict()
        if plotflagi:
            ax = plt.figure().gca()
            probplot(Xi_primei, dist=uniform, plot=ax)
            plt.show()
        nact = np.shape(Xi_primei)[0]
        ind = np.arange(nact) + 1
        Dn_plus = np.max(ind / nact - F_longitude(Xi_primei))
        Dn_minus = np.max(F_longitude(Xi_primei) - (ind - 1) / nact)
        Vn = Dn_minus + Dn_plus
        MU = Vn * (np.sqrt(nact) - 0.467 + 1.623 / np.sqrt(nact))
        reslongitude['stat'] = MU
        reslongitude['crange'] = muvn
        if MU > muvn:
            reslongitude['H0'] = False
        else:
            reslongitude['H0'] = True
        return reslongitude

    def twovariabletest_fisher(Xi_dprimei, plotflag):
        restwovariable = dict()
        if plotflag:
            ax = plt.figure().gca()
            probplot(Xi_dprimei, dist=norm, plot=ax)
            plt.show()

        s2 = np.mean(Xi_dprimei ** 2)
        xi = Xi_dprimei / np.sqrt(s2)
        xio = np.sort(xi)
        nact = np.shape(xio)[0]
        ind = np.arange(nact) + 1
        Dn_plus = np.max(ind / nact - F_twovariable(xio))
        Dn_minus = np.max(F_twovariable(xio) - (ind - 1) / nact)
        Dn = np.max([Dn_minus, Dn_plus])
        MN = Dn * (np.sqrt(nact) - 0.01 + 0.85 / np.sqrt(nact))
        restwovariable['stat'] = MN
        restwovariable['crange'] = mndn
        if MN > mndn:
            restwovariable['H0'] = False
        else:
            restwovariable['H0'] = True
        return restwovariable

    res['colatitute'] = colatitudetest_fisher(Xi, plotflag)
    res['longitude'] = longitudetest_fisher(Xi_prime, plotflag)
    res['twovariable'] = twovariabletest_fisher(Xi_dprime, plotflag)
    res['H0'] = res['colatitute']['H0'] & res['longitude']['H0'] & res['twovariable']['H0']
    res['alpha'] = alpha

    return res


def outliertest(samplecart: dict, alpha: float = 0.05) -> tuple:
    """
    Outlier test for discordancy [1]_

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param alpha: Type-I error level
    :type alpha: float
    :return:
        - A new sample with outliers eliminated
        - Index of the outliers in the original sample
    :rtype: tuple

    [1] Fisher, N. I., Lewis, T. & Willcox, M. E. (1981). Tests of discordancy for samples from Fisher's distribution on the sphere. Appl. Statist. 30, 230-237.
    """
    rs = resultants(samplecart)
    n = samplecart['n']
    Rn = rs['Resultant Length']
    En = []
    for ind in range(samplecart['n']):
        ssamp = excludesample(samplecart, ind)
        Rn_1 = resultants(ssamp)['Resultant Length']
        En.append((n-2) * (1 + Rn_1 - Rn) / (n - 1 - Rn_1))

    def criticalval(xi, n):
        if xi >= (n - 2):
            pi = n * ((n - 2) / (n - 2 + xi)) ** (n - 2)
        else:
            pi = n * ((n - 2) / (n - 2 + xi)) ** (n - 2) - (n - 1) * ((n - 2 - xi) / (n - 2 + xi)) ** (n - 2)
        return pi

    outlierinds = []
    for ind in range(len(En)):
        x = En[ind]
        p = criticalval(x, n)
        if p < alpha:
            outlierinds.append(ind)

    scpy = deepcopy(samplecart)
    for ind in outlierinds:
        scpy = excludesample(scpy, ind)

    return scpy, outlierinds


def fisherparams(samplecart: dict, alpha: float=0.05) -> dict:
    """
    Parameter estimation for the Fisher distribution [1]_

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param alpha: Calculate (1-alpha)% CI for kappa
    :type alpha: float
    :return: Dictionary with the keys...
    - mdir: Mean direction (theta, phi) (tuple)
    - kappa: Concentration parameter (float)
    - thetaalpha: Semivertical angle (float)
    - cikappa: (kappalow, kappahigh) is the (1-alpha)% CI for kappa (tuple)
    :rtype: dict

    [1] Watson, G. S. & Williams, E. J. (1956). On the construction of significance tests on the circle and the sphere. Biometrika 43, 344-352.
    [2] Watson, G. S. (1956). Analysis of dispersion on a sphere. Mon. Not. R. Astr. Soc. Geophys. Suppl. 7, 153-159.
    """
    rs = resultants(samplecart)
    R = rs['Resultant Length']
    n = samplecart['n']
    kap = Symbol('kap')
    f1 = coth(kap) - 1 / kap - R/n
    kappa = nsolve(f1, kap, 1)
    res = meanifsymmetric(samplecart)
    mdir = maptofundamental(res[0])
    thetaalpha = None
    if kappa >= 5:
        thetaalpha = np.arccos(1 - ((n - R) / R) * ((1 / alpha)**(1 / (n - 1)) - 1))
    elif n >= 30:
        thetaalpha = np.arccos(1 + np.log(alpha) / (kappa * R))
    else:
        Warning('The library currently supports appoximate CIs only for sample sizes of n>=30 or a concentration parameters kappa>=5')
        thetaalpha = None

    kappahigh = 0.5 * chi2.ppf(1 - 0.5 * alpha, 2 * n - 2) / (n - R)
    kappalow = 0.5 * chi2.ppf(0.5 * alpha, 2 * n - 2) / (n - R)
    cikappa = (kappalow, kappahigh)
    res = {'mdir': mdir, 'kappa': kappa, 'thetalpha': thetaalpha, 'cikappa': cikappa}
    return res


def meantest(samplecart: dict, mdir0: tuple | list, alpha: float =0.05) -> dict:
    """
    Test for a specified mean direction

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param mdir0: Mean direction to test against (H0: mdir = mdir0)
    :type mdir0: tuple | list
    :param alpha: Type-I error level
    :type alpha: float
    :return: Dictionary including..
        - R: Test statistic (float)
        - Ralpha: Critical value (float)
        - testresult: Test result (bool)
    :rtype: dict
    """
    ''' Test the null hypothesis that mu_Sample = mu_0'''
    # Note: mdir0 is given in polar (theta, phi) coordinates
    assert type(mdir0) == tuple or list
    dcos0 = sph2cart(mdir0[0], mdir0[1])
    n = samplecart['n']
    rsd = fisherparams(samplecart)
    kappahat = rsd['kappa']
    rs = resultants(samplecart)
    dcoshat = rs['Directional Cosines']
    R = rs['Resultant Length']
    C0 = np.dot(dcos0, dcoshat)
    Rz = C0 * R
    try:
        assert kappahat >= 5
    except AssertionError:
        raise ValueError('kappahat=' + str(kappahat) + ': Currently only sample concentration parameters of at least 5 (kappa>=5) are supported')

    Ralpha = n - (n - Rz) * alpha ** (1 / (n - 1))
    result = R <= Ralpha
    res = {'R': R, 'Ralpha': Ralpha, 'result': result}
    return res  # i.e. reject H0 that the mean of the data is mdir0 if R > Ralpha


def kappatest(samplecart, kappa0, alpha=0.05, testtype='!='):
    """
    Test for specified concentration parameter with unknown population mean

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param kappa0: Concentration parameter to test against (H0: kappa=kappa0)
    :type kappa0: float
    :param alpha: Type-I error level
    :type alpha: float
    :param testtype: Either on of !=, > or < indicating the sidedness of the test
    :return: Dictionary with the keys:
        - 'R': Test statistics
        - 'cvaltup': Critical value for the test
        - 'testresult': Test result
    """

    try:
        assert testtype in {'>', '<', '!='}
    except AssertionError:
        raise ValueError('Valid flags for testtype can be > (left-sided), < (right-sided), or != (two-sided)')
    try:
        assert kappa0 >= 5
    except AssertionError:
        raise ValueError('The library currently supports only kappa0 >= 5')

    rs = resultants(samplecart)
    R = rs['Resultant Length']
    n = samplecart['n']
    if testtype == '>':
        cvaldn = n - chi2.ppf(1 - alpha, 2 * n - 2) / (2 * kappa0)
        cvalup = None
        res = R > cvaldn
    elif testtype == '<':
        cvalup = n - chi2.ppf(alpha, 2 * n - 2) / (2 * kappa0)
        cvaldn = None
        res = R < cvalup
    else:
        cvaldn = n - chi2.ppf(1 - alpha / 2, 2 * n - 2) / (2 * kappa0)
        cvalup = n - chi2.ppf(alpha / 2, 2 * n - 2) / (2 * kappa0)
        res = (cvaldn < R < cvalup)
    cvaltup = (cvaldn, cvalup)
    rs = {'R': R, 'cvaltup': cvaltup, 'testresult': res}
    return rs


def kentparams(samplecart):
    """
    Estimation of the parameters of the Kent distribution [1]_

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :return:
        - 'axes': Axes of the distribution (axes[0] is the mean direction)
        - 'kappahat': Concentration parameter of the distribution (float)
        - 'betahat': Ovalness parameter (float)
    :rtype:tuple

    [1] Kent, J. T. (1982). The Fisher-Bingham distribution on the sphere. J.R. Statist. Soc. B 44, 71-80.
    """
    '''Calculate the parameters of a Kent distribution modelling the sample'''
    # Step 1
    rs = resultants(samplecart)
    Rbar = rs['Mean Resultant Length']
    mdir = rs['Mean Direction']
    n = samplecart['n']
    Tmat = orientationmatrix(samplecart)
    T = Tmat['T']
    S = T / n

    # Step 2
    A = rotationmatrix(mdir[0], mdir[1], psi0=0)
    H = A.T
    B = H.T @ S @ H
    b12 = B[0, 1]
    b11 = B[0, 0]
    b22 = B[1, 1]
    psihat = 0.5 * np.arctan2(2 * b12, (b11 - b22))

    # Step 3
    K = np.zeros((3, 3))
    K[0, 0] = np.cos(psihat)
    K[0, 1] = -np.sin(psihat)
    K[1, 0] = np.sin(psihat)
    K[1, 1] = np.cos(psihat)
    K[2, 2] = 1

    Ghat = H @ K
    V = Ghat.T @ S @ Ghat
    v11 = V[0, 0]
    v22 = V[1, 1]
    Q = v11 - v22
    try:
        assert Q > 0
    except AssertionError:
        raise ValueError('The data does not satisfy Q>0. Possible reason: kappa is not large enough!')

    xi1 = Ghat[:, 0].reshape((3,))
    xi2 = Ghat[:, 1].reshape((3,))
    xi3 = Ghat[:, 2].reshape((3,))

    axes = [xi3, xi2, xi1]

    # Step 4
    kappahat = 1 / (2 - 2 * Rbar - Q) + 1 / (2 - 2 * Rbar + Q)
    betahat = 0.5 * (1 / (2 - 2 * Rbar - Q) - 1 / (2 - 2 * Rbar + Q))

    if kappahat < 5:
        print('The sample concentration parameter, kappahat < 5. The results might be invalid.')

    return axes, kappahat, betahat


def kentmeanccone(samplecart: dict, alpha: float = 0.05) -> tuple:
    """
    Elliptical confidence cone for the mean direction [1]_

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param alpha: (1-alpha)% CI is calculated
    :type alpha: float
    :return:
        - cconept: 360 points on the (1-alpha)% cone of confidence (list)
        - ths1: Major semi-axis (in radians)
        - ths2: Minor semi-axis (in radians)
    :rtype: tuple

    [1] Kent, J. T. (1982). The Fisher-Bingham distribution on the sphere. J.R. Statist. Soc. B 44, 71-80.

    """
    '''Calculate the parameters of a Kent distribution modelling the sample'''
    # Step 1
    rs = resultants(samplecart)
    # Rbar = rs['Mean Resultant Length']
    mdir = rs['Mean Direction']
    n = samplecart['n']
    Tmat = orientationmatrix(samplecart)
    T = Tmat['T']
    S = T / n

    # Step 2
    A = rotationmatrix(mdir[0], mdir[1], psi0=0)
    H = A.T
    B = H.T @ S @ H
    b12 = B[0, 1]
    b11 = B[0, 0]
    b22 = B[1, 1]
    psihat = 0.5 * np.arctan2(2 * b12, (b11 - b22))

    # Step 3
    K = np.zeros((3, 3))
    K[0, 0] = np.cos(psihat)
    K[0, 1] = -np.sin(psihat)
    K[1, 0] = np.sin(psihat)
    K[1, 1] = np.cos(psihat)
    K[2, 2] = 1

    Ghat = H @ K

    # Step 1 for sample mean CI: (5.52) onwards
    mubar = 0
    sigma2_1bar = 0
    sigma2_2bar = 0
    for pt in samplecart['points']:
        ptn = Ghat @ pt
        mubar += ptn[2] / n
        sigma2_1bar += ptn[0]**2 / n
        sigma2_2bar += ptn[1]**2 / n

    # Step 2 for sample mean CI
    g = -2 * np.log(alpha)/(n * mubar**2)
    s1 = np.sqrt(sigma2_1bar * g)
    s2 = np.sqrt(sigma2_2bar * g)
    ths1 = np.arcsin(s1)
    ths2 = np.arcsin(s2)

    # Step 3 for sample mean CI
    u0 = np.linspace(-s1, s1, 360)
    v0 = np.sqrt(s2 * (1 - (u0 / s1)**2))
    w0 = np.sqrt(1 - u0**2 - v0**2)

    x0, x0p = u0, u0
    y0, y0p = v0, -v0
    z0, z0p = w0, w0  # Go on from (5.56) onwards

    cconept = []
    for ind in range(360):
        pt1 = np.array([x0[ind], y0[ind], z0[ind]])
        pt2 = np.array([x0p[ind], y0p[ind], z0p[ind]])
        cconept.append(Ghat @ pt1)
        cconept.append(Ghat @ pt2)

    return cconept, ths1, ths2


def isfishervskent(samplecart: dict, alpha: float = 0.05) -> dict:
    """
    Test of whether a sample comes from a Fisher distribution, against the alternative that it comes from a Kent distribution

    :param samplecart: Sample to be tested in cart format
    :type samplecart: dict
    :param alpha: Type-I error level
    :type alpha: float
    :return: Dictionary containing the keys:
        - K: Test statistic (float)
        - cval: Critical value (float)
        - p: p-value (float)
        - testresult: Test result (bool)
    :rtype: dict
    """
    '''Calculate the parameters of a Kent distribution modelling the sample'''
    # Step 1
    rs = resultants(samplecart)
    Rbar = rs['Mean Resultant Length']
    mdir = rs['Mean Direction']
    n = samplecart['n']

    try:
        assert n >= 30
    except AssertionError:
        raise ValueError('The test can only be run with sample sizes greater than 30.')

    Tmat = orientationmatrix(samplecart)
    T = Tmat['T']
    S = T / n

    # Step 2
    A = rotationmatrix(mdir[0], mdir[1], psi0=0)
    H = A.T
    B = H.T @ S @ H
    b12 = B[0, 1]
    b11 = B[0, 0]
    b22 = B[1, 1]
    psihat = 0.5 * np.arctan2(2 * b12, (b11 - b22))

    # Step 3
    K = np.zeros((3, 3))
    K[0, 0] = np.cos(psihat)
    K[0, 1] = -np.sin(psihat)
    K[1, 0] = np.sin(psihat)
    K[1, 1] = np.cos(psihat)
    K[2, 2] = 1

    Ghat = H @ K
    V = Ghat.T @ S @ Ghat
    v11 = V[0, 0]
    v22 = V[1, 1]
    Q = v11 - v22
    try:
        assert Q > 0
    except AssertionError:
        raise AssertionError('The data does not satisfy Q>0. Possible reason: kappa is not large enough!')

    # Step 4
    kappahat = 1 / (2 - 2 * Rbar - Q) + 1 / (2 - 2 * Rbar + Q)
    G = iv(0.5, kappahat) / iv(2.5, kappahat)

    K = n * (0.5 * kappahat)**2 * G * Q**2
    cval = -2 * np.log(alpha)
    result = (K < cval)  # Reject if res == False
    p = np.exp(-0.5 * K)
    res = {'K': K, 'cval': cval, 'p': p, 'testresult': result}
    return res

# 2. Analysis of a sample of unit vectors from a multimodal distribution


def bimodalparams(samplecart: dict, modesfar: bool = False) -> dict:
    """
    Model parameter estimation for a bimodal distribution [1]_

    :param samplecart: Sample to be tested in 'cart' format
    :type samplecart: dict
    :param modesfar: Flag for indicating whether the two modes of the data are far.
    :param modesfar: bool
    :return: Model parameters for the Wood distribution
    :rtype: tuple

    [1] Wood, A. (1982). A bimodal distribution on the sphere. Appl. Statist. 31, 52-58.
    """
    # Checked with data and results in the original paper by Wood (1982) Data in the book has a minus typo for latitude
    def s2fun(gamdel, samplecarti):
        gammai = gamdel[0]
        deltai = gamdel[1]
        mu1i = np.array([np.cos(gammai) * np.cos(deltai), np.cos(gammai) * np.sin(deltai), -np.sin(gammai)])
        mu2i = np.array([-np.sin(deltai), np.cos(deltai), 0])
        mu3i = np.array([np.sin(gammai) * np.cos(deltai), np.sin(gammai) * np.sin(deltai), np.cos(gammai)])
        U, V, W = 0, 0, 0
        for pti in samplecarti['points']:
            U += np.dot(pti, mu3i)
            V += (np.dot(pti, mu1i) ** 2 - np.dot(pti, mu2i) ** 2) / np.sqrt(1 - np.dot(pti, mu3i) ** 2)
            W += 2 * np.dot(pti, mu1i) * np.dot(pti, mu2i) / np.sqrt(1 - np.dot(pti, mu3i) ** 2)
        S2 = U**2 + V**2 + W**2
        return -S2

    def initialgammadelta(samplecarti, modesfari):
        r = resultants(samplecarti)
        if modesfari:
            Tmat = orientationmatrix(samplecarti)
            u = Tmat['u']
            u2 = u[:, 1]
            urrad1 = cart2sph(u2)
            urrad2 = cart2sph(-u2)
            urrad1 = maptofundamental(urrad1)
            urrad2 = maptofundamental(urrad2)
            res1 = minimize(s2fun, np.array(urrad1), samplecarti, method='Nelder-Mead', bounds=[(0, np.pi), (0, 2 * np.pi)])
            res2 = minimize(s2fun, np.array(urrad2), samplecarti, method='Nelder-Mead', bounds=[(0, np.pi), (0, 2 * np.pi)])
            S2_1 = s2fun(res1.x, samplecarti)
            S2_2 = s2fun(res2.x, samplecarti)
            if S2_1 < S2_2:
                initvec = urrad1
            else:
                initvec = urrad2
        else:
            initvec = r['Mean Direction']
        return initvec

    ivec = initialgammadelta(samplecart, modesfar)
    res = minimize(s2fun, ivec, samplecart, method='Nelder-Mead', bounds=[(0, np.pi), (0, 2 * np.pi)])
    fvec = res.x
    n = samplecart['n']
    gamma = fvec[0]
    delta = fvec[1]

    mu1 = np.array([np.cos(gamma) * np.cos(delta), np.cos(gamma) * np.sin(delta), -np.sin(gamma)])
    mu2 = np.array([-np.sin(delta), np.cos(delta), 0])
    mu3 = np.array([np.sin(gamma) * np.cos(delta), np.sin(gamma) * np.sin(delta), np.cos(gamma)])
    Un, Vn, Wn = 0, 0, 0
    for pt in samplecart['points']:
        Un += np.dot(pt, mu3)
        Vn += (np.dot(pt, mu1) ** 2 - np.dot(pt, mu2) ** 2) / np.sqrt(1 - np.dot(pt, mu3) ** 2)
        Wn += 2. * np.dot(pt, mu1) * np.dot(pt, mu2) / np.sqrt(1 - np.dot(pt, mu3) ** 2)
    S2n = Un ** 2. + Vn ** 2. + Wn ** 2.
    Rstar = np.sqrt(S2n)

    alphahat = np.arctan(np.sqrt(Vn**2 + Wn**2) / Un)
    betahat = np.arctan2(Wn, Vn)

    kap = Symbol('kap')
    f1 = coth(kap) - 1 / kap - Rstar / n
    kappa = nsolve(f1, kap, 1)

    try:
        assert alphahat > 0.1
    except AssertionError:
        raise ValueError('The distribution is not bimodal (i.e. Wood)!')

    alpha1, beta1 = alphahat, 0.5 * betahat
    alpha2, beta2 = alphahat, 0.5 * betahat + np.pi

    A = np.zeros((3, 3))
    A[:, 0] = mu1
    A[:, 1] = mu2
    A[:, 2] = mu3

    abvec1 = sph2cart(alpha1, beta1)
    abvec2 = sph2cart(alpha2, beta2)
    abvecstar1 = A @ abvec1
    abvecstar2 = A @ abvec2

    astar1, bstar1 = cart2sph(abvecstar1)
    astar2, bstar2 = cart2sph(abvecstar2)
    res = {'gamma': gamma, 'delta': delta, 'kappa': kappa, 'alpha': alphahat,
           'beta': betahat, 'mode1': {'alpha': astar1, 'beta': bstar1}, 'mode2': {'alpha': astar2, 'beta': bstar2}}
    return res
