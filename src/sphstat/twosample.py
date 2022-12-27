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
===========================================================
Functions for inferential statistics on two or more samples
===========================================================

Functions for inferential statistics on two or more samples
-----------------------------------------------------------

- :func:`iscommonmedian` tests the null hypothesis that two samples have a common median
- :func:`pooledmedian` calculates the pooled median
- :func:`iscommonmean` tests the null hypothesis that two samples have a common population mean
- :func:`pooledmean` calculates the pooled mean
- :func:`isfishercommonmean` tests the null hypothesis that two Fisherian samples have a common population mean
- :func:`fishercommonmean` calculates the common population mean for two Fisherian samples
- :func:`isfishercommonkappa` tests the null hypothesis that two Fisherian samples have the same population concentration parameter
- :func:`fishercommonkappa` calculates the common population concentration parameter for two Fisherian samples

Utility functions
------------------

- :func:`a20`, :func:`a21`, :func:`a23`, :func:`bilinearinterp`, :func:`linearinterp` are used to read and interpolate critical values
- :func:`errors` calculates the non-negative angular error between two vectors

"""

import numpy as np
from scipy.stats import chi2
from scipy.stats import f as fish
import pandas as pd
from math import sqrt

from .descriptives import resultants, mediandir, rotatesample
from .singlesample import meanifsymmetric, fisherparams
from .utils import cart2sph, sph2cart, poolsamples, carttopolar, maptofundamental
import pkg_resources as pkg


def iscommonmedian(samplecartlist: list, similarflag: bool = True, alpha: float = 0.05) -> dict:
    """
    Test for a common median direction of two or more distributions [#]_

    :param samplecartlist: List containing individual samples top be tested in 'cart' format
    :type samplecartlist: list[dict]
    :param similarflag: Flag indicating similar distributions for all samples
    :type similarflag: bool
    :param alpha: Type-I error level
    :type alpha: float
    :return: Dictionary containing...
        - Z2: Test statistic [float]
        - cval: Critical value to test against [float]
        - result: Test result [bool]

    .. [#] Fisher, N. I. (1985). Spherical medians. J.R. Statist. Soc. B47, 342-348.
    """
    r = len(samplecartlist)
    if similarflag:
        gamdel, _ = pooledmedian(samplecartlist, similarflag)
        gammahat = gamdel[0]
        deltahat = gamdel[1]
        Sigmai = np.zeros((2, 2))
        Y2 = 0
        for si in samplecartlist:
            ni = si['n']
            try:
                assert ni >= 25
            except AssertionError:
                raise ValueError('At least one of the sample sizes is less than 25!')
            shat = rotatesample(si, gammahat, deltahat)
            shat = carttopolar(shat)
            gpdp, _, _, _ = mediandir(si)
            gammap, deltap = gpdp[0], gpdp[1]
            sp = rotatesample(si, gammap, deltap)
            sp = carttopolar(sp)
            Ui = ni ** -0.5 * np.array([np.sum(np.cos(np.array(shat['phis']))), np.sum(np.sin(np.array(shat['phis'])))])
            Sigmai[0, 0] = 1 + (1 / ni) * np.sum(np.cos(2 * np.array(sp['phis'])))
            Sigmai[1, 1] = 1 - (1 / ni) * np.sum(np.cos(2 * np.array(sp['phis'])))
            Sigmai[0, 1] = (1 / ni) * np.sum(np.sin(2 * np.array(sp['phis'])))
            Sigmai[1, 0] = Sigmai[0, 1]
            Y2 += Ui @ np.linalg.inv(0.5 * Sigmai) @ Ui
        cval = chi2.ppf(1 - alpha, 2 * r - 2)
        result = Y2 < cval
        testval = Y2
    else:
        si = poolsamples(samplecartlist, 'cart')
        N = si['n']
        gamdel, _ = pooledmedian(samplecartlist, similarflag)
        print(gamdel)
        gammahat = gamdel[0]
        deltahat = gamdel[1]
        srotlist1 = []
        srotlist2 = []
        Sigmai = np.zeros((2, 2))
        # Y2 = 0
        Uilist = []
        Sigma = np.zeros((2, 2))
        for si in samplecartlist:
            ni = si['n']
            try:
                assert ni >= 25
            except AssertionError:
                raise ValueError('At least one of the sample sizes is less than 25!')
            shat = rotatesample(si, gammahat, deltahat)
            shat = carttopolar(shat)
            srotlist1.append(shat)
            gpdp, _, _, _ = mediandir(si)
            gammap, deltap = gpdp[0], gpdp[1]
            sp = rotatesample(si, gammap, deltap)
            sp = carttopolar(sp)
            srotlist2.append(sp)
            Uilist.append(
                ni ** -0.5 * np.array([np.sum(np.cos(np.array(shat['phis']))), np.sum(np.sin(np.array(shat['phis'])))]))
            Sigmai[0, 0] = 1 + (1 / ni) * np.sum(np.cos(2 * np.array(sp['phis'])))
            Sigmai[1, 1] = 1 - (1 / ni) * np.sum(np.cos(2 * np.array(sp['phis'])))
            Sigmai[0, 1] = (1 / ni) * np.sum(np.sin(2 * np.array(sp['phis'])))
            Sigmai[1, 0] = Sigmai[0, 1]
            Sigma += (ni - 1) * Sigmai / (N - r) * 0.5
        Z2 = 0
        for Ui in Uilist:
            Z2 += Ui @ np.linalg.inv(Sigma) @ Ui
        cval = chi2.ppf(1 - alpha, 2 * r - 2)
        result = Z2 < cval
        testval = Z2
    res = {'Z2': testval, 'cval': cval, 'testresult': result}
    return res


def pooledmedian(samplecartlist: list, similarflag: bool = False) -> tuple:
    """
    Estimation of the common median direction of two or more unimodal distributions

    :param samplecartlist: List containing individual samples top be tested in 'cart' format
    :type samplecartlist: list[dict]
    :param similarflag: Flag indicating similar distributions for all samples
    :type similarflag: bool
    :return:
        - pooledmedi: Pooled median in polar coordinates [th, ph]
        - V: Matrix to be used for calculating the confidence cone [np.array]
    :rtype: tuple
    """

    if similarflag:
        Wtot = np.zeros((2, 1))
        V = np.zeros((2, 2))
        for ind in range(len(samplecartlist)):
            try:
                assert samplecartlist[ind]['n'] >= 25
            except AssertionError:
                raise AssertionError(
                    'At least one sample has less than 25 data points. Bootstrap methods currently not supported!')
            medi, _, _, Wi = mediandir(samplecartlist[ind], ciflag=True)
            xvec = sph2cart(medi[0], medi[1])
            Wtot += samplecartlist[ind]['n'] * (Wi @ xvec[:2].reshape(2, 1))
            V += samplecartlist[ind]['n'] * Wi

        pooledmedipre = np.linalg.inv(V) @ Wtot
        th = np.arcsin(np.sqrt(pooledmedipre[0] ** 2 + pooledmedipre[1] ** 2))
        ph = np.arctan2(pooledmedipre[1], pooledmedipre[0])
        pooledmedi = (th[0], ph[0])
        # Still need the confidence cone: (7.7) on p203 and onwards
    else:
        Vinv = np.zeros((2, 2))
        ps = poolsamples(samplecartlist)
        pooledmediprex = 0
        pooledmediprey = 0

        for ind in range(len(samplecartlist)):
            try:
                assert samplecartlist[ind]['n'] >= 25
            except AssertionError:
                raise AssertionError(
                    'At least one sample has less than 25 data points. Bootstrap methods currently not supported!')
            medi, _, _, Wi = mediandir(samplecartlist[ind], ciflag=True)
            xvec = sph2cart(medi[0], medi[1])
            pooledmediprex += xvec[0] * samplecartlist[ind]['n'] / ps['n']
            pooledmediprey += xvec[1] * samplecartlist[ind]['n'] / ps['n']
            Vinv += samplecartlist[ind]['n'] * np.linalg.inv(Wi) / ps['n'] ** 2
        V = np.linalg.inv(Vinv)
        pooledmediprez = np.sqrt(1 - pooledmediprex ** 2 - pooledmediprey ** 2)
        pooledmedipre = [pooledmediprex, pooledmediprey, pooledmediprez]
        pooledmedi = cart2sph(pooledmedipre)
    pooledmedi = maptofundamental(pooledmedi)
    return pooledmedi, V  # V is returned in case confidence cone would be needed; see (7.7)


def iscommonmean(samplecartlist: list, alpha: float = 0.05) -> dict:
    """
    Test of whether two or more axisymmetric distributions have a common mean [#]_

    :param samplecartlist: List containing individual samples top be tested in 'cart' format
    :type samplecartlist: list[dict]
    :param alpha: Type-I error level
    :type alpha: float
    :return: Dictionary containing:
        - Gr: Test statistic [float]
        - cval: Critical value to test against [float]
        - testresult: Test result [bool]

    .. [#] Watson, G. S. (1983a). Statistics on Spheres. University of Arkansas Lecture Notes in the Mathematical Sciences, Volume 6. New York: John Wiley.
    """

    sigmas = []
    muhats = []
    for sample in samplecartlist:
        mdiri, sigmai, _, = meanifsymmetric(sample, alpha)
        sigmas.append(sigmai)
        muhati = sph2cart(mdiri[0], mdiri[1])
        muhats.append(muhati)

    xhatstar, yhatstar, zhatstar = 0, 0, 0
    rhostar = 0
    r = len(samplecartlist)
    for ind in range(r):
        xhatstar += muhats[ind][0] / sigmas[ind] ** 2
        yhatstar += muhats[ind][1] / sigmas[ind] ** 2
        zhatstar += muhats[ind][2] / sigmas[ind] ** 2
        rhostar += 1 / sigmas[ind] ** 2

    Rstar = np.sqrt(xhatstar ** 2 + yhatstar ** 2 + zhatstar ** 2)
    Gr = 4 * (rhostar - Rstar)
    cval = chi2.ppf(1 - alpha, 2 * r - 2)
    result = (Gr <= cval)
    res = {'Gr': Gr, 'cval': cval, 'testresult': result}
    return res


def pooledmean(samplecartlist: list, alpha: float = 0.05) -> tuple:
    """
    Estimation of the common mean direction of two or more rotationally symmetric distributions

    :param samplecartlist: List containing individual samples top be tested in 'cart' format
    :type samplecartlist: list[dict]
    :param alpha: (1-alpha)% confidence cone is calculated
    :type alpha: float
    :return:
        - mdirpooled: Estimated pooled mean direction in radians [np.array]
        - sigmaw: Spherical standard error [float]
        - qw: Semi-vertical angle in radians [float]
    """
    sigmas = []
    muhats = []
    Rbar = []
    ns = []
    xi, yi, zi = [], [], []
    di = []
    for sample in samplecartlist:
        mdiri, sigmai, _, = meanifsymmetric(sample, alpha)
        rs = resultants(sample)
        Rbar.append(rs['Mean Resultant Length'])
        sigmas.append(sigmai)
        muhati = sph2cart(mdiri[0], mdiri[1])
        xi.append(muhati[0])
        yi.append(muhati[1])
        zi.append(muhati[2])
        muhats.append(muhati)
        ns.append(sample['n'])
        xyzhat = rs['Directional Cosines']
        dsum = 0
        for pt in sample['points']:
            dsum += np.sum(xyzhat * np.array(pt)) / sample['n']
        di.append(1 - dsum)

    nsa = np.array(ns)
    sigmaa = np.array(sigmas)
    Rbara = np.array(Rbar)
    nsig = np.sqrt(nsa) * sigmaa
    xia = np.array(xi)
    yia = np.array(yi)
    zia = np.array(zi)
    dia = np.array(di)
    flagtest = (2 * np.min(nsig) <= np.max(nsig))
    if flagtest:
        wi = nsa / np.sum(nsa)
    else:
        Cr = 1. / np.sum(nsa * Rbara / dia)
        wi = nsa * Rbara / dia * Cr

    xwhat = np.sum(wi * Rbara * xia)
    ywhat = np.sum(wi * Rbara * yia)
    zwhat = np.sum(wi * Rbara * zia)
    Rwbar = np.sqrt(xwhat ** 2 + ywhat ** 2 + zwhat ** 2)
    muhatpooled = np.array([xwhat / Rwbar, ywhat / Rwbar, zwhat / Rwbar])
    mdirpooled = cart2sph(muhatpooled)
    rhowhat = np.sum(wi * Rbara)

    if flagtest:
        Vwhat = np.sum(nsa * dia) / (2 * np.sum(nsa * Rbara) ** 2)
    else:
        Vwhat = 1 / np.sum(nsa * Rbara / dia)

    sigmaw = np.sqrt(2 * Vwhat)
    qw = np.arcsin(np.sqrt(-np.log(alpha) * sigmaw ** 2 * rhowhat ** 2 / Rwbar ** 2))

    return mdirpooled, sigmaw, qw


def isfishercommonmean(samplecartlist: list, alpha: float = 0.05) -> dict:
    """
    Test of whether two or more Fisher distributions have a common mean direction [#]_, [#]_, [#]_

    :param samplecartlist: List containing individual samples top be tested in 'cart' format
    :type samplecartlist: list[dict]
    :param alpha: Type-I error level
    :type alpha: float
    :return: Dictionary with the fields...
        - gr: Test statistic [float]
        - cval: Critical value to test against [float]
        - testresult: Test result [bool]
    :rtype: dict

    .. [#] Watson, G. S. (1956). Analysis of dispersion on a sphere. Mon. Not. R. Astr. Soc. Geophys. Suppl. 7, 153-159.

    .. [#] Watson, G. S. & Williams, E. J. (1956). On the construction of significance tests on the circle and the sphere. Biometrika 43, 344-352.

    .. [#] Watson, G. S. (1983). Large sample theory of the Langevin distributions. Journal of Statistical Planning and Inference 8, 245-256.
    """
    try:
        assert len(samplecartlist) >= 2
    except AssertionError:
        raise AssertionError('The number of samples to be tested should be at least 2')

    Zp = 0
    N = 0
    xhati = []
    yhati = []
    zhati = []
    kappahati = []
    Ria = []
    ns = []
    r = len(samplecartlist)
    rs = isfishercommonkappa(samplecartlist)
    kappasameflag = rs['testresult']
    if kappasameflag:
        for sample in samplecartlist:
            rs = resultants(sample)
            Ri = rs['Resultant Length']
            Ria.append(Ri)
            ni = sample['n']
            ns.append(ni)
            N += ni
            Zp += Ri
            xyzhati = rs['Directional Cosines']
            xhati.append(xyzhati[0])
            yhati.append(xyzhati[1])
            zhati.append(xyzhati[2])

        Z = Zp / N
        xhata = np.array(xhati)
        yhata = np.array(yhati)
        zhata = np.array(zhati)
        Rbar = np.sqrt(np.sum(Ria * xhata) ** 2 + np.sum(Ria * yhata) ** 2 + np.sum(Ria * zhata) ** 2) / N
        # print(Rbar)
        if r == 2:
            if ns[0] == ns[1]:
                if Rbar > 0.75:
                    fstar = fish.ppf(1 - alpha, 2, 2 * N - 4)  # Is this upper or lower?
                    z0 = (Rbar + fstar / (N - 2)) / (1 + fstar / (N - 2))  # Critical value
                    res = (Z < z0)
                    result = {'Z': Z, 'z0': z0, 'res': res}
                else:
                    N = ns[0] + ns[1]
                    z0 = a20(N=N, alpha=alpha, Rbar=Rbar)
                    res = (Z < z0)
                    result = {'Z': Z, 'z0': z0, 'res': res}
            else:
                if Rbar > 0.70:  # The books says 0.75 but does not provide values between 0.7 and 0.75 so are using 0.70 instead
                    fstar = fish.ppf(1 - alpha, 2, 2 * N - 4)  # Is this upper or lower?
                    z0 = (Rbar + fstar / (N - 2)) / (1 + fstar / (N - 2))  # Critical value
                    res = (Z < z0)
                    result = {'Z': Z, 'z0': z0, 'res': res}
                else:
                    gamma = ns[0] / ns[1]
                    N = ns[0] + ns[1]
                    if gamma < 1:
                        gamma = 1 / gamma

                    if 1 < gamma < 2:
                        alphagam = gamma - 1
                        z00 = a20(N, alpha, Rbar)
                        z01 = a21(2, N, alpha, Rbar)
                        z0 = linearinterp(z00, z01, alphagam)
                    elif 2 < gamma < 4:
                        z0 = a21(gamma, N, alpha, Rbar)
                    else:
                        raise ValueError('gamma > 4 case is not covered')
                    if z0 is None:
                        fstar = fish.ppf(1 - alpha, 2, 2 * N - 4)  # Is this upper or lower?
                        z0 = (Rbar + fstar / (N - 2)) / (1 + fstar / (N - 2))  # Critical value
                    res = (Z < z0)
                    result = {'Z': Z, 'z0': z0, 'res': res}
        else:
            if np.all(np.array(Ria) >= 0.55):
                fstar = fish.ppf(1 - alpha, 2 * (r - 1) - 2, 2 * N - 2 * (r - 2))
                f = fstar / (N - r) * (r - 1)
                z0 = (Rbar + f) / (1 + f)
                res = (Z < z0)
                result = {'Z': Z, 'z0': z0, 'res': res}
            else:
                raise ValueError('Test for small Ri not yet implemented!')
    else:
        for sample in samplecartlist:
            rs = resultants(sample)
            Ri = rs['Resultant Length']
            Ria.append(Ri)
            ni = sample['n']
            ns.append(ni)
            N += ni
            Zp += Ri
            xyzhati = rs['Directional Cosines']
            xhati.append(xyzhati[0])
            yhati.append(xyzhati[1])
            zhati.append(xyzhati[2])
            kappahat = (ni - 1) / (ni - Ri)
            kappahati.append(kappahat)
        try:
            assert np.all(np.array(ns) >= 25)
        except AssertionError:
            raise AssertionError('All sample sizes should be at least 25!')

        R = np.sum(np.array(kappahati) * np.array(Ria))
        xhat = np.sum(np.array(kappahati) * np.array(Ria) * np.array(xhati))
        yhat = np.sum(np.array(kappahati) * np.array(Ria) * np.array(yhati))
        zhat = np.sum(np.array(kappahati) * np.array(Ria) * np.array(zhati))
        Rw = np.sqrt(xhat ** 2 + yhat ** 2 + zhat ** 2)
        gr = 2 * (R - Rw)
        cval = chi2.ppf(1 - alpha, 2 * (r - 1))
        res = gr < cval
        result = {'gr': gr, 'cval': cval, 'testresult': res}
    return result


def a20(N=12, alpha: float = 0.05, Rbar: float = 0.7):
    """
    Utility function used by isfishercommonmean() to extract tabulated critical values
    """
    '''Tabulated data from Appendix 20 in the book'''
    N1, N2, R1, R2 = 0, 0, 0, 0
    alphaR, betaN = 0., 0.

    def selectval_a20(dfi, Ni, alphai, Rbari):
        z0N = dfi[dfi['N'] == Ni]
        z0alpha = z0N[z0N['alpha'] == alphai]
        z0Rbar = z0alpha[z0alpha['Rbar'] == Rbari]
        z0out = z0Rbar['z0'].values[0]
        return z0out

    try:
        assert alpha in {0.1, 0.05, 0.025, 0.01}
    except AssertionError:
        raise AssertionError('Test level, alpha, should be either 0.1, 0.05, 0.025, or 0.01.')

    filepath = pkg.resource_filename(__name__, 'data/A20.xlsx')
    df = pd.read_excel(filepath)
    Narr = list(set(df['N'].tolist()))
    interpNflag = False
    interpRflag = False
    if N not in Narr:
        No = Narr + [N]
        Na = np.sort(np.array(No))
        Nind = np.where(Na == N)[0][0]
        if Nind == 0:
            N = Na[0]
        elif Nind == len(Narr):
            N = Na[-1]
        else:
            N1 = Na[Nind - 1]
            N2 = Na[Nind + 1]
            betaN = (N - N1) / (N2 - N1)
            interpNflag = True

    Rbararr = list(set(df['Rbar'].tolist()))

    if Rbar not in Rbararr:
        Rbaro = Rbararr + [Rbar]
        Rbara = np.sort(np.array(Rbaro))
        Rind = np.where(Rbara == Rbar)[0][0]
        if Rind == 0:
            Rbar = Rbara[0]
        elif Rind == len(Rbararr):
            Rbar = Rbara[-1]
        else:
            R1 = Rbara[Rind - 1]
            R2 = Rbara[Rind + 1]
            alphaR = (Rbar - R1) / (R2 - R1)
            interpRflag = True

    if not (interpNflag | interpRflag):  # 00
        z0 = selectval_a20(df, N, alpha, Rbar)
        # z0N = dfi[dfi['N']==N]
        # z0alpha = z0N[z0N['alpha']==alpha]
        # z0Rbar = z0alpha[z0alpha['Rbar']==Rbar]
        # z0 = z0Rbar['z0'].values[0]
    elif interpNflag & interpRflag:  # interpolate both
        z00 = selectval_a20(df, N1, alpha, R1)
        z01 = selectval_a20(df, N1, alpha, R2)
        z10 = selectval_a20(df, N2, alpha, R1)
        z11 = selectval_a20(df, N2, alpha, R2)
        z0 = bilinearinterp(z00, z01, z10, z11, alphaR, betaN)
    elif interpNflag & ~interpRflag:  # Interpolate N only
        z00 = selectval_a20(df, N1, alpha, Rbar)
        z10 = selectval_a20(df, N2, alpha, Rbar)
        z0 = linearinterp(z00, z10, betaN)
    else:  # Interpolate Rbar only
        z01 = selectval_a20(df, N, alpha, R1)
        z11 = selectval_a20(df, N, alpha, R2)
        z0 = linearinterp(z01, z11, betaN)
    # Carry out interpolation using the utility functions below
    return z0


def bilinearinterp(z00: float, z01: float, z10: float, z11: float, alpha: float, beta: float):
    """
    Bilinear interpolation between 4 values

    :param z00: Value 1
    :type z00: float
    :param z01: Value 2
    :type z01: float
    :param z10: Value 3
    :type z10: float
    :param z11: Value 4
    :type z11: float
    :param alpha: Interpolation coefficient along axis 1 (0<=alpha<=1)
    :type alpha: float
    :param beta: Interpolation coefficient along axis 2 (0<=beta<=1)
    :type beta: float
    :return: Interpolated value
    :rtype: float
    """
    try:
        assert (0 <= alpha <= 1) and (0 <= beta <= 1)
    except AssertionError:
        raise AssertionError('alpha and beta should be in [0,1]')

    z0 = z00 * (1 - alpha) * (1 - beta) + z01 * (1 - alpha) * beta + z10 * alpha * (1 - beta) + z11 * alpha * beta
    return z0


def linearinterp(z00: float, z01: float, alpha: float) -> float:
    """
    Linear interpolation between 4 values

    :param z00: Value 1
    :type z00: float
    :param z01: Value 2
    :type z01: float
    :param alpha: Interpolation coefficient (0<=alpha<=1)
    :type alpha: float
    :return: Interpolated value
    :rtype: float
    """
    try:
        assert (0 <= alpha <= 1)
    except AssertionError:
        raise AssertionError('alpha should be in [0,1]')
    z0 = z00 * (1 - alpha) + z01 * alpha
    return z0


def a21(gamma=2, N=20, alpha=0.05, Rbar=0.1):
    """
    Utility function used by isfishercommonmean() to extract tabulated critical values
    """
    alist = [0.01, 0.025, 0.05, 0.1]
    Nlist = [20, 24, 30, 40, 60, 120]
    Rlist = [0.1, 0.15, 0.2, 0.25, 0.3]
    interpNflag = interpRflag = False
    N1, N2, R1, R2 = 0, 0, 0, 0
    alphaR, betaN = 0., 0.
    try:
        assert alpha in alist
    except AssertionError:
        raise AssertionError('Test level, alpha, should be either 0.1, 0.05, 0.025, or 0.01.')

    def selectval_a21(dfi, gammai, Ni, alphai, Rbari):
        z0N = dfi[dfi['N'] == Ni]
        z0alpha = z0N[z0N['alpha'] == alphai]
        z0Rbar = z0alpha[z0alpha['Rbar'] == Rbari]
        z0gamma = z0Rbar[z0Rbar['gamma'] == gammai]
        z0i = z0gamma['z0'].values[0]
        return z0i

    filepath = pkg.resource_filename(__name__, 'data/A21.xlsx')
    df = pd.read_excel(filepath)

    if Rbar not in Rlist:
        interpRflag = True
        Rbaro = Rlist + [Rbar]
        Rbara = np.sort(np.array(Rbaro))
        Rind = np.where(Rbara == Rbar)
        if Rind == 0:
            return None
        elif Rind == len(Rbara):
            return None
        else:
            R1 = Rbara[Rind - 1]
            R2 = Rbara[Rind + 1]
            alphaR = (Rbar - R1) / (R2 - R1)
    if N not in Nlist:
        interpNflag = True
        Nao = Nlist + [N]
        Na = np.sort(np.array(Nao))
        Nind = np.where(Na == N)
        if Nind == 0:
            N = Nlist[0]
        elif Nind == len(Nlist):
            N = Nlist[-1]
        else:
            N1 = Nlist[Nind - 1]
            N2 = Nlist[Nind + 1]
            betaN = (N - N1) / (N2 - N1)
    '''Tabulated data from Appendix 21 in the book'''
    if interpNflag and interpRflag:
        z00 = selectval_a21(df, gamma, N1, alpha, R1)
        z01 = selectval_a21(df, gamma, N1, alpha, R2)
        z10 = selectval_a21(df, gamma, N2, alpha, R1)
        z11 = selectval_a21(df, gamma, N2, alpha, R2)
        z0 = bilinearinterp(z00, z01, z10, z11, alphaR, betaN)
    elif interpNflag and ~interpRflag:
        z00 = selectval_a21(df, gamma, N1, alpha, Rbar)
        z01 = selectval_a21(df, gamma, N2, alpha, Rbar)
        z0 = linearinterp(z00, z01, betaN)
    elif interpRflag and ~interpNflag:
        z00 = selectval_a21(df, gamma, N, alpha, R1)
        z01 = selectval_a21(df, gamma, N, alpha, R2)
        z0 = linearinterp(z00, z01, alphaR)
    else:
        z0 = selectval_a21(df, gamma, N, alpha, Rbar)
    return z0


def a23(nu=2, r=2):
    """
    Utility function used by isfishercommonkappa() to extract tabulated critical values
    """
    nuarro = [2, 4, 6, 8, 10, 12, 20, 30, 60, 1e10]
    nu1, nu2, r1, r2 = 0, 0, 0, 0
    betar, betanu = 0, 0
    rarr = list(range(2, 13))
    interpRflag, interpNflag = False, False

    rarro = []
    if r not in rarr:
        interpRflag = True
        rarro += [r]
        rao = np.sort(np.array(rarro))
        rind = np.where(rao == r)[0][0]
        if rind == 0:
            raise ValueError('The number of samples cannot be 1!')
        elif rind == len(rarr):
            raise ValueError('The number of samples cannot be more than 12!')
        else:
            r1 = rao[rind - 1]
            r2 = rao[rind + 1]
            betar = (r - r1) / (r2 - r1)

    if nu not in nuarro:
        interpNflag = True
        nuarro += [nu]
        nuo = np.sort(np.array(nuarro, dtype=int))
        nuind = np.where(nuo == nu)[0][0]
        if nuind == 0:
            raise ValueError('nu cannot be less than 2!')
        elif nuind == len(nuarro):
            raise ValueError('The number of samples cannot be more than 12!')
        else:
            nu1 = nuo[nuind - 1]
            nu2 = nuo[nuind + 1]
            betanu = (nu - nu1) / (nu2 - nu1)

    filepath = pkg.resource_filename(__name__, 'data/A23.xlsx')
    df = pd.read_excel(filepath)

    def selectval_a23(dfi, nui, ri):
        dfnu = dfi[dfi['nu'] == nui]
        dfnur = dfnu[dfnu['r'] == ri]
        z0i = dfnur.values[0][2]
        return z0i

    if interpNflag and interpRflag:
        z00 = selectval_a23(df, nu1, r1)
        z01 = selectval_a23(df, nu1, r2)
        z10 = selectval_a23(df, nu2, r1)
        z11 = selectval_a23(df, nu2, r2)
        z0 = bilinearinterp(z00, z01, z10, z11, betar, betanu)
    elif interpNflag and ~interpRflag:
        z00 = selectval_a23(df, nu1, r)
        z01 = selectval_a23(df, nu2, r)
        z0 = linearinterp(z00, z01, betanu)
    elif interpRflag and ~interpNflag:
        z00 = selectval_a23(df, nu, r1)
        z01 = selectval_a23(df, nu, r2)
        z0 = linearinterp(z00, z01, betar)
    else:
        z0 = selectval_a23(df, nu, r)

    return z0


def fishercommonmean(samplecartlist: list, alpha: float = 0.05) -> tuple:
    """
    Estimation of the common mean direction of two or more Fisher distributions [#]_, [#]_

    :param samplecartlist: List containing individual samples top be tested in 'cart' format
    :type samplecartlist: list[dict]
    :param alpha: Semi-vertical angle for (1-alpha)% confidence cone is calculated
    :type alpha: float
    :return:
        - mdir: Tuple containing the common mean direction [th, ph]
        - qw: Semi-vertical angle [float]
    :rtype: tuple

    .. [#] Fisher, N. I. & Lewis, T. (1983). Estimating the common mean direction of several circular or spherical distributions with differing dispersions. Biometrika 70, 333-341.

    .. [#] Watson, G. S. (1983). Statistics on Spheres. University of Arkansas Lecture Notes in the Mathematical Sciences, Volume 6. New York: John Wiley.
    """
    res = isfishercommonkappa(samplecartlist)
    mdirlist, kappalist = [], []
    kappasameflag = res['testresult']
    Rbarlist = []
    xhatilist = []
    nilist = []
    for sample in samplecartlist:
        rsd = fisherparams(sample)
        mdir = rsd['mdir']
        kappa = rsd['kappa']
        mdirlist.append(mdir)
        kappalist.append(kappa)
        nilist.append(sample['n'])
        r = resultants(sample)
        Rbarlist.append(r['Mean Resultant Length'])
        xhatilist.append(r['Directional Cosines'])

    Rbara = np.array(Rbarlist)

    kapparatio = max(kappalist) / min(kappalist)
    kappaa = np.array(kappalist)
    nia = np.array(nilist)
    N = np.sum(nia)

    xhat = np.zeros((3,))
    if kappasameflag:
        samplecart = poolsamples(samplecartlist)
        mdir, kappa, qw, _ = fisherparams(samplecart)
        alphahat, betahat = mdir[0], mdir[1]
    elif kapparatio <= 4:
        wia = nia / N
        for xhati in xhatilist:
            xhat += np.sum(Rbara * xhati * wia)
        xhat /= np.linalg.norm(xhat)
        alphahat, betahat = cart2sph(xhat)
        qw = np.arcsin(
            sqrt(-np.log(alpha) * np.sum(2 * nia * Rbara / (kappaa * N ** 2 * np.linalg.norm(xhat) ** 2))))
    else:
        wia = nia * kappaa / np.sum(nia * kappaa)
        for xhati in xhatilist:
            xhat += np.sum(Rbara * xhati * wia)
        xhat /= np.linalg.norm(xhat)
        alphahat, betahat = cart2sph(xhat)
        qw = np.arcsin(sqrt(-np.log(alpha) * np.sum(
            2 * nia * Rbara * kappaa / (np.sum(nia * kappaa) ** 2 * np.linalg.norm(xhat) ** 2))))
    mdir = (alphahat, betahat)
    return mdir, qw


def isfishercommonkappa(samplecartlist: list) -> dict:
    """
    Test of whether two or more Fisher distributions (with unknown means) have a common concentration parameter at 0.05 level [#]_, [#]_

    :param samplecartlist: List containing individual samples top be tested in 'cart' format
    :type samplecartlist: list[dict]
    :return: Dictionary containing test results...
        - Z: Test statistics [float]
        - cval: Critical value [float]
        - df: Degrees of freedom [int, tuple(int, int)]
        - testresult: Test result [bool]
    :rtype: dict

    .. [#] Watson, G. S. & Irving, E. (1957). Statistical methods in rock magnetism. Mon. Not. R. astr. Soc. geophys. Suppl. 7, 289-300. (66, 136, 224)

    .. [#] Watson, G. S. & Williams, E. J. (1956). On the construction of significance tests on the circle and the sphere. Biometrika 43, 344-352. (14, 133, 211, 224)
    """

    alpha = 0.05
    Rbara = []
    nia = []
    Ra = []
    for samp in samplecartlist:
        r = resultants(samp)
        Rbara.append(r['Mean Resultant Length'])
        Ra.append(r['Resultant Length'])
        nia.append(samp['n'])

    r = len(samplecartlist)
    sampleflag = all([nia[ind] == nia[0] for ind in range(r)])
    rflag = all([Rbara[ind] > 0.65 for ind in range(r)])

    if r == 2:
        try:
            assert rflag
        except AssertionError:
            raise AssertionError(
                'Mean resultant lengths of samples must be greater than 0.65. See Mardia (1972, §9.4.2) for approximate tests')
        Z0 = ((nia[1] - 1) * (nia[0] - Ra[0])) / ((nia[0] - 1) * (nia[1] - Ra[1]))
        if Z0 < 1:
            Z0 = 1 / Z0
        cval = fish.ppf(1 - alpha / 2, 2 * nia[0] - 1, 2 * nia[1] - 1)
        testresult = Z0 < cval
        res = {'Z': Z0, 'cval': cval, 'df': {'Fdfn': 2 * nia[0] - 1, 'Fdfd': 2 * nia[1] - 1}, 'testresult': testresult}
    else:
        if sampleflag:
            n = nia[0]
            Z = (n - min(Ra)) / (n - max(Ra))
            nu = 2 * n - 2
            cval = a23(nu, r)
            testresult = Z < cval
            res = {'Z': Z, 'cval': cval, 'df': {'nu': nu, 'r': r}, 'testresult': testresult}
        else:
            nia = np.array(nia)
            mia = np.array(nia) - 1
            M = np.sum(mia)
            Ria = np.array(Ra)
            Z0 = 2 * M * np.log(np.sum(nia - Ria) / M) - 2 * np.sum(mia * np.log((nia - Ria) / mia))
            D = 1 + (np.sum(1 / mia) - (1 / M)) / (3 * (r - 1))
            Z = Z0 / D
            cval = chi2.ppf(1 - alpha, r - 1)
            testresult = Z < cval
            res = {'Z': Z, 'cval': cval, 'df': r - 1, 'testresult': testresult}
    return res


def fishercommonkappa(samplecartlist, alpha=0.05):
    """
    Estimation of the common concentration parameter of two or more Fisher distributions

    :param samplecartlist: List containing individual samples top be tested in 'cart' format
    :type samplecartlist: list[dict]
    :param alpha: Semi-vertical angle for (1-alpha)% confidence cone is calculated
    :type alpha: float
    :return:
        - kappahat: Pooled concentration parameter [float]
        - ku, kl: Upper and lower critical values for the (1-alpha)% CI
    :rtype: tuple
    """
    try:
        for sample in samplecartlist:
            assert sample['type'] == 'cart'
    except AssertionError:
        raise AssertionError('All samples should be in cart format.')

    Rilist, nilist = [], []
    ri = len(samplecartlist)
    for sample in samplecartlist:
        r = resultants(sample)
        Rilist.append(r['Resultant Length'])
        nilist.append(sample['n'])
    N = float(sum(nilist))
    Rs = sum(Rilist)
    kappahat = (N - ri) / (N - Rs)
    kl = 0.5 * chi2.ppf(alpha / 2, 2 * N - 2 * ri) / (N - Rs)
    ku = 0.5 * chi2.ppf(1 - alpha / 2, 2 * N - 2 * ri) / (N - Rs)
    return kappahat, (kl, ku)


def errors(samplecart: dict, srcpos: tuple) -> list:
    """
    Calculate angular error from a given direction

    :param samplecart: Sample in cart format
    :type samplecart: dict
    :param srcpos: Direction (th, ph) with respect to which the error will be calculated
    :return: Errors in radians
    :rtype: list
    """
    errs = []
    srcvec = sph2cart(srcpos[0], srcpos[1])
    for pt in samplecart['points']:
        errs.append(np.arccos(np.dot(pt, srcvec)))
    return errs
