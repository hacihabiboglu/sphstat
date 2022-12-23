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
Functions to calculate descriptive statistics on spherical data
==================================================================

- :func:`resultants` calculates the descriptive statistics of a sample
- :func:`rotationmatrix` calculates the rotation matrix for a given set of rotation angles
- :func:`rotationmatrix_withaxis` calculates the rotation matrix with respect to an axis
- :func:`rotate` rotates a point with the given set of rotation angles
- :func:`rotatesample` rotate all observations in a sample with the given set of rotation angles
- :func:`movematrix` calculates the matrix which moves a vector to another vector
- :func:`orientationmatrix` calculates the orientation matrix for a sample
- :func:`momentofinertia` calculates the moment of intertia of a sample with respect to a point
- :func:`pointsonanellipse` calculates a number of points on an ellipse
- :func:`mediandir` calculates the median direction of a sample

"""

import numpy as np
from scipy.optimize import minimize
from .utils import cart2sph, carttopolar, cot, maptofundamental


def resultants(samplecart: dict) -> dict:
    """
    Summary statistics of the data in sample. Data must be in cartesian coordinates (i.e. convert first using
    polartocart() if data is in polar coordinates)

    :param samplecart: Data in 'cart format'
    :return: A dictionary containing different summary statistics for the data
    :rtype: dict
    """

    try:
        assert samplecart['type'] == 'cart'
    except:
        raise AssertionError('Sample type should be cart')
    resstat = dict()
    resvec = np.zeros((3,))
    for pt in samplecart['points']:
        resvec += pt
    reslen = np.linalg.norm(resvec)  # R
    dircos = resvec / reslen  # xhat, yhat, zhat
    meandir = cart2sph(dircos)  # thetahat, phihat
    meanreslen = reslen / samplecart['n']  # Rbar
    resstat['Directional Cosines'] = dircos  # xhat, yhat, zhat
    resstat['Resultant Vector'] = resvec  # Sx, Sy, Sz
    resstat['Resultant Length'] = reslen  # R
    resstat['Mean Direction'] = meandir   # thetahat, phihat
    resstat['Mean Resultant Length'] = meanreslen  # Rbar
    return resstat


def rotationmatrix(th0: float, ph0: float, psi0: float = 0.) -> np.ndarray:
    """
    Rotation matrix to rotate data in a given direction (th0, ph0)

    :param th0: Inclination angle of the rotation
    :param ph0: Azimuth angle of the rotation
    :param psi0: Rotation angle about the polar axis (Default is 0)
    :return: Rotation matrix
    :rtype: np.ndarray
    """
    A = np.zeros((3, 3))
    A[0, 0] = np.cos(th0) * np.cos(ph0) * np.cos(psi0) - np.sin(ph0) * np.sin(psi0)
    A[0, 1] = np.cos(th0) * np.sin(ph0) * np.cos(psi0) + np.cos(ph0) * np.sin(psi0)
    A[0, 2] = - np.sin(th0) * np.cos(psi0)

    A[1, 0] = - np.cos(th0) * np.cos(ph0) * np.sin(psi0) - np.sin(ph0) * np.cos(psi0)
    A[1, 1] = - np.cos(th0) * np.sin(ph0) * np.sin(psi0) + np.cos(ph0) * np.cos(psi0)
    A[1, 2] = np.sin(th0) * np.sin(psi0)

    A[2, 0] = np.sin(th0) * np.cos(ph0)
    A[2, 1] = np.sin(th0) * np.sin(ph0)
    A[2, 2] = np.cos(th0)

    return A


def rotationmatrix_withaxis(ua: np.array, ub: np.array) -> np.ndarray:
    """
    Calculate a rotation matrix around an axis

    :param ua: First unit vector
    :type ua: np.array
    :param ub: First unit vector
    :type ub: np.array
    :return: Rotation matrix
    :rtype: np.ndarray
    """
    try:
        ua /= np.linalg.norm(ua)
        ub /= np.linalg.norm(ub)
    except ZeroDivisionError:
        raise ZeroDivisionError('Vectors have to be unit norm.')

    try:
        assert len(ua) == len(ub) == 3
    except AssertionError:
        raise AssertionError('The vectors are not of the correct shape.')

    ab = np.dot(ua, ub)
    ca = ua - ub * ab
    ca /= np.linalg.norm(ca)
    B = ub.reshape((3, 1)) @ ca.reshape((1, 3))
    B -= B.T
    th = np.arccos(ab)
    A = np.eye(3) + np.sin(th) * B + (np.cos(th) - 1) * ((ub.reshape((3, 1)) @ ub.reshape((1, 3))) + ca.reshape((3, 1))
                                                         @ ca.reshape((1, 3)))
    return A


def rotate(pt: np.array, th0: float, ph0: float, psi0: float = 0.0) -> np.ndarray:
    """
    Rotate a single point by (th0, ph0, psi0)

    :param pt: Point to be rotated
    :type pt: np.ndarray
    :param th0: Inclination rotation angle (in rad)
    :type th0: float
    :param ph0: Azimuth rotation angle (in rad)
    :type ph0: float
    :param psi0: Polar axis rotation angle (Defaults to 0.0)
    :type psi0: float
    :return: Rotated point
    :rtype: np.ndarray
    """
    try:
        assert type(pt) == np.ndarray
        assert np.shape(pt) == (3,)
    except AssertionError:
        raise AssertionError('Input point is not of the correct type/shape.')

    A = rotationmatrix(th0, ph0, psi0)
    return A @ pt


def rotatesample(sample: dict, th0: float, ph0: float, psi0: float = 0.) -> dict:
    """
    Rotate all observations in a sample by (th0, ph0, psi0)

    :param sample: Sample to be rotated in cart format
    :type sample: dict
    :param th0: Inclination rotation angle (in rad)
    :type th0: float
    :param ph0: Azimuth rotation angle (in rad)
    :type ph0: float
    :param psi0: Polar axis rotation angle (Defaults to 0.0)
    :type psi0: float
    :return: Sample with all points rotated by the given rotation parameters
    :rtype: dict
    """
    try:
        assert sample['type'] == 'cart'
    except AssertionError:
        raise AssertionError('Sample should be of type cart.')

    samplerot = dict()
    samplerot['type'] = 'cart'
    samplerot['n'] = sample['n']
    samplerot['points'] = []
    for ind in range(sample['n']):
        pt = sample['points'][ind]
        samplerot['points'].append(rotate(pt.reshape((3,)), th0, ph0, psi0))
    return samplerot


def movematrix(pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    """
    Move a vector pt1 to pt2

    :param pt1: Source point
    :type pt1: np.ndarray
    :param pt2: Destination point
    :type pt2: np.ndarray
    :return: Matrix that moves pt1 to pt2 such that pt2 = H @ pt1,
    :rtype: np.ndarray
    """
    try:
        assert np.isclose(np.linalg.norm(pt1), 1) and np.isclose(np.linalg.norm(pt2), 1)
    except AssertionError:
        raise AssertionError('Points should have unit norm.')

    try:
        assert pt1.shape[0] == pt2.shape[0] == 3
    except AssertionError:
        raise AssertionError('Points should have 3 components.')

    H = np.outer(pt1 + pt2, pt1 + pt2) / (1 + pt1.T @ pt2) - np.eye(3)
    return H


def orientationmatrix(samplecart):
    """
    Calculate the orientation matrix and its eigenvectors / eigenvalues for a given sample

    :param samplecart: Sample in cart format
    :type samplecart: dict
    :return: Orientation matrix
    :rtype: np.ndarray
    """

    try:
        assert samplecart['type'] == 'cart'
    except AssertionError:
        raise AssertionError('Sample type should be cart.')

    T = np.zeros((3, 3))
    for n in range(samplecart['n']):
        for ind in range(3):
            for jnd in range(3):
                T[ind, jnd] += samplecart['points'][n][ind] * samplecart['points'][n][jnd]
    tauhat, u = np.linalg.eigh(T)
    taubar = tauhat / samplecart['n']
    Tmat = dict()
    Tmat['T'] = T
    Tmat['u'] = u
    Tmat['tauhat'] = tauhat
    Tmat['taubar'] = taubar
    return Tmat


def momentofinertia(samplecart: dict, pt0: np.ndarray) -> float:
    """
    Calculate the moment of inertia for a point given a sample

    :param samplecart: Sample in cart format
    :type samplecart: dict
    :param pt0: Point
    :type pt0: np.ndarray
    :return: Moment of inertia
    :rtype: float
    """
    try:
        assert samplecart['type'] == 'cart'
    except AssertionError:
        raise AssertionError('Sample type should be cart.')

    Tmat = orientationmatrix(samplecart)
    inertia = samplecart['n'] - np.dot(pt0, (Tmat['T'] @ pt0))
    return inertia


def pointsonanellipse(h1: np.ndarray, h2: np.ndarray, h3: np.ndarray, ellipsecoeffs: tuple, psinum: int = 360) -> list:
    """
    Calculate a number of points on an ellipse on the sphere

    :param h1: First axis
    :type h1: np.ndarray
    :param h2: Second axis
    :type h2: np.ndarray
    :param h3: Third axis
    :type h3: np.ndarray
    :param ellipsecoeffs: Parameters of the ellipse
    :type ellipsecoeffs: tuple
    :param psinum: Number of points to calculate
    :type psinum: int
    :return: Points on a sphere as a list
    :rtype: list
    """
    try:
        assert np.isclose(np.linalg.norm(h1), 1)
        assert np.isclose(np.linalg.norm(h2), 1)
        assert np.isclose(np.linalg.norm(h3), 1)
        assert np.isclose(np.dot(h1, h2), 0)
        assert np.isclose(np.dot(h1, h3), 0)
        assert np.isclose(np.dot(h2, h3), 0)
    except AssertionError:
        raise AssertionError('The three axes should have unit norm and should be orthogonal')

    Hmat = np.zeros((3, 3))
    Hmat[:, 0] = h1
    Hmat[:, 1] = h2
    Hmat[:, 2] = h3

    # Step 1
    A, B, C, D = ellipsecoeffs
    Z = np.matrix([[A, B], [B, C]])
    w, u = np.linalg.eigh(Z)
    a = u[0, 0]
    b = u[1, 0]
    t1 = w[0]
    t2 = w[1]

    # Step 2
    g1 = np.sqrt(D/t1)
    g2 = np.sqrt(D/t2)

    # Step 3
    psis = np.linspace(0, 2 * np.pi, psinum)
    v1 = g1 * np.cos(psis)
    v2 = g2 * np.sin(psis)
    x = a * v1 - b * v2
    y = b * v1 + a * v2
    z = np.sqrt(1 - x**2 - y**2)

    # Step 4
    ptsonellipse = []
    for ind in range(psinum):
        pt = np.array([x[ind], y[ind], z[ind]])
        ptsonellipse.append(Hmat @ pt)

    return ptsonellipse


def shapestrength(sample: dict) -> tuple:
    """
    Exploratory summary metrics for assessing properties of the underlying distribution

    :param sample: Sample in cart format
    :type sample: dict
    :return:
        - shape: Shape parameter
        - strength: Strength parameter
    :rtype: tuple
    """
    try:
        assert sample['type'] == 'cart'
    except AssertionError:
        raise AssertionError('Sample type should be cart.')

    Tmat = orientationmatrix(sample)
    shape = np.log(Tmat['taubar'][2]/Tmat['taubar'][1]) / np.log(Tmat['taubar'][1]/Tmat['taubar'][0])
    strength = np.log(Tmat['taubar'][2]/Tmat['taubar'][0])
    return shape, strength


def mediandir(sample: dict, ciflag: bool = True, alpha: float = 0.05) -> tuple:
    """
    Find the median direction and the confidence cone of the sample.
    Before this, make sure that the sample comes from a unimodal distribution

    :param sample: Sample to be tested in 'cart' format
    :type sample: dict
    :param ciflag: (1-alpha)% confidence cone calculation
    :type ciflag: bool
    :param alpha: Type-I error level for the CI  (e.g. 0.05)
    :type alpha: float
    :return: Tuple with:
        - Median direction [theta, phi] (np.array)
        - Optimisation successful? (bool)
        - Confidence cone parameters (dict)
        - W matrix (np.ndarray)
    :rtype: tuple
    """
    # Cross-checked with Appendix_B1 data
    assert sample['type'] == 'cart'
    r = resultants(sample)
    ms = r['Mean Direction']
    def errmin(thph, sample):
        th = thph[0]
        ph = thph[1]
        xp = np.sin(th) * np.cos(ph)
        yp = np.sin(th) * np.sin(ph)
        zp = np.cos(th)
        x = np.array((xp,yp,zp))
        pts = sample['points']
        err = 0
        for pt in pts:
            err += np.arccos(np.dot(pt, x))
        return err
    # We will use Nelder-Mead to minimise the arc lengths
    res = minimize(errmin, np.array([ms[0], ms[1]]), args = (sample), method='Nelder-Mead', tol=1e-6)
    medi = res.x
    medi[0] = np.mod(medi[0], np.pi)
    medi[1] = np.mod(medi[1], 2 * np.pi)
    success = res.success
    ccone = None
    W = None
    if ciflag:
        try:
            assert sample['n'] >= 25
        except AssertionError:
            raise AssertionError('The sample size cannot be less than 25 for CI calculation.')
        samprot = rotatesample(sample, medi[0], medi[1], 0.)
        samppol = carttopolar(samprot)
        th = np.array(samppol['tetas'])
        ph = np.array(samppol['phis'])

        C11 = np.sum(cot(th) * (1 - np.cos(2 * ph)) / 2) / samppol['n']
        C22 = np.sum(cot(th) * (1 + np.cos(2 * ph)) / 2) / samppol['n']
        C12 = -np.sum(cot(th) * np.sin(2 * ph) / 2) / samppol['n']
        C = np.matrix([[C11, C12],[C12, C22]])

        sigma11 = 1 + np.sum(np.cos(2 * ph)) / samppol['n']
        sigma22 = 1 - np.sum(np.cos(2 * ph)) / samppol['n']
        sigma12 = np.sum(np.sin(2 * ph)) / samppol['n']

        Sigma = 0.5 * np.matrix([[sigma11, sigma12],[sigma12, sigma22]])

        W = np.matmul(np.matmul(C, np.linalg.inv(Sigma)), C)

        A = W[0, 0]
        B = W[0, 1]
        C = W[1, 1]
        D = -2 * np.log(alpha) / sample['n']

        v, u = np.linalg.eigh(W)
        ccone = dict()
        ccone['coeffs'] = (A, B, C, D)
        ccone['eigv'] = v
        ccone['eigu'] = u

    return medi, success, ccone, W
