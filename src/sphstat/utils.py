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
Utility functions
==================================================================

- :func:`degtorad` converts sample in 'deg' to a sample in 'rad'
- :func:`readsample` reads one or more samples from an Excel file
- :func:`polartocart` converts sample in 'rad' format to a sample in 'cart' format
- :func:`carttopolar` converts sample in 'cart' format to a sample in 'rad' format
- :func:`excludesample` removes an observation from a sample
- :func:`deepcopy` generates a deep copy of a sample
- :func:`poolsamples` combines multiple samples into a single sample
- :func:`angle` calculates the angle between two vectors
- :func:`cart2sph` converts a point from Cartesian to polar format
- :func:`sph2cart` converts a point from polar to Cartesian format
- :func:`negatesample` generates a sample with observations in the opposing direction of the observations in a sample
- :func:`cot` calculates the cotangent of the input
- :func:`poltoll` converts a pair of polar angles to langitude/latitude 'lonlat' form
- :func:`poltodi` converts a pair of polar angles to declination/inclination 'decinc' form
- :func:`convert` converts a sample in 'decinc' or 'lonlat' form to polar form
- :func:`maptofundamental` maps the angles to the fundamental range, i.e. 0 <= th <= pi and 0 <= phi < 2 * pi
- :func:`randompermutations` creates a number of random permutation indices
- :func:`jackknife` is a generic jackknife function
- :func:`prettyprintdict` prints a dictionary (e.g. results from a hypothesis test) in an easy to read format

"""

from math import factorial, sqrt
from typing import Callable
import numpy as np
import openpyxl as pyxlsx
from scipy.stats import norm


def degtorad(sample: dict) -> dict:
    """
    Utility function to convert observations on the unit sphere given in degrees to radians and create a new 'sample'

    :param sample: Dictionary that contains the data (e.g. from readsample())
    :type sample: dict
    :return: Converted dictionary containing data, number of data points, and type of data
    :rtype: dict
    """

    sampleconverted = sample
    if sample['type'] == 'rad':
        return sample
    try:
        assert sample['type'] == 'deg'
    except AssertionError:
        raise AssertionError('The sample type should be deg')
    ln = sample['n']
    for ind in range(ln):
        sampleconverted['tetas'][ind] = sampleconverted['tetas'][ind] / 360. * 2 * np.pi
        sampleconverted['phis'][ind] = sampleconverted['phis'][ind] / 360. * 2 * np.pi
        sampleconverted['type'] = 'rad'
    return sampleconverted


def readsample(path: str, wsindex=0, typ='deg') -> dict:
    """
    Reads a number of observations comprising a sample. Data assumed to be polar and in degrees by default

    :param path: String containing the PATH to the file containing data to be processed
    :type path: str
    :param wsindex: Index of the worksheet in which the data resides, Defaults to 0 which is the active worksheet
    :type wsindex: int, optional
    :param typ: 'deg', 'rad', 'cart' Defaults to 'deg'
    :type typ: str
    :return: Dictionary containing data, number of data points, and type of data
    :rtype: dict
    """

    wb = pyxlsx.load_workbook(path)
    wa = wb.worksheets[wsindex]
    ind = 1
    cella = 'A' + str(1)
    cellb = 'B' + str(1)
    tetas = []
    phis = []
    sample = dict()
    while wa[cella].value is not None:
        dataa = wa[cella].value
        datab = wa[cellb].value
        tetas.append(dataa)
        phis.append(datab)
        ind += 1
        cella = 'A' + str(ind)
        cellb = 'B' + str(ind)
    sample['tetas'] = tetas
    sample['phis'] = phis
    assert len(tetas) == len(phis)
    sample['n'] = len(tetas)
    sample['type'] = typ
    sample = degtorad(sample)
    return sample


def polartocart(sample: dict) -> dict:
    """
    Convert the sample given in polar coordinates to Cartesian

    :param sample: Dictionary that contains the data (e.g. from readsample())
    :type sample: dict
    :return: Converted dictionary containing data, number of data points, and type of data
    :rtype: dict
    """
    assert sample['type'] == 'rad'
    th = sample['tetas']
    ph = sample['phis']
    samplecart = dict()
    samplecart['points'] = []
    for ind in range(len(th)):
        x = np.sin(th[ind]) * np.cos(ph[ind])
        y = np.sin(th[ind]) * np.sin(ph[ind])
        z = np.cos(th[ind])
        samplecart['points'].append(np.array((x, y, z)))
    samplecart['n'] = sample['n']
    samplecart['type'] = 'cart'
    return samplecart


def carttopolar(sample: dict) -> dict:
    """
    Convert the sample given in Cartesian coordinates to polar/spherical

    :param sample: Dictionary that contains the data (e.g. from readsample()) in 'cart' format
    :type sample: dict
    :return: Converted dictionary containing data, number of data points, and type of data
    :rtype: dict
    """

    assert sample['type'] == 'cart'
    samplepol = dict()
    samplepol['tetas'] = []
    samplepol['phis'] = []

    for ind in range(sample['n']):
        th, ph = cart2sph(sample['points'][ind])
        thi, phi = maptofundamental((th, ph))
        samplepol['tetas'].append(thi)
        samplepol['phis'].append(phi)

    samplepol['n'] = sample['n']
    samplepol['type'] = 'rad'
    return samplepol


def excludesample(samplecart: dict, excludeind: int) -> dict:
    """
    Exclude the datum with a given index from the data and generate new sample with the outlier eliminated

    :param samplecart: Dictionary that contains the data (e.g. from readsample()) in 'cart' format
    :type samplecart: dict
    :param excludeind: Index of the datum to be excluded
    :return: Dictionary with the datum removed
    :rtype: dict
    """

    assert samplecart['type'] == 'cart'
    ssamp = dict()
    ssamp['n'] = samplecart['n']-1
    scpy = deepcopy(samplecart)
    scpy['points'].pop(excludeind)
    ssamp['points'] = []
    for pt in scpy['points']:
        ssamp['points'].append(pt)
    ssamp['type'] = 'cart'
    return ssamp


def deepcopy(samplecart: dict) -> dict:
    """
    Generate a new data dictionary that is a (deep) copy of the original

    :param samplecart: Dictionary that contains the data (e.g. from readsample()) in 'cart' format
    :type samplecart: dict
    :return: Dictionary with the datum removed
    :rtype: dict
    """

    ssamp = dict()
    ssamp['n'] = samplecart['n']
    ssamp['points'] = []
    for pt in samplecart['points']:
        ssamp['points'].append(pt)
    ssamp['type'] = 'cart'
    return ssamp


def poolsamples(samplelist: list, typ='cart') -> dict:
    """
    Combine multiple samples in a single data dictionary

    :param samplelist: List of data dictionaries
    :type samplelist: list
    :param typ: 'deg', 'rad', 'cart' Defaults to 'cart'
    :type typ: str
    :return: Dictionary that contains all data in the input list of data dictionaries
    :rtype: dict
    """

    ntot = 0

    for sample in samplelist:
        assert sample['type'] == typ  # Make sure that the type of all data are the same

    pooledsample = dict()
    pooledsample['type'] = typ
    pooledsample['points'] = []

    for sample in samplelist:
        ntot += sample['n']
        for pt in sample['points']:
            pooledsample['points'].append(pt)
    pooledsample['n'] = ntot

    return pooledsample


def angle(pt1: np.ndarray, pt2: np.ndarray) -> float:
    """
    Return the angle between two unit vectors

    :param pt1: Vector with unit norm
    :type pt1: numpy.array
    :param pt2: Vector with unit norm
    :type pt2: numpy.array
    :return: Angle between the two vectors in radians
    :rtype: float
    """

    assert type(pt1) == np.ndarray
    assert type(pt2) == np.ndarray
    assert np.isclose(np.linalg.norm(pt1), 1)
    assert np.isclose(np.linalg.norm(pt2), 1)
    return np.arccos(np.dot(pt1, pt2))


def cart2sph(pt: np.array) -> tuple:
    """
    Convert unit vector from Cartesian to polar (i.e. spherical) coordinates

    :param pt: Unit norm vector in Cartesian coordinates
    :return: Tuple containing inclination and azimuth angles: th, ph
    :rtype: tuple
    """
    assert np.isclose(np.linalg.norm(pt), 1)
    th = np.arccos(pt[2])
    if pt[1] == 0:
        ph = 0
        return th, ph
    if pt[0] == 0:
        ph = np.pi / 2
        return th, ph
    ph = np.arctan2(pt[1], pt[0])
    return th, ph


def sph2cart(th: float, ph: float) -> np.array:
    """
    Convert unit vector from polar (i.e. spherical) tp Cartesian coordinates

    :param th: Inclination angle of the vector 0 <= th <= pi
    :type th: float
    :param ph: Azimuth angle of the vector 0 <= th <= 2 * pi
    :type ph: float
    :return: Array containing x, y, z components of the vector
    :rtype: np.array
    """
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = np.cos(th)
    return np.array([x, y, z])


def negatesample(samplecart: dict) -> dict:
    """
    Negate the vectors in a sample

    :param samplecart: Dictionary that contains the data (e.g. from readsample())
    :type samplecart: dict
    :return: Sample with observations containing negated directions
    :rtype: dict
    """

    assert samplecart['type'] == 'cart'
    sampleneg = deepcopy(samplecart)
    for ind in range(samplecart['n']):
        sampleneg['points'][ind] = -sampleneg['points'][ind]
    return sampleneg


def cot(x):
    """
    Cotangent of the argument

    :param x: Angle in radians
    :type x: float | np.array
    :return: Cotangent of the input
    :rtype: float
    """
    return 1/np.tan(x)


def poltoll(th: float, ph: float) -> tuple:
    """
    Convert polar form to longitude/latitude form

    :param th: Colatitude angle (0 <= th <= np.pi)
    :type th: float
    :param ph: Longitude angle (0 <= ph < 2 * np.pi)
    :type ph: float
    :return:
        - lat: Latitude angle
        - lon: Longitude angle
    :rtype: tuple
    """
    try:
        assert (0 <= th <= np.pi) and (0 <= ph < 2 * np.pi)
    except AssertionError:
        raise AssertionError('Colatitude and longitude angles are not in fundamental range!')

    lat = th - np.pi / 2
    lon = ph
    return lat, lon


def poltodi(th, ph):
    """
    Convert polar form to declination/inclination form

    :param th: Colatitude angle (0 <= th <= np.pi)
    :type th: float
    :param ph: Longitude angle (0 <= ph < 2 * np.pi)
    :type ph: float
    :return:
        - lat: Inclination angle
        - lon: Declination angle
    :rtype: tuple
    """
    if not (0 <= th <= np.pi) and (0 <= ph < 2 * np.pi):
        th, ph =maptofundamental((th, ph))

    inc = th - np.pi/2
    dec = np.mod(2 * np.pi - ph, 2 * np.pi)
    return inc, dec


def convert(sample: dict, fromformat: str) -> dict:
    """
    Convert sample to the polar coordinates used in sphstat package
    :param sample: Sample in 'rad' format
    :param fromformat: One of 'latlon', 'decinc', 'plunge'...
    :return:
    """
    assert sample['type'] == 'rad'
    if fromformat=='latlon':
        for ind in range(sample['n']):
            sample['tetas'][ind] = -(sample['tetas'][ind] - np.pi/2)
    elif fromformat=='decinc':
        for ind in range(sample['n']):
            sample['tetas'][ind] = np.pi/2 - sample['tetas'][ind]
            sample['phis'][ind] = np.mod(sample['phis'][ind], 2 * np.pi)
    return sample


def maptofundamental(dirct: tuple) -> tuple:
    """
    Utility function to map a pair of angles to the fundamental polar range

    :param dirct: Inclination and azimuth angles
    :type dirct: tuple
    :return: Remapped angles th and ph
    :rtype: tuple
    """
    th = np.mod(dirct[0], np.pi)
    ph = np.mod(dirct[1], 2 * np.pi)
    return th, ph


def randompermutations(rangein: int, numpermute: int) -> list:
    """
    Generate random permutations

    :param rangein: Number of observations to be permuted (n)
    :type rangein: int
    :param numpermute: Number of unique permutations to use
    :type numpermute: int
    :return: List of lists of index permutations
    :rtype: list
    """
    try:
        assert numpermute <= factorial(rangein)
    except AssertionError:
        raise ValueError('Number of permutations cannot be larger than n!')
    perms = []
    while len(perms) < numpermute:
        perm = list(np.random.permutation(rangein))
        if all(v != perm for v in perms):
            perms.append(perm)
    return perms


def jackknife(estfun: Callable, funargs: tuple, sample: dict, dictkey: str = None, tupleind: int = None,
              alpha: float = 0.05, unbiasflag: bool = True) -> tuple:
    """
    Jackknife method for calculating an approximate confidence interval for an statistical parameter of a sample

    :param estfun: Function that outputs the desired statistical parameter
    :type estfun: Callable
    :param funargs: Arguments to be passed to estfun
    :type funargs: tuple
    :param sample: Sample to be used in the computations
    :type sample: dict
    :param dictkey: If the function returns a dictionary use the value for this key
    :param dictkey: 'str'
    :param tupleind: If the function returns a tuple use the value at this index
    :param tupleind: int
    :param alpha: (1-alpha)% CI is calculated
    :type alpha: float
    :param unbiasflag: Flag indicating whether estfun gives an unbiased estimate
    :type unbiasflag: bool
    :return:
        - psijhat: Unbiased estimate using the jackknife method (float)
        - ci: (1-alpha)% confidence interval
    :rtype: tuple
    """
    psilist = []
    n = sample['n']
    res = estfun(sample, *funargs)

    if type(res) == dict:
        psihat = res[dictkey]
    elif type(res) == tuple:
        psihat = res[tupleind]
    elif type(res) == (int or float or np.ndarray):
        psihat = res
    else:
        raise ValueError('Output type of the estimator is not recognised!')

    for ind in range(n):
        samplered = excludesample(sample, ind)
        res = estfun(samplered, *funargs)
        if type(res) == dict:
            psii = res[dictkey]
        elif type(res) == tuple:
            psii = res[tupleind]
        elif type(res) == (int or float):
            psii = res
        else:
            raise ValueError('Output type of the estimator is not recognised!')

        psiihat = n * psihat - (n - 1) * psii
        psilist.append(psiihat)
    psiihata = np.array(psilist)
    if unbiasflag:
        psijhat = psihat
    else:
        psijhat = np.sum(psiihata) / n
    sj2 = 1 / (n * (n - 1)) * np.sum((psijhat - psiihata) ** 2)
    za2 = norm.ppf(1 - alpha / 2)
    ci = (psijhat - sqrt(sj2) * za2, psijhat + sqrt(sj2) * za2)
    return psijhat, ci


def prettyprintdict(data: dict):
    """
    Print a dictionary contents in a user friendly way

    :param data: A dictionary containing some data
    :type data: dict
    :return: None
    :rtype: None
    """
    try:
        assert type(data) == dict
    except AssertionError:
        raise AssertionError('Data should be a dictionary')

    for key in data.keys():
        print(str(key) + ': ', data[key])
    pass
