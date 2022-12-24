.. sphstat documentation master file, created by
   sphinx-quickstart on Thu Dec  8 11:27:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to sphstat's documentation!
===================================

|image0|

The **sphstat** package offers a range of tools for performing both
descriptive and inferential statistical analysis on data that
lies on the unit sphere. It includes implementations of various
tests and algorithms developed by Fisher, Lewis, and Embleton [1]_.

**N.B.** Tests and methods for axial data have not yet been implemented
as of version 1.0


`sphstat` implements several modules whose names are pretty self-explanatory. These are:

* `descriptives`: Functions for descriptive statistics on spherical data
* `distributions`: Functions for generating data from different spherical distributions
* `singlesample`: Functions for single-sample tests on spherical data
* `twosample`: Functions for inferential staticts on two or more samples
* `modelling`: Functions for cross-correlation, regression, and temporal analysis
* `utils`: Utility functions used by other modules

sphstat was implemented by Huseyin Hacihabiboglu, (https://www.hacihabiboglu.org)

The source code is available on Github: https://www.github.com/hacihabiboglu/sphstat

.. [1] Fisher, N.I., Lewis, T. and Embleton, B.J., 1993. Statistical analysis of spherical data. Cambridge University Press.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples
   sphstat

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |image0| image:: ./images/sphstatlogo.png
            :width: 250