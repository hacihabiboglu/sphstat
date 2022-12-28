from setuptools import setup, find_packages

setup(
    name = 'sphstat',
    version = '1.0.5',
    description = 'A Python 3 package for inferential statistics on vectorial data on the unit sphere',
    url='https://github.com/hacihabiboglu/sphstat',
    download_url = 'https://github.com/hacihabiboglu/sphstat/archive/refs/tags/sphstat-1.0.tar.gz',
    author='Huseyin Hacihabiboglu',
    author_email='mailto:hhuseyin@metu.edu.tr',
    license='MIT',
    keywords=['spherical statistics', 'vector statistics', 'hypothesis testing', 'spherical regression', 'inferential statistics'],
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    py_modules=['sphstat.utils', 'sphstat.distributions', 'sphstat.descriptives', 'sphstat.singlesample', 'sphstat.twosample', 'sphstat.modelling', 'sphstat.plotting'],
    install_requires=['numpy',
                      'scipy',
                      'matplotlib',
                      'openpyxl',
                      'sympy',
                      'pandas',
                      'setuptools'
                      ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    python_requires='>=3.8',
    include_package_data=True
)
