from setuptools import setup, find_packages

setup(
    name='sphstat',
    version='1.0',
    description='A Python package for inferential statistics on vectorial data on the unit sphere',
    url='https://github.com/hacihabiboglu/sphstat',
    author='Huseyin Hacihabiboglu',
    author_email='mailto:hhuseyin@metu.edu.tr',
    license='MIT License',
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
        'Operating System :: POSIX :: Linux'
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    python_requires='>=3.6',
    include_package_data=True
)
