# coding: utf-8

from setuptools import setup
from hdnnpy import __version__

setup(
    name='hdnnpy',
    version=__version__,
    description='High Dimensional Neural Network Potential package',
    long_description=open('README.md').read(),
    author='Masayoshi Ogura',
    author_email='ogura@cello.t.u-tokyo.ac.jp',
    url='https://github.com/ogura-edu/HDNNP',
    license='MIT',
    packages=[
        'hdnnpy',
        'hdnnpy.cli',
        'hdnnpy.dataset',
        'hdnnpy.dataset.descriptor',
        'hdnnpy.dataset.property',
        'hdnnpy.format',
        'hdnnpy.model',
        'hdnnpy.preprocess',
        'hdnnpy.training',
        'hdnnpy.training.loss_function',
        ],
    scripts=[
        'scripts/merge_xyz',
        'scripts/outcar2xyz',
        'scripts/poscars2xyz',
        ],
    entry_points={
        'console_scripts': ['hdnnpy = hdnnpy.cli:main'],
        },
    zip_safe=False,
    )
