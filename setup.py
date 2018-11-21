from setuptools import setup

setup(
    name='hdnnpy',
    version='0.1.0',
    description='High Dimensional Neural Network Potential package',
    long_description=open('README.md').read(),
    author='Masayoshi Ogura',
    author_email='ogura@cello.t.u-tokyo.ac.jp',
    url='https://github.com/ogura-edu/HDNNP',
    license='MIT',
    # src/__init__.py
    packages=['hdnnpy'],
    scripts=['scripts/merge_xyz', 'scripts/vasp2xyz'],
    entry_points={
        'console_scripts': ['hdnnpy = hdnnpy.hdnnp:main'],
    },
    zip_safe=False,
)
