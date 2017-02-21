################################################################################
#  TLGProb: Two-Layer Gaussian Process Regression Model For
#           Winning Probability Calculation of Two-Team Sports
#  Github: https://github.com/MaxInGaussian/TLGProb
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.0.9'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if 'git+' not in x]

setup(
    name='TLGProb',
    version=__version__,
    description='TLGProb: Two-Layer Gaussian Process Regression Model For Winning Probability Calculation of Two-Team Sports',
    long_description=long_description,
    url='https://github.com/MaxInGaussian/TLGProb',
    download_url='https://github.com/MaxInGaussian/TLGProb/tarball/' + __version__,
    license='BSD',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='Max W. Y. Lam',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='maxingaussian@gmail.com'
)
