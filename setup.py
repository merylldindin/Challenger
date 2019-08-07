# Author:  DINDIN Meryll
# Date:    03/03/2019
# Project: Challenger

from distutils.core import setup
from setuptools import find_packages

# Launch the setup

setup(name='challenger',
      version='1.0',
      description='Challenges Routines',
      url='http://github.com/Coricos/challenger',
      author='Dindin Meryll',
      author_email='meryll_dindin@berkeley.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
      	'joblib>=0.13.0',
		'numpy>=1.16.4',
		'catboost>=0.12.2',
		'lightgbm>=2.2.3',
		'xgboost>=0.81',
		'scipy>=1.1.0',
		'hyperopt>=0.1.1',
		'scikit_learn>=0.21.3',
		'Keras>=2.2.4',
		'pqdict>=1.0.0',
		'pywt>=1.0.6',
		'pandas>=0.23.4',
		'statsmodels>=0.9.0',
		'nolds>=0.5.1'
      ])