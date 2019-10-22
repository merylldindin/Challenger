# Author:  DINDIN Meryll
# Date:    01/03/2019
# Project: optimizers

# General
import os
import six
import time
import json
import logging
import warnings
import joblib
import numpy as np
import pandas as pd

# Testing
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes

# Modeling
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import log_loss
from sklearn.metrics import median_absolute_error
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Bayesian Optimization
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

# Hyperband Optimization
from hyperopt import hp
from hyperopt.pyll.stochastic import sample

# Hyperopt Optimization
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import STATUS_OK

# Visualization packages
try:
	import seaborn as sns
	import matplotlib.pyplot as plt
	from matplotlib import cm
	from matplotlib.gridspec import GridSpec
except: 
	pass