# Author:  DINDIN Meryll
# Date:    02/03/2019
# Project: optimizers

try: from optimizers.imports import *
except: from imports import *

# Cast float to integers when needed

def handle_integers(params):

    new_params = {}

    for k, v in params.items():
        if type(v) == float and int(v) == v: new_params[k] = int(v)
        else: new_params[k] = v
    
    return new_params

# Defines the scoring function

def kappa_score(true, pred):

    cfm = confusion_matrix(true, pred)
    n_c = len(np.unique(true))
    s_0 = np.sum(cfm, axis=0)
    s_1 = np.sum(cfm, axis=1)
    exp = np.outer(s_0, s_1).astype(np.double) / np.sum(s_0) 
    mat = np.ones([n_c, n_c], dtype=np.int)
    mat.flat[::n_c + 1] = 0
    sco = np.sum(mat * cfm) / np.sum(mat * exp)
    
    return 1 - sco

# Defines the mean percentage error as metric

def mean_percentage_error(true, pred):

    msk = true != 0.0
    
    return np.nanmean(np.abs(true[msk] - pred[msk]) / true[msk])
