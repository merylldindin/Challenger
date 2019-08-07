# Author:  DINDIN Meryll
# Date:    02/03/2019
# Project: optimizers

SPACE = dict()

SPACE['LGR'] = {
    
    'tol': ('uniform_float', (1e-5, 0.1)),

    'C': ('uniform_log', (1e-3, 10)),

    'fit_intercept': ('choice', [True, False]),

    'solver': ('choice', ['newton-cg', 'sag', 'saga', 'lbfgs'])

    }

SPACE['SGD'] = {
    
    'loss': ('choice', ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']),

    'penalty': ('choice', ['none', 'l1', 'l2', 'elasticnet']),

    'alpha': ('uniform_log', (1e-10, 1.0)),

    'l1_ratio': ('uniform_float', (0.0, 1.0)),

    'fit_intercept': ('choice', [True, False]),

    'max_iter': ('uniform_int', (100, 1000)),

    'tol': ('uniform_float', (1e-5, 0.1)),

    'epsilon': ('uniform_log', (1e-5, 0.1)),

    'learning_rate': ('choice', ['constant', 'optimal', 'invscaling', 'adaptive']),

    'eta0': ('uniform_log', (1e-5, 1.0)),

    'power_t': ('uniform_float', (0.1, 1.0))

    }

SPACE['SVM'] = {
    
    'C': ('uniform_log', (1e-3, 10)),

    'kernel': ('choice', ['linear', 'poly', 'rbf', 'sigmoid']),

    'degree': ('uniform_int', (2, 5)),

    'gamma': ('uniform_log', (1e-8, 1.0)),

    'shrinking': ('choice', [True, False]),

    'tol': ('uniform_float', (1e-5, 0.1)),

    }

SPACE['ETS'] = {

    'n_estimators': ('uniform_int', (10, 1000)),

    'max_depth': ('uniform_int', (2, 16)),

    'min_samples_split': ('uniform_int', (2, 20)),

    'min_samples_leaf': ('uniform_int', (1, 10)),

    'min_weight_fraction_leaf': ('uniform_float', (0, 0.5)),  

    'criterion': ('choice', ['gini', 'entropy']),

    'bootstrap': ('choice', [True, False]),

    'max_features': ('uniform_float', (0.1, 1))

    }

SPACE['RFS'] = {

    'n_estimators': ('uniform_int', (10, 1000)),

    'max_depth': ('uniform_int', (2, 16)),

    'min_samples_split': ('uniform_int', (2, 20)),

    'min_samples_leaf': ('uniform_int', (1, 10)),

    'min_weight_fraction_leaf': ('uniform_float', (0, 0.5)),  

    'criterion': ('choice', ['gini', 'entropy']),

    'bootstrap': ('choice', [True, False]),

    'max_features': ('uniform_float', (0.1, 1))

    }

SPACE['LGB'] = {

    'n_estimators': ('uniform_int', (10, 1000)),

    'max_depth': ('uniform_int', (2, 16)),

    'learning_rate': ('uniform_log', (0.001, 0.5)),

    'num_leaves': ('uniform_int', (2, 50)),

    'min_child_weight': ('uniform_log', (1e-4, 0.1)),

    'min_child_samples': ('uniform_int', (10, 30)),

    'colsample_bytree': ('uniform_float', (0.5, 1.0)),

    'reg_alpha': ('uniform_log', (1e-10, 1)),

    'reg_lambda': ('uniform_log', (1e-3, 10))
    
    }

SPACE['XGB'] = {

    'n_estimators': ('uniform_int', (10, 1000)),

    'max_depth': ('uniform_int', (2, 16)),

    'learning_rate': ('uniform_log', (0.001, 0.5)),

    'num_leaves': ('uniform_int', (2, 50)),

    'min_child_weight': ('uniform_log', (1e-4, 0.1)),

    'min_child_samples': ('uniform_int', (10, 30)),

    'colsample_bytree': ('uniform_float', (0.5, 1.0)),

    'colsample_bylevel': ('uniform_float', (0.1, 1.0)),

    'reg_alpha': ('uniform_log', (1e-10, 1)),

    'reg_lambda': ('uniform_log', (1e-3, 10)),

    'gamma': ('uniform_float', (0.0, 1.0)),

    'subsample': ('uniform_float', (0.0, 1.0)), 
    
    }

SPACE['CAT'] = {
    
    'learning_rate': ('uniform_log', (0.001, 0.5)),

    'n_estimators': ('uniform_int', (10, 200)),

    'reg_lambda': ('uniform_log', (1e-3, 10)),

    'max_depth': ('uniform_int', (2, 12)),

    'colsample_bylevel': ('uniform_float', (0.1, 1.0)),

    }