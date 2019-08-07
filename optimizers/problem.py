# Author:  DINDIN Meryll
# Date:    01/03/2019
# Project: optimizers

try: from optimizers.utils import *
except: from utils import *

class Logger:

    def __init__(self, filename, level=logging.DEBUG):

        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        fle = logging.FileHandler(filename, 'a')
        fle.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s | %(levelname)-7s > %(message)s', datefmt='%I:%M:%S')
        fle.setFormatter(fmt)
        self.logger.addHandler(fle)

    def info(self, message):

        self.logger.info(message)

class Problem:

    def __init__(self, function):

        self.cache = []
        self.score = []
        self.function = function

    def checkSaved(self, parameters):

        paramArray = np.asarray([parameters[key] for key in sorted(parameters.keys())])
        cacheArray = np.asarray([[ele[key] for key in sorted(ele.keys())] for ele in self.cache])

        for idx, observation in enumerate(cacheArray):
            if np.sum(~(paramArray == observation)) == 0:
                return True, idx

        return False, None

    def evaluate(self, parameters, forced={}, random_seed=None):

        parameters.update(forced)
        boolean, index = self.checkSaved(parameters)

        if boolean: 
            return self.score[index]
        else:
            self.cache.append(parameters)
            self.score.append(self.function(**parameters))
            return self.score[-1]

    def reset(self):

        self.score = []
        self.cache = []

    def bestScore(self):

        return max(self.score)

    def bestParameters(self):

        return self.cache[np.asarray(self.score).argmax()]

class Prototype:

    def __init__(self, x_t, x_v, y_t, y_v, model, objective, metric, threads=1, weights=False):

        self.x_t = x_t
        self.x_v = x_v
        self.y_t = y_t
        self.y_v = y_v

        self.model_id = model
        self.objective = objective
        self.metric = metric
        self.threads = threads
        self.weights = weights

        self.cache = []
        self.score = []

    def loadBoundaries(self):

        if self.objective == 'regression':
            try: from optimizers.reg_params import SPACE
            except: from reg_params import SPACE
            return SPACE[self.model_id]

        if self.objective == 'classification':
            try: from optimizers.clf_params import SPACE
            except: from clf_params import SPACE
            return SPACE[self.model_id]

    def computeWeights(self, vector):

        if self.weights:

            res = np.zeros(len(vector))
            
            if self.objective == 'regression':
                tmp = np.digitize(vector, np.linspace(min(vector), 0.9*max(vector), num=10))
                tmp = LabelEncoder().fit_transform(tmp)
            if self.objective == 'classification':
                tmp = LabelEncoder().fit_transform(vector)

            wei = compute_class_weight('balanced', np.unique(vector), vector)
            wei = wei / sum(wei)
            for ele in np.unique(tmp): res[np.where(tmp == ele)[0]] = wei[int(ele)]
            return res

        else:
            return np.ones(len(vector))

    def matchModel(self, parameters, random_seed):

        if self.model_id == 'LGR':

            if self.objective == 'classification':
                arg = {'n_jobs': self.threads, 'random_state': random_seed}
                return LogisticRegression(**arg, **parameters)
        
        if self.model_id == 'SGD':

            if self.objective == 'classification':
                return SGDClassifier(random_state=random_seed, **parameters)
            if self.objective == 'regression':
                return SGDRegressor(random_state=random_seed, **parameters)
        
        if self.model_id == 'SVM':

            if self.objective == 'classification':
                return SVC(random_state=random_seed, **parameters)
            if self.objective == 'regression':
                return SVR(random_state=random_seed, **parameters)

        if self.model_id == 'ETS':

            if self.objective == 'classification':
                arg = {'n_jobs': self.threads, 'random_state': random_seed}
                return ExtraTreesClassifier(**arg, **parameters)
            if self.objective == 'regression':
                arg = {'n_jobs': self.threads, 'random_state': random_seed}
                return ExtraTreesRegressor(**arg, **parameters)

        if self.model_id == 'RFS':

            if self.objective == 'classification':
                arg = {'n_jobs': self.threads, 'random_state': random_seed}
                return RandomForestClassifier(**arg, **parameters)
            if self.objective == 'regression':
                arg = {'n_jobs': self.threads, 'random_state': random_seed}
                return RandomForestRegressor(**arg, **parameters)

        if self.model_id == 'LGB':

            if self.objective == 'classification' and len(np.unique(self.y_t)) > 2:
                arg = {'verbosity': -1, 'objective': 'multiclass'}
                oth = {'n_jobs': self.threads, 'seed': random_seed}
                return LGBMClassifier(**oth, **arg, **parameters)
            if self.objective == 'classification' and len(np.unique(self.y_t)) == 2:
                arg = {'verbosity': -1, 'objective': 'binary'}
                oth = {'n_jobs': self.threads, 'seed': random_seed}
                return LGBMClassifier(**oth, **arg, **parameters)
            if self.objective == 'regression':
                arg = {'verbosity': -1, 'n_jobs': self.threads, 'seed': random_seed}
                return LGBMRegressor(**arg, **parameters)
                
        if self.model_id == 'XGB':

            if self.objective == 'classification': 
                oth = {'n_jobs': self.threads, 'seed': random_seed}
                return XGBClassifier(**oth, **parameters)
            if self.objective == 'regression':
                oth = {'n_jobs': self.threads, 'seed': random_seed}
                return XGBRegressor(**oth, **parameters)
        
        if self.model_id == 'CAT':

            if self.objective == 'classification':
                oth = {'thread_count': self.threads, 'random_seed': random_seed}
                return CatBoostClassifier(loss_function='MultiClass', **oth, **parameters)
            if self.objective == 'regression':
                oth = {'thread_count': self.threads, 'random_seed': random_seed}
                return CatBoostRegressor(loss_function='MAPE', **oth, **parameters)

    def getScoring(self, true, pred):

        if type(self.metric) == str:

            if self.metric == 'acc': return accuracy_score(true, pred)
            if self.metric == 'f1s': return f1_score(true, pred, average='weighted')
            if self.metric == 'kap': return kappa_score(true, pred)
            if self.metric == 'mcc': return matthews_corrcoef(true, pred)
            if self.metric == 'lls': return -log_loss(true, pred)
            if self.metric == 'mae': return -mean_absolute_error(true, pred)
            if self.metric == 'mse': return -mean_squared_error(true, pred)
            if self.metric == 'med': return -median_absolute_error(true, pred)
            if self.metric == 'mpe': return -mean_percentage_error(true, pred)
            if self.metric == 'rec': return recall_score(true, pred)
            if self.metric == 'pre': return precision_score(true, pred)

        else:
            return self.metric(true, pred)

    def checkSaved(self, parameters):

        paramArray = np.asarray([parameters[key] for key in sorted(parameters.keys())])
        cacheArray = np.asarray([[ele[key] for key in sorted(ele.keys())] for ele in self.cache])

        for idx, observation in enumerate(cacheArray):
            if np.sum(~(paramArray == observation)) == 0:
                return True, idx

        return False, None

    def fitModel(self, parameters, random_seed):

        warnings.simplefilter('ignore')
        model = self.matchModel(parameters, random_seed)
        arg = {'sample_weight': self.computeWeights(self.y_t)}
        model.fit(self.x_t, self.y_t, **arg)

        return model

    def evaluate(self, parameters, forced={}, random_seed=None):

        parameters.update(forced)
        parameters = handle_integers(parameters)
        boolean, index = self.checkSaved(parameters)

        if boolean: 
            return self.score[index]

        else:
            model = self.fitModel(parameters, random_seed)
            self.cache.append(parameters)
            self.score.append(self.getScoring(self.y_v, model.predict(self.x_v)))
            return self.score[-1]

    def reset(self):

        self.score = []
        self.cache = []

    def bestScore(self):

        return max(self.score)

    def bestParameters(self):

        return self.cache[np.asarray(self.score).argmax()]

    def bestModel(self, filename=None, random_seed=None):

        if not os.path.exists('models'): os.mkdir('models')

        model = self.fitModel(self.bestParameters(), random_seed)

        if not filename is None: joblib.dump(model, '/'.join(['models', filename]))

        return model
