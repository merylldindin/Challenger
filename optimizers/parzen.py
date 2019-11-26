# Author:  DINDIN Meryll
# Date:    01/03/2019
# Project: optimizers

try: from optimizers.problem import *
except: from problem import *

class Parzen:

    def __init__(self, problem, boundaries, logger, seed=42):

        self.logger = logger
        self.logger.info('# Hyperopt Optimization')

        self.space = dict()
        self.problem = problem
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)

        for key in sorted(boundaries.keys()):

            selection, bounds = boundaries[key]

            if selection == 'uniform_int': 
                self.space[key] = hp.quniform(key, bounds[0], bounds[1]+1, 1)
            if selection == 'uniform_float': 
                self.space[key] = hp.uniform(key, bounds[0], bounds[1])
            if selection == 'uniform_log': 
                self.space[key] = hp.loguniform(key, bounds[0], np.log(bounds[1]))
            if selection == 'choice':
                self.space[key] = hp.choice(key, bounds)

    def run(self, n_iter=100):

        def objective(params):

            try: score = self.problem.evaluate(handle_integers(params), random_seed=self.seed)
            except: score = -np.inf

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        arg = {'algo': tpe.suggest, 'max_evals': n_iter, 'show_progressbar': False}
        params = fmin(objective, self.space, trials=trials, **arg)
        self.logger.info('Best Score : {:4f}'.format(self.problem.bestScore()))
        self.logger.info('End Parameter Search \n')

if __name__ == '__main__':

    def black_box_function(x, y):
        
        return -x ** 2 - (y - 1) ** 2 + 1

    prb = Problem(black_box_function)

    bnd = {'x': ('uniform_float', (-3, 3)), 'y': ('uniform_float', (0, 2))}
    opt = Parzen(prb, bnd, Logger('test.log'))
    opt.run(n_iter=100)
