# Author:  DINDIN Meryll
# Date:    25/11/2019
# Project: optimizers

try: from optimizers.problem import *
except: from problem import *

class Optunas:

    def __init__(self, problem, boundaries, logger, seed=42):

        self.logger = logger
        self.logger.info('# Optuna Optimization')

        optuna.logging.disable_default_handler()

        self.problem = problem
        self.boundaries = boundaries
        self.study = optuna.create_study(sampler=TPESampler(seed=seed))

    def generateParams(self, trial):

        prm = dict()

        for key in sorted(self.boundaries.keys()):

            selection, bounds = self.boundaries[key]

            if selection == 'uniform_int':
                prm[key] = trial.suggest_int(key, bounds[0], bounds[1]+1)
            if selection == 'uniform_float': 
                prm[key] = trial.suggest_uniform(key, bounds[0], bounds[1])
            if selection == 'uniform_log': 
                prm[key] = trial.suggest_loguniform(key, bounds[0], bounds[1])
            if selection == 'choice':
                prm[key] = trial.suggest_categorical(key, bounds)

        return prm

    def evaluateParams(self, trial, iterations):

        self.problem.evaluate(self.generateParams(trial))
        arg = (len(self.problem.score), iterations, self.problem.score[-1])
        self.logger.info('{:3d}/{:3d} > Last Score : {:4f}'.format(*arg))

        return -self.problem.score[-1]

    def run(self, n_iter=100):

        if n_iter > 0: self.logger.info('Start optimization')
        self.study.optimize(partial(self.evaluateParams, iterations=n_iter), n_trials=n_iter)
        self.logger.info('Best Score : {:4f}'.format(self.problem.bestScore()))
        self.logger.info('End Optimization \n')

if __name__ == '__main__':

    def black_box_function(x, y):
        
        return -x ** 2 - (y - 1) ** 2 + 1

    prb = Problem(black_box_function)

    bnd = {'x': ('uniform_float', (-3, 3)), 'y': ('uniform_float', (0, 2))}
    opt = Optunas(prb, bnd, Logger('test.log'))
    opt.run(n_iter=100)
