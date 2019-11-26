# Author:  DINDIN Meryll
# Date:    25/11/2019
# Project: optimizers

try: 
    from optimizers.parzen import *
    from optimizers.bayesian import *
    from optimizers.optunas import *
except: 
    from parzen import *
    from bayesian import *
    from optunas import *

class Benchmark:

    def __init__(self, problem, logger, seed=42, boundaries=None):

        self.seed = seed
        self.logger = logger
        self.problem = problem
        if not (boundaries is None): self.boundaries = boundaries
        else: self.boundaries = self.problem.loadBoundaries()

    def run(self, iterations=100):

        scores = dict()

        opt = Parzen(self.problem, self.boundaries, self.logger, seed=self.seed)
        opt.run(n_iter=iterations)
        scores['parzen'] = {'scores': self.problem.score, 'bestParams': self.problem.bestParameters()}
        self.problem.reset()

        opt = Bayesian(self.problem, self.boundaries, self.logger, seed=self.seed)
        opt.run(n_init=iterations, n_iter=0)
        scores['random'] = {'scores': self.problem.score, 'bestParams': self.problem.bestParameters()}
        self.problem.reset()

        opt = Bayesian(self.problem, self.boundaries, self.logger, seed=self.seed)
        opt.run(n_init=int(0.8*iterations), n_iter=int(0.2*iterations))
        scores['bayesian'] = {'scores': self.problem.score, 'bestParams': self.problem.bestParameters()}
        self.problem.reset()

        opt = Evolutive(self.problem, self.boundaries, self.logger, seed=self.seed)
        opt.run(n_tpe=int(0.2*iterations), n_random=int(0.6*iterations), n_bayesian=int(0.2*iterations))
        scores['evolutive'] = {'scores': self.problem.score, 'bestParams': self.problem.bestParameters()}
        self.problem.reset()

        return scores
