# Author:  DINDIN Meryll
# Date:    01/03/2019
# Project: optimizers

try: 
    from optimizers.parzen import *
    from optimizers.bayesian import *
except: 
    from parzen import *
    from bayesian import *

class Evolutive:

    def __init__(self, problem, boundaries, logger, seed=42):

        self.seed = seed
        self.logger = logger
        self.problem = problem
        self.boundaries = boundaries

    def extractBestChoices(self):

        forced = dict()
        params = self.problem.bestParameters()

        for key in sorted(self.boundaries.keys()):

            selection, _ = self.boundaries[key]
            if selection == 'choice': forced[key] = params[key]

        return forced

    def choiceBoundaries(self, best_choices):

        forced = dict()
        for key in best_choices.keys(): 
            forced[key] = ('choice', [best_choices[key]])

        return forced

    def restrictProblem(self, best_choices):

        def matchParams(set1, set2):

            boolean = True

            for key in set1.keys(): 
                if set2[key] != set1[key]: boolean = False

            return boolean

        origin = len(self.problem.score)
        select = [matchParams(best_choices, ele) for ele in self.problem.cache]
        select = np.where(np.asarray(select))[0]

        self.problem.cache = [param for i, param in enumerate(self.problem.cache) if i in select]
        self.problem.score = [score for i, score in enumerate(self.problem.score) if i in select]

        return origin - len(select)

    def run(self, n_tpe=100, n_random=50, n_bayesian=20):

        arg = {'logger': self.logger, 'seed': self.seed}
        tpeoptim = Parzen(self.problem, self.boundaries, **arg)
        tpeoptim.run(n_iter=n_tpe)

        boundaries = self.boundaries.copy()
        best_choices = self.extractBestChoices()
        n_missing = self.restrictProblem(best_choices)
        boundaries.update(self.choiceBoundaries(best_choices))
        self.logger.info('Selection Implies {} Missing Simulations'.format(n_missing))

        arg = {'logger': self.logger, 'seed': self.seed}
        bayesian = Bayesian(self.problem, boundaries, **arg)
        bayesian.run(n_init=n_missing+n_random, n_iter=n_bayesian)

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

if __name__ == '__main__':

    # x,y = load_iris(return_X_y=True)
    # x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=0.2, shuffle=True)
    # prb = Prototype(x_t, x_v, y_t, y_v, 'ETS', 'classification', 'acc', threads=6, weights=False)
    # opt = Evolutive(prb, prb.loadBoundaries(), logger)
    # opt.run(n_tpe=20, n_random=60, n_bayesian=20)

    def black_box_function(x, y):
        return - x ** 2 - (y - 1) ** 2 + 1

    prb = Problem(black_box_function)
    bnd = {'x': ('uniform_float', (-3, 3)), 'y': ('uniform_float', (0, 2))}

    opt = Benchmark(prb, Logger('test.log'), boundaries=bnd)
    scores = opt.run(iterations=100)