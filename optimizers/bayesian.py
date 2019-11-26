# Author:  DINDIN Meryll
# Date:    01/03/2019
# Project: optimizers

try: from optimizers.problem import *
except: from problem import *

class Bayesian:

    def __init__(self, problem, boundaries, logger, seed=42):

        self.logger = logger
        self.logger.info('# Bayesian Optimization')

        self.problem = problem
        self.boundaries = boundaries
        self.seed = seed
        self.random = np.random.RandomState(seed=seed)

        arg = {'alpha': 1e-6, 'n_restarts_optimizer': 25}
        self.gauss = GPR(kernel=Matern(nu=2.5), normalize_y=True, **arg)

        self.bounds = []
        self.forced = {}
        self.bound_keys = []

        for key in sorted(boundaries.keys()):
            selection, bound = boundaries[key] 
            if selection == 'choice': 
                self.forced[key] = bound[0]
            else: 
                self.bounds.append(np.asarray(boundaries[key][1]))
                self.bound_keys.append(key)

        self.bounds = np.asarray(self.bounds)
        self.bound_keys = np.asarray(self.bound_keys)

    def generateParams(self):

        prm = dict()

        for key in sorted(self.boundaries.keys()):

            selection, bounds = self.boundaries[key]

            if selection == 'uniform_int': 
                prm[key] = self.random.randint(bounds[0], bounds[1]+1)
            if selection == 'uniform_float': 
                prm[key] = self.random.uniform(bounds[0], bounds[1])
            if selection == 'uniform_log': 
                prm[key] = np.exp(self.random.uniform(np.log(bounds[0]), np.log(bounds[1])))

        return prm

    def paramsToArray(self, parameters):

        keys = list(set(sorted(parameters.keys())) & set(self.bound_keys))
        return np.asarray([parameters[key] for key in keys])

    def arrayToParams(self, vector):

        prm = dict()
        for idx, key in enumerate(sorted(self.bound_keys)):
            selection, _ = self.boundaries[key]
            if selection.split('_')[-1] == 'int':
                prm[key] = int(vector[idx])
            else: 
                prm[key] = vector[idx]

        return handle_integers(prm)

    def mapSpace(self, iterations):

        arg = {'size': (iterations, self.bounds.shape[0])}
        return self.random.uniform(self.bounds[:,0], self.bounds[:,1], **arg)

    def randomEvaluation(self, iterations):

        if iterations > 0: self.logger.info('Random Parameters')
        for idx in range(iterations):
            self.problem.evaluate(self.generateParams(), forced=self.forced, random_seed=self.seed)
            arg = (idx+1, iterations, self.problem.score[-1])
            self.logger.info('{:3d}/{:3d} > Last Score : {:4f}'.format(*arg))

        self.logger.info('Best Score : {:4f}'.format(self.problem.bestScore()))
        self.logger.info('End Random Search \n')

    def fitKernel(self):

        space = np.asarray([self.paramsToArray(ele) for ele in self.problem.cache])
        self.gauss.fit(space, self.problem.score)

    def estimation(self, space, method='ucb', kappa=2.576, xi=0.0):

        warnings.simplefilter('ignore')
        mean, std = self.gauss.predict(space, return_std=True)

        if method == 'ucb':
            return mean + kappa * std

        if method == 'e_i':
            y_max = max(self.problem.score)
            z = (mean - y_max - xi) / std
            return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

        if method == 'poi':
            y_max = max(self.problem.score)
            z = (mean - y_max - xi) / std
            return norm.cdf(z)

    def generateSuggestion(self, n_warm=int(1e3), n_iter=int(1e2)):

        self.fitKernel()

        swarm = self.mapSpace(n_warm)
        score = self.estimation(swarm)
        x,acq = swarm[score.argmax()], score.max()
        siter = self.mapSpace(n_iter)

        warnings.simplefilter('ignore')
        
        for initial_guess in siter:

            arg = {'bounds': self.bounds, 'method': 'L-BFGS-B'}
            fun = lambda x: -self.estimation(x.reshape(1,-1))
            res = minimize(fun, initial_guess, **arg)

            if not res.success: continue
            if acq is None or -res.fun[0] >= acq: x,acq = res.x, -res.fun[0]

        return self.arrayToParams(np.clip(x, self.bounds[:, 0], self.bounds[:, 1]))

    def suggestEvaluation(self, iterations):

        if iterations > 0: self.logger.info('Suggested Parameters')
        for idx in range(iterations):
            self.problem.evaluate(self.generateSuggestion(), forced=self.forced, random_seed=self.seed)
            arg = (idx+1, iterations, self.problem.score[-1])
            self.logger.info('{:3d}/{:3d} > Last Score : {:4f}'.format(*arg))

        self.logger.info('Best Score : {:4f}'.format(self.problem.bestScore()))
        self.logger.info('End Suggested Search \n')

    def run(self, n_init=100, n_iter=20):

        self.randomEvaluation(n_init)
        self.suggestEvaluation(n_iter)        

if __name__ == '__main__':

    def black_box_function(x, y):
        
        return -x ** 2 - (y - 1) ** 2 + 1

    prb = Problem(black_box_function)

    bnd = {'x': ('uniform_float', (-3, 3)), 'y': ('uniform_float', (0, 2))}
    opt = Bayesian(prb, bnd, Logger('test.log'))
    opt.run(n_init=80, n_iter=20)
