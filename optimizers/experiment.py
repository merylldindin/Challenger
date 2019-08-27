# Author:  DINDIN Meryll
# Date:    27/08/2019
# Project: optimizers

try: from optimizers.evolutive import *
except: from evolutive import *

class Experiment:

    BAYESIAN_INIT = 50
    BAYESIAN_OPTI = 10
    PARZEN_NITERS = 60

    def __init__(self, name=str(int(time.time()))):

        self._id = name
        self.dir = 'experiments/{}'.format(self._id)
        if not os.path.exists(self.dir): os.makedirs(self.dir, exist_ok=True)
        self.log = Logger('/'.join([self.dir, 'logs.log']))

    def single(self, problem, random_state=42, method='bayesian'):

        self.log.info('Launch training for {} model'.format(problem.model_id))
        self.log.info('Use {} concurrent threads\n'.format(problem.threads))

        if method == 'bayesian':
            # Launch the Bayesian optimization
            opt = Bayesian(problem, problem.loadBoundaries(), self.log, seed=random_state)
            opt.run(n_init=self.BAYESIAN_INIT, n_iter=self.BAYESIAN_OPTI)

        if method == 'parzen':
            # Launch the Parzen optimization
            opt = Parzen(problem, problem.loadBoundaries(), self.log, seed=random_state)
            opt.run(n_iter=self.PARZEN_NITERS)

        # Serialize the configuration file
        cfg = {'strategy': 'single', 'model': problem.model_id, 'id': self._id}
        cfg.update({'objective': problem.objective, 'optimization': method})
        cfg.update({'random_state': random_state, 'threads': problem.threads})
        if method == 'bayesian': 
            cfg.update({'trial_init': self.BAYESIAN_INIT, 'trial_opti': self.BAYESIAN_OPTI})
        if method == 'parzen': 
            cfg.update({'trial_iter': self.PARZEN_NITERS})
        cfg.update({'best_score': problem.bestScore(), 'validation_metric': problem.metric})
        nme = '/'.join([self.dir, 'config.json'])
        with open(nme, 'w') as raw: json.dump(cfg, raw, indent=4, sort_keys=True)

        # Serialize parameters
        prm = problem.bestParameters()
        nme = '/'.join([self.dir, 'params.json'])
        with open(nme, 'w') as raw: json.dump(prm, raw, indent=4, sort_keys=True)

    def saveModel(self, problem, random_state=42):

        # Fit and save the best model
        _ = problem.bestModel(filename='/'.join([self.dir, 'model.jb']), random_seed=random_state)

    def getModel(self):

        return joblib.load('/'.join([self.dir, 'model.jb']))

    def evaluateModel(self, problem):

        # Retrieve the model
        model = self.getModel()

        # Predict on the validation set
        y_v = problem.y_v
        y_p = model.predict(problem.x_v)

        if problem.objective == 'classification':

            lab = ['accuracy', 'f1 score', 'precision', 'recall', 'kappa']
            sco = np.asarray([
                accuracy_score(y_v, y_p),
                f1_score(y_v, y_p, average='weighted'),
                precision_score(y_v, y_p, average='weighted'),
                recall_score(y_v, y_p, average='weighted'),
                cohen_kappa_score(y_v, y_p)])
            cfm = confusion_matrix(y_v, y_p)

            plt.figure(figsize=(18,4))
            grd = gridspec.GridSpec(1, 3)

            arg = {'y': 1.05, 'fontsize': 14}
            plt.suptitle('General Classification Performances for Experiment {}'.format(self._id), **arg)

            ax0 = plt.subplot(grd[0, 0])
            crs = cm.Greens(sco)
            plt.bar(np.arange(len(sco)), sco, width=0.4, color=crs)
            for i,s in enumerate(sco): plt.text(i-0.15, s-0.05, '{:1.2f}'.format(s))
            plt.xticks(np.arange(len(sco)), lab)
            plt.xlabel('metric')
            plt.ylabel('percentage')

            ax1 = plt.subplot(grd[0, 1:])
            sns.heatmap(cfm, annot=True, fmt='d', axes=ax1, cbar=False, cmap="Greens")
            plt.ylabel('y_true')
            plt.xlabel('y_pred')

            plt.tight_layout()
            plt.show()

    def getImportances(self, problem, n_display=30, features=None):

        # Retrieve the model
        model = self.getModel()

        # Retrieve the feature names
        if features is None: 
            sze = int((np.log(problem.x_t.shape[1]) / np.log(10)) + np.finfo(np.float32).eps)
            sze = r'fea_{:0' + str(sze) + r'}'
            features = [sze.format(i) for i in range(problem.x_t.shape[1])]

        imp = model.feature_importances_ / np.sum(model.feature_importances_)
        imp = pd.DataFrame(np.vstack((features, imp)).T, columns=['feature', 'importance'])
        imp = imp.sort_values(by='importance', ascending=False)
        imp = imp[:n_display]

        # Set the style of the axes and the text color
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['xtick.color'] = '#333F4B'
        plt.rcParams['ytick.color'] = '#333F4B'
        plt.rcParams['text.color'] = '#333F4B'

        # Numeric placeholder for the y axis
        rge = list(range(1, len(imp.index)+1))

        fig, ax = plt.subplots(figsize=(18,10))
        # Create for each feature an horizontal line 
        arg = {'color': 'salmon', 'alpha': 0.4, 'linewidth': 5}
        plt.hlines(y=rge, xmin=0, xmax=imp.importance.values.astype('float'), **arg)
        # Create for each feature a dot at the level of the percentage value
        arg = {'markersize': 5, 'color': 'red', 'alpha': 0.3}
        plt.plot(imp.importance.values.astype('float'), rge, "o", **arg)

        # Set labels
        ax.set_xlabel('importance', fontsize=14, fontweight='black', color = '#333F4B')
        ax.set_ylabel('')
        # Set axis
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.yticks(rge, imp.feature)
        # Change the style of the axis spines
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        # Set the spines position
        ax.spines['bottom'].set_position(('axes', -0.04))
        ax.spines['left'].set_position(('axes', 0.015))
        plt.show()

    def submitPredictions(self, test_set):

        # Retrieve the best model
        model = self.getModel()

        y_p = model.predict(test_set)
        y_p = pd.DataFrame(np.vstack((np.arange(len(test_set)), y_p)).T, columns=['id', 'label'])
        y_p = y_p.set_index('id')
        y_p.to_csv('/'.join([self.dir, 'predictions.csv']))
