# Author:  DINDIN Meryll
# Date:    27/08/2019
# Project: optimizers

try: from optimizers.evolutive import *
except: from evolutive import *

class Experiment:

    BAYESIAN_INIT = 50
    BAYESIAN_OPTI = 10
    PARZEN_NITERS = 60

    def __init__(self, name=str(int(time.time())), random_state=42):

        self._id = name
        self.dir = 'experiments/{}'.format(self._id)
        if not os.path.exists(self.dir): os.makedirs(self.dir, exist_ok=True)
        self.log = Logger('/'.join([self.dir, 'logs.log']))
        self.rnd = random_state

    def run(self, problem, optimization='bayesian', index=None, model_file=None):

        self.log.info('Launch training for {} model'.format(problem.model_id))
        self.log.info('Use {} concurrent threads'.format(problem.threads))
        self.log.info('Train on {} samples'.format(problem.x_t.shape[0]))
        self.log.info('Valid on {} samples\n'.format(problem.x_v.shape[0]))

        if optimization == 'bayesian':
            # Launch the Bayesian optimization
            opt = Bayesian(problem, problem.loadBoundaries(), self.log, seed=self.rnd)
            opt.run(n_init=self.BAYESIAN_INIT, n_iter=self.BAYESIAN_OPTI)

        if optimization == 'parzen':
            # Launch the Parzen optimization
            opt = Parzen(problem, problem.loadBoundaries(), self.log, seed=self.rnd)
            opt.run(n_iter=self.PARZEN_NITERS)

        # Serialize the configuration file
        cfg = {'strategy': 'single', 'model': problem.model_id, 'id': self._id}
        cfg.update({'objective': problem.objective, 'optimization': optimization})
        cfg.update({'random_state': self.rnd, 'threads': problem.threads})
        if optimization == 'bayesian': 
            cfg.update({'trial_init': self.BAYESIAN_INIT, 'trial_opti': self.BAYESIAN_OPTI})
        if optimization == 'parzen': 
            cfg.update({'trial_iter': self.PARZEN_NITERS})
        cfg.update({'best_score': problem.bestScore(), 'validation_metric': problem.metric})
        if index is None: nme = '/'.join([self.dir, 'config.json'])
        else: nme = '/'.join([self.dir, 'config_{}.json'.format(index)])
        with open(nme, 'w') as raw: json.dump(cfg, raw, indent=4, sort_keys=True)

        # Serialize parameters
        prm = problem.bestParameters()
        if index is None: nme = '/'.join([self.dir, 'params.json'])
        else: nme = '/'.join([self.dir, 'params_{}.json'.format(index)])
        with open(nme, 'w') as raw: json.dump(prm, raw, indent=4, sort_keys=True)

        # Serialize model if required
        if not model_file is None:
            mod = problem.bestModel(filename=model_file, random_seed=self.rnd)

    def getModel(self, model_file):

        return joblib.load(model_file)

    def getProbabilities(self, arrays, model_file):

        mod = self.getModel(model_file)
        # Get probabilities on given set
        return tuple([mod.predict_proba(arr) for arr in arrays])

    def evaluateModel(self, problem, axes=None):

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

            if axes is None:
                plt.figure(figsize=(18,4))
                grd = GridSpec(1, 3)

            if axes is None: ax0 = plt.subplot(grd[0, 0])
            else: ax0 = axes[0]

            crs = cm.Greens(sco)
            ax0.bar(np.arange(len(sco)), sco, width=0.4, color=crs)
            # Change the style of the axis spines
            ax0.spines['top'].set_color('none')
            ax0.spines['right'].set_color('none')
            ax0.spines['left'].set_smart_bounds(True)
            ax0.spines['bottom'].set_smart_bounds(True)
            for i,s in enumerate(sco): ax0.text(i-0.15, s-0.05, '{:1.2f}'.format(s))
            ax0.set_xticks(np.arange(len(sco)), lab)
            ax0.set_xlabel('metric')
            ax0.set_ylabel('percentage')

            if axes is None: ax1 = plt.subplot(grd[0, 1:])
            else: ax1 = axes[1]
            sns.heatmap(cfm, annot=True, fmt='d', axes=ax1, cbar=False, cmap="Greens")
            ax1.set_ylabel('y_true')
            ax1.set_xlabel('y_pred')

            if axes is None:
                plt.tight_layout()
                plt.show()

    def getImportances(self, problem, n_display=30, features=None, ax=None):

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

        # Numeric placeholder for the y axis
        rge = list(range(1, len(imp.index)+1))

        if ax is None: fig, ax = plt.subplots(figsize=(18,10))
        # Create for each feature an horizontal line 
        arg = {'color': 'salmon', 'alpha': 0.4, 'linewidth': 5}
        plt.hlines(y=rge,    xmin=0, xmax=imp.importance.values.astype('float'), **arg)
        # Create for each feature a dot at the level of the percentage value
        arg = {'markersize': 5, 'color': 'red', 'alpha': 0.3}
        plt.plot(imp.importance.values.astype('float'), rge, "o", **arg)

        # Set labels
        ax.set_xlabel('importance')
        ax.set_ylabel('feature')
        # Set axis
        plt.yticks(rge, imp.feature)
        # Change the style of the axis spines
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        plt.show()

    def dashboard(self, problem, n_display=30):

        def display_config(jsonpath, n_truncate, ax):

            with open(jsonpath) as raw: cfg, dic = json.load(raw), dict()
            for k,v in cfg.items():
                key = '_'.join(k.split('_')[:n_truncate])
                if isinstance(v, float): dic[key] = [np.round(v, 3)]
                else: dic[key] = [v]
            cfg = pd.DataFrame.from_dict(dic)
            
            return dtf_to_img(cfg, ax=ax)

        plt.figure(figsize=(18,15))
        gds = GridSpec(14, 3)

        ax0 = plt.subplot(gds[0,:])
        display_config('/'.join([self.dir, 'config.json']), 1, ax0)
        
        ax1 = plt.subplot(gds[1,:])
        display_config('/'.join([self.dir, 'params.json']), 3, ax1)
        
        ax2 = plt.subplot(gds[3:5,:])
        ax2.set_title('Training History')
        ax2.scatter(np.arange(len(problem.score)), problem.score, color='dodgerblue')
        ax2.plot(problem.score, linestyle='--', color='black', linewidth=0.5)
        ax2.set_ylabel('metric: {}'.format(problem.metric))
        ax2.set_xlabel('iterations')
        ax2.spines['top'].set_color('none')
        ax2.spines['right'].set_color('none')
        ax2.spines['left'].set_smart_bounds(True)
        ax2.spines['bottom'].set_smart_bounds(True)

        ax3 = plt.subplot(gds[6:9,0])
        ax4 = plt.subplot(gds[6:9,1:])
        self.evaluateModel(problem, axes=(ax3,ax4))

        ax5 = plt.subplot(gds[10:,:])
        self.getImportances(problem, n_display=n_display, ax=ax5)

        plt.tight_layout()
        plt.show()

class WrapperCV:

    def __init__(self, x_t, x_v, y_t, name=str(int(time.time())), folds=3, random_state=42):

        self.x_t = x_t
        self.x_v = x_v
        self.y_t = y_t

        self.nme = name
        self.dir = 'experiments/{}'.format(self.nme)
        if not os.path.exists(self.dir): os.makedirs(self.dir, exist_ok=True)
        self.fls = folds
        self.rnd = random_state

    def run(self, model, objective, metric, scaler, threads=1, weights=False, optimization='bayesian', config=None):

        skf = StratifiedKFold(n_splits=self.fls, random_state=self.rnd, shuffle=True)
        # Prepare the predictions for stacking
        n_c = len(np.unique(self.y_t))
        p_t = np.full((self.x_t.shape[0], n_c), np.nan)
        p_v = np.full((self.x_v.shape[0], self.fls*n_c), np.nan)

        # Run the stratified cross-validation
        for idx, (i_t, i_v) in enumerate(skf.split(self.y_t, self.y_t)):

            # Apply selective scaling
            x_t = scaler.fit_transform(self.x_t[i_t])
            x_v = scaler.transform(self.x_t[i_v])
            vec = (x_t, x_v, self.y_t[i_t], self.y_t[i_v])
            # Build a prototype
            arg = {'threads': threads, 'weights': weights}
            prb = Prototype(*vec, model, objective, metric, **arg)
            # Build an experiment
            exp = Experiment(name=self.nme, random_state=self.rnd)
            # Override configuration if given
            if not config is None:
                for k, v in config.items(): setattr(exp, k.upper(), v)
            # Define a model prefix
            fle = '/'.join([exp.dir, '{}_{}.jb'.format(model, idx)])
            # Run the experiment
            exp.run(prb, optimization=optimization, index=idx, model_file=fle)
            exp.log.terminate()
            # Stack the probabilities
            prd = exp.getProbabilities([prb.x_v, scaler.transform(self.x_v)], fle)
            p_t[i_v,:] = prd[0]
            p_v[:,n_c*idx:n_c*(idx+1)] = prd[1]
            # Memory efficiency
            del arg, vec, prb, exp, fle, prd

        # Serialize resulting arrays
        np.save('/'.join([self.dir, 'proba_train.npy']), p_t)
        np.save('/'.join([self.dir, 'proba_valid.npy']), p_v)

if __name__ == '__main__':

    x,y = load_iris(return_X_y=True)
    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=0.2, shuffle=True)
    prb = WrapperCV(x_t, x_v, y_t, y_v)
    cfg = {'BAYESIAN_INIT': 3, 'BAYESIAN_OPTI': 0}
    pip = Pipeline([('mms', MinMaxScaler()), ('sts', StandardScaler())])
    prb.run('ETS', 'classification', 'acc', pip, threads=6, weights=False, config=cfg)
