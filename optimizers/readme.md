# Challenger - Optimization

This submodule is focused on model optimization for fast prototyping in machine learning. Lots of hackathons, and lots of academic projects have been relying on this package as I was drafting it. Different objectives are handled and managed through it, but the general goal is certainly controlled experimentation, reproducibility and fast parametrization. For now, it handles two type of optimization for classic models: **PARZEN** (inspired by [_hyperopt_](https://github.com/hyperopt/hyperopt)) and **BAYESIAN** (inspired by [_github_](https://github.com/fmfn/BayesianOptimization)) optimization. I also combined those two methods in an **EVOLUTIVE** method, as parzen, which is tree-based, easily handles choices, while bayesian makes greater sense on comparable / continuous variables.

## Run an experiment

Based on my own experience, here is what you want to control:
- The **validation_split**, which is defined by you;
- The type of **model** you want to train, among a variety of scikit-learn models;
- The given **objective** of the optimization, either _classification_ or _regression_;
- The **metric** defining improvement;
- Whether you want to use **weights** to balance the training;
- The number of **concurrent threads** you allow the model to use;
- The **random_state** for reproducibility;

```python
# Load the dataset
x,y = load_iris(return_X_y=True)
# Split it in training and validation
x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
# Build a representation of the problem
prb = Prototype(x_t, x_v, y_t, y_v, 'ETS', 'classification', 'acc', threads=cpu_count(), weights=False)
# Instantiate and run a one-shot learning based on the given configuration
exp = Experiment()
exp.single(prb, random_state=42)
```

## Defined models

- **ETS**: Extra Trees
- **RFS**: Random Forests
- **LGB**: LightGBM
- **XGB**: XGBoost
- **SVM**: Support Vector Machines
- **CAT**: CatBoost
- **SGD**: Stochastic Gradient Descent
- **LGR**: Logistic Linear Regression

## Logging strategy

For each experiment, a subfolder will be created in 'current/path/experiments', which is timestamped. In that subfolder, you will find what is needed to exactly reproduce the experiment, but also extract the desired results.

### _logs.log_
```bash
02:35:49 | INFO    > Launch training for LGB model
02:35:49 | INFO    > Use 32 concurrent threads

02:35:59 | INFO    > # Bayesian Optimization
02:35:59 | INFO    > Random Parameters
02:36:58 | INFO    >   1/ 50 > Last Score : 0.503104
02:37:53 | INFO    >   2/ 50 > Last Score : 0.502027

03:36:42 | INFO    > Best Score : 0.524959
03:36:42 | INFO    > End Suggested Search 
```

### _config.json_
```bash
{
    "best_score": 0.5249594625897614,
    "id": "1565645738",
    "model": "LGB",
    "optimization": "bayesian",
    "random_state": 42,
    "strategy": "single",
    "test_size": 0.33,
    "threads": 32,
    "trial_init": 50,
    "trial_opti": 0,
    "validation_metric": "acc"
}
```

### _params.json_
```bash
{
    "colsample_bytree": 0.7703175608050532,
    "learning_rate": 0.052529610914953374,
    "max_depth": 5,
    "min_child_samples": 25,
    "min_child_weight": 0.0846362160256195,
    "n_estimators": 645,
    "num_leaves": 43,
    "reg_alpha": 1.696542444552246e-07,
    "reg_lambda": 1.5161591075671732
}
```