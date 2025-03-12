# writ_rl

RL models and Gym environment for the Word Rule Inference Task (WRIT). 

Using feature-weighted RL models a la [Niv et al. 2015](https://www.jneurosci.org/lookup/doi/10.1523/JNEUROSCI.2978-14.2015)

```
├── README.md
├── requirements.txt
├── setup.py
├── tests
│   └── test_feature_rl.py # tests for some feature RL code
└── writ_rl
    ├── __init__.py
    └── models
        ├── __init__.py
        ├── RL_constants.py # priors and other model constants
        ├── decisionmaker.py # base class for RL to inherit from
        ├── env.py # environment for 
        ├── feature_rl.py # the model classes
        ├── fit_rl.py # model fitting
        ├── gen_rpe.py # backing out model RPEs for human data
        ├── lci.py # Latent cause inference models
        └── utils.py Utilities for running quick simulations with the models
```
