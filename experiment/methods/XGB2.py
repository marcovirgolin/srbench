import xgboost
import numpy as np

hyper_params = [
    {
        'n_estimators' : (10, 100, 1000, 10000,),
        'learning_rate' : (0.0001, 0.001, 0.01, 0.1,),
        'gamma' : (0, 0.1, 0.2, 0.3, 0.4,),
        'subsample' : (0.5, 1,),
        'max_depth' : (2, 4, 6,),
    },
]

est=xgboost.XGBRegressor()

def complexity(est):
    return np.sum([m.count(':') for m in est._Booster.get_dump()])
model = None
