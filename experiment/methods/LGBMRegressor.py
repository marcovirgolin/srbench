import lightgbm
import numpy as np

hyper_params = {
        'n_estimators' : (10, 100, 1000, 10000,),
        'learning_rate' : (0.0001, 0.001, 0.01, 0.1,),
        'subsample' : (0.5, 0.75, 1,),
        'boosting_type' : ('gbdt', 'dart', 'goss',),
        'max_depth' : (2,4,6,),
    }

est=lightgbm.LGBMRegressor(
                           max_depth=6,
                           deterministic = True,
                           force_row_wise = True
                          )

def complexity(est):
    return np.sum([x['num_leaves'] for x in est._Booster.dump_model()['tree_info']])

model = None
