from pygpg.sk import GPGRegressor as GPGR
import sympy as sp

hyper_params = [
    { # 1
     'd' : (3,), 'rci' : (0.0,),
    },
    { # 2
     'd' : (4,), 'rci' : (0.0, 0.1),
    },
    { # 2
     'd' : (5,), 'rci' : (0.0, 0.1,),
    },
    { # 1
     'd' : (6,), 'rci' : (0.1,),  'no_univ_exc_leaves_fos' : (True,),
    },
]

est = GPGR(t=2*60*60, g=-1, e=499500, tour=4, d=4,
        disable_ims=True, pop=1024, feat_sel=16,
        no_large_fos=True, no_univ_exc_leaves_fos=False,
        finetune=True, finetune_max_evals=500,
        bs=2048,
        fset='+,-,*,/,log,sqrt,sin,cos', cmp=0.0, rci=0.0,
        random_state=0
        )

def complexity(est):
  m = est.model
  c = 0
  for _ in sp.preorder_traversal(m):
    c+=1
  return c

def model(est):
    return str(est.model)
