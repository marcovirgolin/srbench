from pyGPGOMEA import GPGOMEARegressor as GPG

hyper_params = [
    {
        'tournament' : (4,8,16), 
        'subcross' : (0.9,),
        'submut' : (0.1,), 
    },
    {
        'tournament' : (4,8,16), 
        'subcross' : (0.5,),
        'submut' : (0.5,), 
    },
]

est = GPG(gomea=False, time=2*60*60, generations=-1, popsize=1024, 
        maxsize=50,
        evaluations=500000, initmaxtreeheight=6,
        functions='+_-_*_p/_plog_sqrt_sin_cos',
        ims=False, erc=True, linearscaling=True, silent=True, parallel=False)

def complexity(est):
    return est.get_n_nodes()

def model(est):
    return est.get_model()
