"""Collates json-formatted results, cleans them up and saves them as .feather
files."""
# Author: William La Cava, williamlacava@gmail.com
# SRBENCH
# License: GPLv3

################################################################################
# Black-box problems
################################################################################
import pandas as pd
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import sys
import sympy as sp
import time
import timeout_decorator

rdir = '../results_blackbox/'

if len(sys.argv) > 1:
    rdir = sys.argv[1]
else:
    print('no rdir provided, using',rdir)
print('reading results from  directory', rdir)
    
symbolic_algs = [
    'AFP', 
    'AFP_FE',
    'BSR',
    'DSR',
    'FFX',
    'FEAT',
    'EPLEX',
    'GP-GOMEA',
    'GP-GOMEAv2 (RT)',
    'GP-GOMEAv2 (LT)',
    'gplearn',
    'gpg',
    'ITEA', 
    'MRGP', 
    'Operon',
    'SBP-GP',
    'AIFeynman',
    'PySR',
]
nongp_algs = [
    'BSR',
    'DSR',
    'AIFeynman'
]
gp_algs = [
    'AFP', 
    'AFP_FE',
    'FFX',
    'FEAT',
    'EPLEX',
    'GP-GOMEA',
    'GP-GOMEAv2 (RT)',
    'GP-GOMEAv2 (LT)',
    'gplearn',
    'ITEA', 
    'MRGP', 
    'Operon',
    'SBP-GP',
    'PySR',
    'gpg',
]
##########
# load data from json
##########
frames = []
comparison_cols = [
    'dataset',
    'algorithm',
    'random_state',
    'time_time',
    'model_size',
    'symbolic_model',
    'mae_train',
    'mse_train',
    'r2_train',
    'r2_test',
    'mse_test',
    'mae_test',
    'params'
]
fails = []
import pdb
for f in tqdm(glob(rdir + '/*/*.json')):
    if 'cv_results' in f: 
        continue
    # leave out symbolic data
    if 'feynman_' in f or 'strogatz_' in f:
        continue
    # leave out LinearReg, Lasso (we have SGD with penalty)
    if any([m in f for m in ['LinearRegression','Lasso','EHCRegressor']]):
        continue
    # leave out ClassicGP (slow because contains performance-over-time)
    if any([m in f for m in ['ClassicGP']]):
        continue
    try: 
        r = json.load(open(f,'r'))
        if isinstance(r['symbolic_model'],list):
#             print(f)
            sm = ['B'+str(i)+'*'+ri for i, ri in enumerate(r['symbolic_model'])]
            sm = '+'.join(sm)
            r['symbolic_model'] = sm
            
        sub_r = {k:v for k,v in r.items() if k in comparison_cols}
    #     df = pd.DataFrame(sub_r)
        frames.append(sub_r) 
    #     print(f)
    #     print(r.keys())
    except Exception as e:
        fails.append([f,e])
        pass
    
print(len(fails),'fails:',fails)
# df_results = pd.concat(frames)
df_results = pd.DataFrame.from_records(frames)
df_results['params_str'] = df_results['params'].apply(str)
df_results = df_results.drop(columns=['params'])
#df_results = df_results[df_results.algorithm != 'ClassicGP']
##########
# cleanup
##########
df_results = df_results.rename(columns={'time_time':'training time (s)'})
df_results.loc[:,'training time (hr)'] = df_results['training time (s)']/3600
# remove regressor from names
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('Regressor','')) 
#Rename SGD to Linear
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: 'Linear' if x=='SGD' else x)
# rename sembackpropgp to SBP
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('sembackpropgp','SBP-GP'))
# rename FE_AFP to AFP_FE
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('FE_AFP','AFP_FE'))
# rename GPGOMEA to GP-GOMEA
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('GPGOMEA','GP-GOMEA'))
# rename gpgLT to GP-GOMEA_v2 (LT)
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('gpgLT','gpg'))
# rename gpg2LT to GP-GOMEA_v2.1 (LT)
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('gpg2LT','GP-GOMEAv2.1 (LT)'))
# rename gpg to GP-GOMEA_v2 (RT)
df_results['algorithm'] = df_results['algorithm'].apply(lambda x: x.replace('gpgRT','gpg (RT)'))
# add modified R2 with 0 floor
df_results['r2_zero_test'] = df_results['r2_test'].apply(lambda x: max(x,0))
# label friedman ddatasets
df_results.loc[:,'friedman_dataset'] = df_results['dataset'].str.contains('_fri_')
print('loaded',len(df_results),'results')
# additional metadata
df_results['symbolic_alg'] = df_results['algorithm'].apply(lambda x: x in symbolic_algs)

for col in ['algorithm','dataset']:
    print(df_results[col].nunique(), col+'s')


##########
# save results
##########
if os.path.exists("../results/black-box_results.feather"):
    new_algs = df_results["algorithm"].unique()
    # load prev results and remove new algs (in case they were saved already) 
    # in order to update them
    df_prev = pd.read_feather("../results/black-box_results.feather")
    df_prev = df_prev[~df_prev.algorithm.isin(new_algs)]
    df_prev = df_prev[~df_prev.algorithm.str.startswith("GP-GOMEAv2")]
    df_prev = df_prev[~df_prev.algorithm.str.startswith("gpg")]
    df_prev.reset_index(inplace=True, drop=True)
    df_results = pd.concat((df_prev, df_results))
    df_results.reset_index(inplace=True, drop=True)
    
df_results.to_feather('../results/black-box_results.feather')
print('results saved to ../results/black-box_results.feather')

########
print('mean trial count:')
print(df_results.groupby('algorithm')['dataset'].count().sort_values()
      / df_results.dataset.nunique())

########
# include Pierre's-MCTS & E2ET
for pierre_path in ["e2e","dgsr_mcts"]:
    df_pierre = pd.read_feather(f"../results/{pierre_path}.feather")
    if pierre_path == "e2e" and "model_size" not in df_pierre.columns:
        df_pierre["model_size"] = float('nan')
        numexpr_equivalence = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "pow": "**",
            "inv": "1/",
        }
        # process trees to get model size
        print("computing tree complexities for E2ET...")
        for index, row in df_pierre.iterrows():
            print(np.round(index/len(df_pierre)*100,3), "%", end="\r")
            expr = row["predicted_tree"]
            for old, new in numexpr_equivalence.items():
                expr=expr.replace(old, new)

            @timeout_decorator.timeout(2, timeout_exception=StopIteration)
            def timed_simplify(expr):
                return sp.simplify(expr)
            
            @timeout_decorator.timeout(2, timeout_exception=StopIteration)
            def timed_sympify(expr):
                return sp.sympify(expr)

            try:
                simpl_expr = timed_simplify(expr)
            except:
                try:
                    simpl_expr = timed_sympify(expr)
                except:
                    continue
            
            # count symbols
            def model_size(expr):
                c=0
                for _ in sp.preorder_traversal(expr):
                    c += 1
                return c
            df_pierre.loc[index, "model_size"] = model_size(simpl_expr)
        print("storing results E2ET incl size to feather...")
        df_pierre.to_feather(f"../results/{pierre_path}.feather")
        
    df_pierre["algorithm"] = "DGSR-MCTS" if pierre_path == "dgsr_mcts" else "E2ET"
    df_pierre["symbolic_alg"] = True
    df_pierre = df_pierre[~df_pierre["dataset"].str.contains("feynman")]
    df_pierre = df_pierre[~df_pierre["dataset"].str.contains("strogatz")]
    df_pierre.loc[:,'friedman_dataset'] = df_pierre['dataset'].str.contains('_fri_')
    #df_filtered = df_results[df_results["dataset"].isin(df_pierre["dataset"])]
    df_results = pd.concat((df_results, df_pierre))
    df_results.reset_index(inplace=True, drop=True)
    # stitch together
df_results.to_feather('../results/black-box_results_inclPierre.feather')

