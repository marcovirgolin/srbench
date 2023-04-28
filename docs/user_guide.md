# User Guide

## Installation

We have provided a [conda environment](environment.yml), [configuration script](configure.sh) and [installation script](install.sh) that should make installation straightforward.
We've currently tested this on Ubuntu and CentOS. 
Steps:

1. Install the conda environment:

```bash
conda env create -f environment.yml
conda activate srbench
```

2. Install the benchmark methods:

```bash
bash install.sh
```

3. Download the PMLB datasets:

```bash
git clone https://github.com/EpistasisLab/pmlb/ [/path/to/pmlb/]
cd /path/to/pmlb
git lfs pull
```

For Docker users,
```bash
docker build --pull --rm -f "Dockerfile" -t srbench:latest "."
```

## Reproducing the benchmark results

Experiments are launched from the `experiments/` folder via the script `analyze.py`.
The script can be configured to run the experiment in parallel locally, on an LSF job scheduler, or on a SLURM job scheduler. 
To see the full set of options, run `python analyze.py -h`. 

**WARNING**: running some of the commands below will submit tens of thousands of experiments. 
Use accordingly. 

### Black-box experiment
After installing and configuring the conda environment, the complete black-box experiment can be started via the command:

```bash
python analyze.py /path/to/pmlb/datasets -n_trials 10 -results ../results_blackbox -time_limit 48:00
```

### Ground-truth experiment

**Train the models**: we train the models subject to varying levels of noise using the options below. 

```bash
# submit the ground-truth dataset experiment. 

for data in "/path/to/pmlb/datasets/strogatz_" "/path/to/pmlb/datasets/feynman_" ; do # feynman and strogatz datasets
    for TN in 0 0.001 0.01 0.1; do # noise levels
        python analyze.py \
            $data"*" \ #data folder
            -results ../results_sym_data \ # where the results will be saved
            -target_noise $TN \ # level of noise to add
            -sym_data \ # for datasets with symbolic models
            -n_trials 10 \
            -m 16384 \ # memory limit in MB
            -time_limit 9:00 \ # time limit in hrs
            -job_limit 100000 \ # this will restrict how many jobs actually get submitted.
            -tuned # use the tuned version of the estimators, rather than performing hyperparameter tuning.
        if [ $? -gt 0 ] ; then
            break
        fi
    done
done
```

**Symbolic Assessment**: Following model training, the trained models are assessed for symbolic equivalence with the ground-truth data-generating processes. 
This is handled in [assess_symbolic_model.py](experiment/assess_symbolic_model.py). 
Use `analyze.py` to generate batch calls to this function as follows:

```bash
# assess the ground-truth models that were produced using sympy
for data in "/path/to/pmlb/datasets/strogatz_" "/path/to/pmlb/datasets/feynman_" ; do # feynman and strogatz datasets
    for TN in 0 0.001 0.01 0.1; do # noise levels
        python analyze.py \
            -script assess_symbolic_model \
            $data"*" \ #data folder
            -results ../results_sym_data \ # where the results will be saved
            -target_noise $TN \ # level of noise to add
            -sym_data \ # for datasets with symbolic models
            -n_trials 10 \
            -m 8192 \ # memory limit in MB
            -time_limit 1:00 \ # time limit in hrs
            -job_limit 100000 \ # this will restrict how many jobs actually get submitted.
            -tuned # use the tuned version of the estimators, rather than performing hyperparameter tuning.
        if [ $? -gt 0 ] ; then
            break
        fi
    done
done
```

**Output**: next to each `.json` file, an additional file named `.json.updated` is saved with the symbolic assessment included. 

### Post-processing

Navigate to the [postprocessing](postprocessing) folder to begin postprocessing the experiment results. 
The following two scripts collate the `.json` files into two `.feather` files to share results more easily. 
You will notice these `.feather` files are loaded to generate figures in the notebooks. 
They also perform some cleanup like shortening algorithm names, etc.

```
python collate_blackbox_results.py
python collate_groundtruth_results.py
```

**Visualization**

- [groundtruth_results.ipynb](postprocessing/groundtruth_results.ipynb): ground-truth results comparisons
- [blackbox_results.ipynb](postprocessing/blackbox_results.ipynb): ground-truth results comparisons
- [statistical_comparisons.ipynb](postprocessing/statistical_comparisons.ipynb): post-hoc statistical comparisons
- [pmlb_plots](postprocessing/pmlb_plots.ipynb): the [PMLB](https://github.com/EpistasisLab/pmlb) datasets visualization 


## Using your own datasets

To use your own datasets, you want to check out / modify read_file in read_file.py: https://github.com/cavalab/srbench/blob/4cc90adc9c450dad3cb3f82c93136bc2cb3b1a0a/experiment/read_file.py

If your datasets follow the convention of https://github.com/EpistasisLab/pmlb/tree/master/datasets, i.e. they are in a pandas DataFrame with the target column labelled "targert", you can call `read_file` directly just passing the filename like you would with any of the PMLB datasets. 
The file should be stored and compressed as a `.tsv.gz` file. 
