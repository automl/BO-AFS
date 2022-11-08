# Towards Automated Design of Bayesian Optimization via Exploratory Landscape Analysis
This is the repository for the submission.

## Installation
Create a fresh conda environment.
```bash
conda create -n afs python=3.10
conda activate afs
```

Download repo, switch to branch and install.
```bash
git clone https://github.com/automl-private/DAC-BO.git
cd DAC-BO
git checkout select_AFs
pip install -e .
```


> :warning: **If you want to run `evaluation/train_ela.py`**, you need to install `auto-sklearn` which is
> in conflict with the `ioh` package. Therefore you need to use the env you set up before for the
> schedule rollouts and then switch to a new env with `auto-sklearn` installed in order to find a suitable
> model pipeline.


## Instructions

Our motivation is to make Bayesian Optimization (BO) even more sample-efficient and performant.
For this we dynamically set BO's hyperparameters (HPs) or components.
As a starter we chose to dynamically switch the acquisition function (AF).
Available choices are EI (more explorative) and PI (more exploitative).

We create seven manual schedules composed of those two acquisition functions and evaluate them on the BBOB functions from the COCO benchmark (5d).
Then, we learn which schedule to use based on the initial design.
For this we compute ELA features on the initial design and train a naive Random Forest to
regress the performance of the individual schedules based on the ELA features.
We select the AF schedule with with the best predicted performance per run and compare
the choice to the virtual best solver (oracle), our AF selector and the manual schedules
on a held out test set.

Size of initial design: 10d = 50
Number of surrogate-based evaluations: 20d: 100
Number of runs/seeds: 60



Creating the insights requires following steps:
1. Rollout schedules on all BBOB functions (run BO with dynamic AF schedule)
2. Extract ELA features for initial design
3. Train prediction regressor
4. Predict which schedule to use
5. Plot 😊

### Schedules
* static (EI): only EI
* static (PI): only PI
* random: random EI or PI
* round robin: EI, PI, EI, PI, EI, PI, ...
* explore-exploit: first EI, then PI. switch after certain percentage of surrogate-based budget
  * 0.25, 0.5, 0.75

Down below you find rough instructions to reproduce the results.

### Rollout Schedules
Use script `evaluation/evaluate_manual.py`.
In particalur, use these commands:
```bash
python evaluation/evaluate_manual.py '+baseline=glob(*)' 'coco_instance.function=range(1,25)' 'coco_instance.dimension=5' 'seed=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]' 'wandb.debug=true' -m
```
This creates a hydra job array deployed on slurm. You can also use a local launcher (submitit local).

### Collect Rollout Data
Use `evaluation/analyse_metabbob.ipynb`, cells belonging to "Paths" and "Data Collection and Conversion".

### Extract ELA Features
The ELA features are extracted with the R package [flacco](https://github.com/kerschke/flacco).
Before they can be calculated with `evaluation/compute_ELA_features.R`, the initial design data set needs to be transformed with
`evaluation/reformat_initial_design.ipynb`.
It is assumed that the initial design data set `initial_design.csv` lies in the same folder.

### Train Prediction Regressor and Predict
See `evaluation/train_ela_slick.ipynb` ("Train Acquisiton Function Selector").

### Create plots
See `evaluation/train_ela_slick.ipynb` ("Plot").
* Violinplot of final regret of the different manual schedules + VBS + AFS
* Convergence plot (regret over time)
* Ranks

