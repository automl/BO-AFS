# Analyse Landscape on BBOB
> :warning: **These are notes.** :smile:

```bash
# evaluate
# debug
python evaluation/evaluate_manual.py +baseline=staticPI seed=3 coco_instance.function=10 coco_instance.dimension=2 wandb.debug=true

# run (2400 jobs)
python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,11)' 'coco_instance.function=range(1,25)' 'coco_instance.dimension=2,5' -m 

``` 
Runtime:
- 10 episodes with 2 dim: ~ 5min
- 10 episodes with 5 dim: ~50min



New iteration
- [x] EI, PI with different ratios
- [x] track configuration per step
- [ ] 20 seeds
- [x] ioh
- [x] fix initial design
  
```bash
python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,21)' 'coco_instance.function=range(1,25)' 'coco_instance.dimension=2,5' -m 
```
- [x] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,11)' 'coco_instance.function=range(1,12)' 'coco_instance.dimension=2' 'hydra.launcher.timeout_min=15' -m 
- [x] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(11,21)' 'coco_instance.function=range(1,12)' 'coco_instance.dimension=2' 'hydra.launcher.timeout_min=15' -m 
- [x] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,11)' 'coco_instance.function=range(12,25)' 'coco_instance.dimension=2' 'hydra.launcher.timeout_min=15' -m 
- [x] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(11,21)' 'coco_instance.function=range(12,25)' 'coco_instance.dimension=2' 'hydra.launcher.timeout_min=15' -m 


- [x] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,11)' 'coco_instance.function=range(1,12)' 'coco_instance.dimension=5' 'hydra.launcher.partition=cpu_short' 'hydra.launcher.timeout_min=45' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(11,21)' 'coco_instance.function=range(1,12)' 'coco_instance.dimension=5' 'hydra.launcher.timeout_min=45' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(1,11)' 'coco_instance.function=range(12,25)' 'coco_instance.dimension=5' 'hydra.launcher.timeout_min=45' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(11,21)' 'coco_instance.function=range(12,25)' 'coco_instance.dimension=5' 'hydra.launcher.timeout_min=45' -m 



## Rerun explore-exploit
[r] python evaluation/evaluate_manual.py '+baseline=glob(exploreexploit_*)' 'seed=range(1,21)' 'coco_instance.function=range(1,9)' 'coco_instance.dimension=2,5' -m 
[r] python evaluation/evaluate_manual.py '+baseline=glob(exploreexploit_*)' 'seed=range(1,21)' 'coco_instance.function=range(9,16)' 'coco_instance.dimension=2,5' -m 
[r] python evaluation/evaluate_manual.py '+baseline=glob(exploreexploit_*)' 'seed=range(1,21)' 'coco_instance.function=range(16,22)' 'coco_instance.dimension=2,5' -m 
[ ] python evaluation/evaluate_manual.py '+baseline=glob(exploreexploit_*)' 'seed=range(1,21)' 'coco_instance.function=range(22,25)' 'coco_instance.dimension=2,5' -m 



## Rerum 5d with more seeds
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(21,41)' 'coco_instance.function=range(1,7)' 'coco_instance.dimension=5' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(21,41)' 'coco_instance.function=range(7,13)' 'coco_instance.dimension=5' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(21,41)' 'coco_instance.function=range(13,19)' 'coco_instance.dimension=5' -m 
- [r] python evaluation/evaluate_manual.py '+baseline=glob(*)' 'seed=range(21,41)' 'coco_instance.function=range(19,25)' 'coco_instance.dimension=5' -m 



## Copy figures
scp sy2019-33-2:/home/benjamin/Dokumente/code/tmp/DAC-BO/evaluation/tmp/figures/* C:\Users\numina\Pictures\misc\paris_bbob\



Requirements
- ioh
- smac=1.4.0
- pandas
- hydra-core
- hydra-colorlog
- hydra-submitit-launcher
- matplotlib
- seaborn
- wandb
- gym==0.23.0
- rich
- tqdm
- PyBenchFCN
- jupyterlab
- tensorboard


## Bigger initial design
python evaluation/evaluate_manual.py '+baseline=glob(*)' 'coco_instance.function=range(1,25)' 'coco_instance.dimension=5' 'seed=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]' 'hydra.launcher.timeout_min=1440' 'wandb.debug=true' -m



python evaluation/evaluate_manual.py '+baseline=glob(*)' 'coco_instance.function=range(1,25)' 'coco_instance.instance=range(1,6)' 'coco_instance.dimension=5' 'seed=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]' 'hydra.launcher.timeout_min=1440' 'wandb.debug=true' -m
python evaluation/evaluate_manual.py '+baseline=glob(*)' 'coco_instance.function=range(1,25)' 'coco_instance.instance=range(1,6)' 'coco_instance.dimension=5' 'seed=[32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]' 'hydra.launcher.timeout_min=1440' 'wandb.debug=true' -m




# Train ELA - reg
conda activate bbobela
cd evaluation
python train_ela.py 'seed=range(70,80)' 'mode=regression,classification' 'hydra.launcher.timeout_min=120' 'hydra.launcher.partition=cpu_short' -m


reg (regret), clf, 3600, ensemble size 20
./exp_sweep_ela/2022-09-30/12-04-13


Regression:
600, regret log scaled
./exp_sweep_ela/2022-09-30/14-01-34
