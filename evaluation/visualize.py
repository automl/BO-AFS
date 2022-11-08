import ast
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rich import print as printr
from rich.progress import track
from tqdm import tqdm

# from wandb.sdk.wandb_helper import parse_config
from evaluation.run_evaluation import find_multirun_paths


def lazy_json_load(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    return data


folder_train = "/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/exp_sweep/2022-07-01/14-14-50_benchmark_train/"
folder_eval = "/home/benjamin/Dokumente/code/tmp/tntcomp/CARL/multirun/2022-07-06/17-21-44"
folder_eval = "/home/benjamin/Dokumente/code/tmp/DAC-BO/tmp/log/2022-07-15/17-04-33"
# folder_eval = "/home/benjamin/Dokumente/code/tmp/DAC-BO/tmp/log/2022-07-18/11-59-50"


def recover_traincfg_from_wandb(fn_wbcfg: str, to_dict: bool = False) -> Optional[Union[DictConfig, Dict]]:
    wbcfg = OmegaConf.load(fn_wbcfg)
    if not "traincfg" in wbcfg:
        return None
    traincfg = wbcfg.traincfg
    traincfg = OmegaConf.to_container(cfg=traincfg, resolve=False, enum_to_str=True)["value"]
    traincfg = ast.literal_eval(traincfg)
    traincfg = OmegaConf.create(traincfg)
    if to_dict:
        traincfg = OmegaConf.to_container(cfg=traincfg, resolve=True, enum_to_str=True)
    return traincfg


def load_wandb_table(fn: Union[str, Path]) -> pd.DataFrame:
    data = lazy_json_load(fn)
    data = pd.DataFrame(data=np.array(data["data"], dtype=object), columns=data["columns"])
    return data


reload: bool = True

fn_config = ".hydra/config.yaml"
fn_wbsummary = "wandb/latest-run/files/wandb-summary.json"
fn_wbconfig = "wandb/latest-run/files/config.yaml"
fn_rollout_data = Path("rolloutdata.csv")
if not fn_rollout_data.is_file():
    reload = True

paths = find_multirun_paths(result_dir=folder_eval)

if reload:
    data_list = []
    for i, p in tqdm(enumerate(paths)):
        p = Path(p)
        fn_cfg = p / fn_config
        fn_wbsum = p / fn_wbsummary
        fn_wbcfg = p / fn_wbconfig
        if not fn_wbcfg.is_file() or not fn_wbsum.is_file() or not fn_cfg.is_file():
            continue
        cfg = OmegaConf.load(fn_cfg)
        traincfg = recover_traincfg_from_wandb(fn_wbcfg)
        summary = lazy_json_load(fn_wbsum)

        if "average_return" in summary:
            average_return = summary["average_return"]
        else:
            average_return = None

        if average_return is None:
            continue

        path_to_table = fn_wbsum.parent / summary["rollout_data"]["path"]
        rollout_data = load_wandb_table(path_to_table)

        data_list.append(rollout_data)

        # if i == 2:
        #     break

    df = pd.concat(data_list)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(fn_rollout_data, index=False)
else:
    df = pd.read_csv(fn_rollout_data)


import matplotlib.pyplot as plt
import seaborn as sns


# Plot return
def plot_return(data: pd.DataFrame) -> None:
    group_keys = ["instance", "episode", "policy_name", "seed"]
    groups = data.groupby(group_keys)
    df_return = groups.apply(lambda x: np.sum(a=x["reward"])).reset_index().rename(columns={0: "return"})

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.boxplot(data=df_return, x="policy_name", y="return", ax=ax, palette="colorblind")
    ax.set_yscale("log")
    fig.set_tight_layout(True)
    plt.show()


plot_return(data=df)


# Plot action
def plot_action_over_steps(data: pd.DataFrame, hue: Optional[str] = None) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.lineplot(data=data, x="step", y="action", hue=hue, ax=ax)
    fig.set_tight_layout(True)
    plt.show()


plot_action_over_steps(data=df, hue="policy_name")
