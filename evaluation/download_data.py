from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import ioh
except:
    print("IOH not installed")


import ast
import json
import os
from typing import Dict, Union

import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def lazy_json_load(filename):
    with open(filename, "r") as file:
        data = json.load(file)

    return data


def get_folder(run_info: pd.Series) -> str:
    config = run_info["config"]
    if "slurm_id" not in config:
        folder = None
    else:
        slurm_id = config["slurm_id"]
        hydra_id = slurm_id.split("_")[-1]
        outdir = config["outdir"]
        date = "/".join(config["wandb"]["id"].split("_")[:-1])
        folder = Path(outdir) / date / hydra_id
        folder = str(folder)
    return folder


def add_folders_to_runs(runs_df: pd.DataFrame):
    folders = [get_folder(run) for index, run in runs_df.iterrows()]
    runs_df["outdir"] = folders
    return runs_df


def recover_traincfg_from_wandb(fn_wbcfg: str, to_dict: bool = False) -> DictConfig | Dict | None:
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


fn_config = ".hydra/config.yaml"
fn_wbsummary = "wandb/latest-run/files/wandb-summary.json"
fn_wbconfig = "wandb/latest-run/files/config.yaml"

runs_fn = "tmp/wandb_runs.pickle"
fn_rollout_data = "tmp/rollout_bbob.csv"


def scale(data):
    X = data["regret"].to_numpy(dtype=float)
    log_regret = np.log(X + 1e-10)
    xmax = log_regret.max()
    xmin = log_regret.min()
    # if xmin == -np.inf:
    #     xmin = -50
    def scaler(x):
        x = np.log(x + 1e-10)
        x = (x - xmin) / (xmax - xmin)
        return x

    data["regret_log_scaled"] = data["regret"].apply(scaler)
    return data


def download(group="paris2") -> pd.DataFrame:
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    filters = {
        # "config.env": env_name,
        # "config.experiment": "benchmarking",
        # "config.wandb.job_type": "train",
        # "config.context_sampler.n_samples": 1000,
        # "config.coco_instance.dimension": 2,
        "config.wandb.group": group,
        # "state": "finished",
    }
    metrics = ["eval/return", "train/global_step"]
    runs = api.runs("benjamc/dacbo", filters=filters)

    summary_list, config_list, name_list, metrics_list = [], [], [], []
    for run in tqdm(runs, total=len(runs)):
        # Check metrics first. If not all available, do not append run
        rows = []
        for i, row in run.history(keys=metrics).iterrows():
            if all([metric in row for metric in metrics]):
                # df = df.append(row, ignore_index=True)
                rows.append(row)
            else:
                continue
        df = pd.DataFrame(rows)
        metrics_list.append(df)

        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
            "metrics": metrics_list,
        }
    )

    runs_df = add_folders_to_runs(runs_df)
    runs_df.to_pickle(runs_fn)
    print(f"Saved to {os.path.join(os.getcwd(), runs_fn)}")
    return runs_df


def scale_regret(df: pd.DataFrame) -> pd.DataFrame:
    groups = df.groupby(by=["bbob_function", "bbob_dimension", "bbob_instance"])
    new_df = []
    for group_id, group_df in groups:
        group_df = scale(group_df)
        new_df.append(group_df)
    df = pd.concat(new_df)
    return df


def add_regret(data: pd.DataFrame) -> pd.DataFrame:
    groups = data.groupby(by=["bbob_function", "bbob_dimension", "bbob_instance"])
    new_df = []
    for group_id, group_df in groups:
        bbob_function_id, bbob_dim, bbob_instance = group_id
        problem = ioh.get_problem(
            fid=bbob_function_id,
            instance=bbob_instance,
            dimension=bbob_dim,
            problem_type="BBOB",
        )
        optimum = problem.objective.y
        group_df["regret"] = group_df["reward"] - optimum
        new_df.append(group_df)
    df = pd.concat(new_df)
    return df


if __name__ == "__main__":
    runs_df = download()
