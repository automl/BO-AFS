from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import coax
import glob
import json
import os
import pickle
import random
import string
from datetime import datetime
from functools import partial
from importlib.metadata import metadata
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Optional, Tuple, Union

import hydra
import numpy as onp
import numpy as np
import pandas as pd
import wandb
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.estimators import AutoSklearnEstimator
from autosklearn.regression import AutoSklearnRegressor
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from rich import print as printr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dacbo.utils.typing import Vector


def save_askl(model: AutoSklearnEstimator, path: str) -> str:
    fn = Path(path) / "automl_model.pickle"
    with open(fn, "wb") as file:
        pickle.dump(model, file)
    return str(fn)


def load_askl(path: str) -> AutoSklearnEstimator:
    fn = Path(path) / "automl_model.pickle"
    with open(fn, "rb") as file:
        automl_loaded = pickle.load(file)
    return automl_loaded


def save_metadata(data: dict, path: str) -> str:
    fn = Path(path) / "metadata.json"
    with open(fn, "w") as file:
        json.dump(data, file)
    return fn


def load_metadata(path: str) -> dict:
    fn = Path(path) / "metadata.json"
    with open(fn, "r") as file:
        data = json.load(file)
    return data


def get_schedules(data: pd.DataFrame) -> list[str]:
    schedules = list(data["policy_name"].unique())
    schedules.sort()
    return schedules


def get_schedule_id(schedule: str, schedules: list[str]) -> int:
    schedules.sort()
    return schedules.index(schedule)


def evaluate(y_true, y_pred, metric_funs, metric_funs_kwargs: list[dict] = None) -> dict[str, float]:
    test_performance = {}
    for i, metric_fun in enumerate(metric_funs):
        kwargs = {}
        if metric_funs_kwargs is not None:
            kwargs = metric_funs_kwargs[i]
        p = metric_fun(y_true=y_true, y_pred=y_pred, **kwargs)
        test_performance[metric_fun.__name__] = p
    return test_performance


def build_dataset(
    fn_ela_features: str | pd.DataFrame,
    fn_rollout_data: str | pd.DataFrame,
    performance_at_step: int = 100,
    mode: str = "regression",
    performance_name: str = "regret",
) -> tuple[np.array, np.array]:
    if type(fn_ela_features) == str:
        ela_features = pd.read_csv(fn_ela_features)
    else:
        ela_features = fn_ela_features
    if "Unnamed: 0" in ela_features:
        del ela_features["Unnamed: 0"]
    if type(fn_rollout_data) == str:
        performance_data = pd.read_csv(fn_rollout_data)
    else:
        performance_data = fn_rollout_data
    performance_data = performance_data[performance_data["step"] == performance_at_step - 1]
    n_schedules = performance_data["policy_name"].nunique()
    schedules = get_schedules(data=performance_data)
    n_points = len(ela_features)
    index_cols = ["bbob_function", "bbob_instance", "seed"]
    if "bbob_dim" in ela_features.columns:
        index_cols.append("bbob_dim")
    n_cols = len(ela_features.columns)
    ela_feature_cols = [c for c in ela_features.columns if c not in index_cols]
    n_ela_features = n_cols - len(index_cols)
    X = np.zeros((n_points, n_ela_features))
    y_dim = n_schedules
    dtype = float

    if mode == "classification":
        y_dim = 1
        dtype = int
    Y = np.zeros((n_points, y_dim), dtype=dtype)
    for i, (index, row) in enumerate(ela_features.iterrows()):
        dim = row.get("bbob_dim", 5)
        fid = row["bbob_function"]
        inst = row["bbob_instance"]
        seed = row["seed"]

        # ELA features
        vals = row[ela_feature_cols].to_numpy()
        X[i] = vals

        # Performance
        perf_subset = performance_data[
            (performance_data["bbob_dimension"] == dim)
            & (performance_data["bbob_function"] == fid)
            & (performance_data["bbob_instance"] == inst)
            & (performance_data["seed"] == seed)
        ]
        # sort by policy name so that y values always are in correct order
        assert len(perf_subset) == n_schedules, f"{len(perf_subset)} != {n_schedules}, {fid}, {perf_subset}"
        perf_subset = perf_subset.sort_values(by="policy_name")

        if mode == "regression":
            perf_value = perf_subset[performance_name]
        else:
            idx = perf_subset[performance_name].argmin()
            perf_value = perf_subset["policy_name"].iloc[idx]
            perf_value = int(get_schedule_id(perf_value, schedules))
            # print(perf_subset["policy_name"].iloc[idx], perf_value, schedules)

        Y[i] = perf_value
    return X, Y


def train_askl(cfg: DictConfig):
    mode = cfg.mode
    cv = cfg.cv
    seed = cfg.seed
    test_size = cfg.test_size
    # scoring = "mse"
    ensemble_size = cfg.ensemble_size
    time_left_for_this_task = cfg.time_left_for_this_task
    per_run_time_limit = cfg.per_run_time_limit
    performance_at_step = cfg.performance_at_step

    include = dict(cfg.include)
    include = {k: list(v) for k, v in include.items()}

    X, Y = build_dataset(
        fn_ela_features=cfg.fn_ela_features,
        fn_rollout_data=cfg.fn_rollout_data,
        performance_at_step=performance_at_step,
        mode=mode,
        performance_name=cfg.performance_name,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    exclude = None
    metric = cfg.metric
    memory_limit = 4000
    n_jobs = 1
    n_folds_cv = cv
    dt_str = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    tmp_folder = Path(cfg.save_dir).resolve() / "logs"
    save_dir = "."
    ensemble_kwargs = {"ensemble_size": ensemble_size}

    if mode == "regression":
        askl_class = AutoSklearnRegressor
    else:
        askl_class = AutoSklearnClassifier

    print(mode, askl_class)
    automl = askl_class(
        time_left_for_this_task=time_left_for_this_task,
        per_run_time_limit=per_run_time_limit,
        tmp_folder=tmp_folder,
        delete_tmp_folder_after_terminate=False,
        ensemble_kwargs=ensemble_kwargs,
        include=include,
        exclude=exclude,
        metric=metric,
        memory_limit=memory_limit,
        n_jobs=n_jobs,
        seed=seed,
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": n_folds_cv},
        logging_config=None,
    )
    automl.fit(X_train, y_train, dataset_name=f"BBOB_{performance_at_step}")
    # print(automl.leaderboard())
    # pprint(automl.show_models(), indent=4)
    # predictions = automl.predict(X_test)
    # print("R2 score:", r2_score(y_test, predictions))
    # print(automl.get_configuration_space(X_train, y_train))

    automl.refit(X_train, y_train)
    model_fn = save_askl(model=automl, path=save_dir)
    metadata = {
        "seed": seed,  # can recover the train test split
    }
    save_metadata(metadata, save_dir)
    print(f"Saved model to {model_fn}.")


base_dir = os.getcwd()


@hydra.main("config_ela", "base", version_base=None)
def main(cfg: DictConfig):
    seed = cfg.seed

    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

    # unique_id = id_generator()
    # hydra_job = (
    #     os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
    #     + "_"
    #     + os.path.basename(HydraConfig.get().run.dir)
    # )
    # cfg.wandb.id = hydra_job + "_" + unique_id
    hydra_cfg = HydraConfig.get()
    command = f"{hydra_cfg.job.name}.py " + " ".join(hydra_cfg.overrides.task)
    if not OmegaConf.is_missing(hydra_cfg.job, "id"):
        slurm_id = hydra_cfg.job.id
    else:
        slurm_id = None

    cfg.save_dir = HydraConfig.get().run.dir
    train_askl(cfg=cfg)

    return None


if __name__ == "__main__":
    main()
