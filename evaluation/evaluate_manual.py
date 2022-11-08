from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print
from rich import print as printr

from dacbo.create_env import (evaluate, get_baseline_policy, id_generator,
                              make_dacbo_env, set_seed_everywhere)
from dacbo.utils.typing import Vector

base_dir = os.getcwd()


def add_meta_data(data: pd.DataFrame | pd.Series, meta_data: dict) -> pd.DataFrame | pd.Series:
    if type(data) == pd.DataFrame:
        for k, v in meta_data.items():
            data[k] = [v] * len(data)
    else:
        for k, v in meta_data.items():
            data[k] = v

    return data


def entry_to_list(entry: list | Any) -> list:
    if type(entry) == ListConfig or type(entry) == list:
        entries = list(entry)
    else:
        entries = [entry]
    return entries


@hydra.main("configs", "base")
def main(cfg: DictConfig):
    seed = cfg.seed
    seeds = entry_to_list(seed)
    bbob_functions = entry_to_list(cfg.coco_instance.function)
    bbob_instances = entry_to_list(cfg.coco_instance.instance)
    bbob_dimensions = entry_to_list(cfg.coco_instance.dimension)

    init_dir = Path(HydraConfig.get().run.dir)

    # cfg.seed = seed
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

    unique_id = id_generator()
    outdir = init_dir  # / unique_id
    # outdir.mkdir(parents=True, exist_ok=True)
    # os.chdir(outdir)

    hydra_job = (
        os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
        + "_"
        + os.path.basename(HydraConfig.get().run.dir)
    )
    cfg.wandb.id = hydra_job + "_" + unique_id

    # traincfg = OmegaConf.load(str(Path(cfg.results_path) / ".hydra" / "config.yaml"))

    # wandbdir = Path(cfg.results_path) / "wandb"

    run = wandb.init(
        id=cfg.wandb.id,
        resume="allow",
        mode="offline" if cfg.wandb.debug else None,
        project=cfg.wandb.project,
        job_type=cfg.wandb.job_type,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        dir=os.getcwd(),
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
    )
    hydra_cfg = HydraConfig.get()
    command = f"{hydra_cfg.job.name}.py " + " ".join(hydra_cfg.overrides.task)
    if not OmegaConf.is_missing(hydra_cfg.job, "id"):
        slurm_id = hydra_cfg.job.id
    else:
        slurm_id = None
    wandb.config.update({"command": command, "slurm_id": slurm_id}, allow_val_change=True)

    data_list = []
    initial_design_list = []
    for seed in seeds:
        for bbob_function in bbob_functions:
            for bbob_instance in bbob_instances:
                for bbob_dimension in bbob_dimensions:
                    cfg.seed = seed
                    set_seed_everywhere(cfg.seed)
                    cfg.coco_instance.function = bbob_function
                    cfg.coco_instance.dimension = bbob_dimension
                    cfg.coco_instance.instance = bbob_instance
                    printr(cfg)

                    if cfg.baseline is not None:
                        budget = cfg.coco_instance.dimension * cfg.dacbo.benchmark.budget_multiplier
                        cfg.dacbo.benchmark.budget = budget
                        policy_type = cfg.baseline
                        if "policy" in cfg:
                            policy = list(cfg.policy)
                            # Cap policy to actual policy length in order not to confuse logging
                            # Actually it does not matter how long the sequence is because we will stop iterating over
                            # the policy as soon the environment signals done.
                            policy = policy[:budget]
                            cfg.policy = policy
                        else:
                            policy = get_baseline_policy(
                                policy_type=policy_type,
                                budget=budget,
                                seed=cfg.seed,
                                cfg=cfg,
                            )
                        cfg.policy_id = str(policy)

                    # ----------------------------------------------------------------------
                    # Instantiate environment
                    # ----------------------------------------------------------------------
                    env = make_dacbo_env(cfg=cfg)

                    # ----------------------------------------------------------------------
                    # Log experiment
                    # ----------------------------------------------------------------------
                    print(OmegaConf.to_yaml(cfg))
                    print(env)
                    print(f"Observation Space: ", env.observation_space)
                    print(f"Action Space: ", env.action_space)
                    output_dir = os.getcwd()
                    print("Output directory:", output_dir)

                    # ----------------------------------------------------------------------
                    # Evaluate
                    # ----------------------------------------------------------------------
                    data = evaluate(
                        policy=policy,
                        env=env,
                        n_eval_episodes=cfg.n_eval_episodes,
                        seed=cfg.seed,
                        policy_name=cfg.policy_name,
                        policy_id=cfg.policy_id,
                    )
                    initial_design = pd.Series({"initial_design": data["initial_design"].iloc[0]})
                    del data["initial_design"]

                    # Seed already is added
                    meta_data = {
                        "bbob_function": bbob_function,
                        "bbob_instance": bbob_instance,
                        "bbob_dimension": bbob_dimension,
                        "seed": seed,
                    }
                    data = add_meta_data(data, meta_data)
                    avg_return = data["reward"].mean()
                    data_list.append(data)

                    initial_design = add_meta_data(initial_design, meta_data)
                    initial_design = pd.DataFrame(initial_design).T
                    initial_design_list.append(initial_design)

    data = pd.concat(data_list).reset_index(drop=True)
    initial_design_df = pd.concat(initial_design_list).reset_index(drop=True)
    wandb.log(
        {
            "average_return": avg_return,
            "rollout_data": wandb.Table(dataframe=data),
            "initial_design": wandb.Table(dataframe=initial_design_df),
        }
    )

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    run.finish()

    return avg_return


if __name__ == "__main__":
    main()
