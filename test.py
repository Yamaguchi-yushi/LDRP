import warnings
warnings.filterwarnings("ignore")

import yaml
from itertools import product
import gym
import sys
import numpy as np
from argparse import Namespace
import argparse
import torch
from runner import Runner
import unicodedata


def _width(s):
    # 表示したときの幅。日本語 (全角) は英字 2 文字分
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in s)


def _pad(s, w, right=False):
    sp = " " * max(0, w - _width(s))
    return sp + s if right else s + sp


def print_summary_table(title, columns):
    """行 = 指標・列 = 条件の表を出す。見出しは 2 段 (上: 到着 / 下: 再割当の有無).

    columns: [((上の見出し, 下の見出し), [(指標名, 値の文字列), ...]), ...]
    上の見出しが同じ列が続くときは、上の段をまとめて 1 つにする。
    """
    if not columns:
        return
    names = [name for name, _ in columns[0][1]]
    label_w = max(_width(n) for n in names) + 2
    col_w = [max([_width(bottom)] + [_width(v) for _, v in rows]) + 2
             for (top, bottom), rows in columns]

    # 上の見出しが同じ列のかたまり [(上の見出し, 先頭の列, 最後の列), ...]
    groups = []
    for k, ((top, _), _) in enumerate(columns):
        if groups and groups[-1][0] == top:
            groups[-1][2] = k
        else:
            groups.append([top, k, k])
    # かたまりの幅が上の見出しより狭ければ、最後の列を広げる
    for top, i, j in groups:
        lack = _width(top) + 6 - sum(col_w[i:j + 1])
        if lack > 0:
            col_w[j] += lack

    top_line = " " * label_w
    for top, i, j in groups:
        span = sum(col_w[i:j + 1])
        text = " " + top + " "
        dash = span - 2 - _width(text)          # 左の 2 文字は列の間の空白
        top_line += "  " + "-" * (dash // 2) + text + "-" * (dash - dash // 2)
    print("\n" + title, flush=True)
    print(top_line)
    print(" " * label_w + "".join(_pad(bottom, w, right=True)
                                  for ((_, bottom), _), w in zip(columns, col_w)))
    for i, name in enumerate(names):
        print(_pad(name, label_w) + "".join(_pad(rows[i][1], w, right=True)
                                           for (_, rows), w in zip(columns, col_w)), flush=True)


if __name__ == "__main__":
    reward_list = {
        "goal": 100,
        "collision": -100,
        "wait": -10,
        "move": -1,
    }

    with open("./src/config/default.yaml", 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Namespace(**config_dict)

    if len(sys.argv) > 1:
        #map,agent,path,task,[method_tag],[mat_model_agent_num]
        config.map_name = sys.argv[1]
        config.agent_num = int(sys.argv[2])
        config.path_planner = sys.argv[3]
        config.task_assigner = sys.argv[4]
        for tok in sys.argv[5:]:
            if tok == "":
                continue
            if "=" in tok:
                key, val = tok.split("=", 1)
                if key == "model_seed":
                    config.model_seed = int(val)
                elif key == "use_safe_env":
                    config.use_safe_env = val.lower() in ("1", "true", "yes") 
                elif key == "use_dynamic_agents":
                    config.use_dynamic_agents = val.lower() in ("1", "true", "yes")
                elif key == "arrival":
                    config.eval_arrivals = val.split(",")
                elif key == "allow_reassign":
                    config.eval_allow_reassign = [v.lower() in ("1", "true", "yes") for v in val.split(",")]
                else:
                    raise ValueError(f"Unknown key in argument: {key}")
            elif tok in ("base", "reassign"):
                config.reassign_before_pickup = tok
            elif tok.isdigit():
                config.mat_model_agent_num = int(tok)
            else:
                config.method_tag = tok

    use_safe_env = bool(getattr(config, "use_safe_env", True))
    prefix = "drp_safe-" if use_safe_env else "drp-"
    env_name = f"drp_env:{prefix}{config.agent_num}agent_{config.map_name}-v2"
    config.env_name = env_name
    print(f"[test] env={env_name} method_tag={getattr(config, 'method_tag', '') or '(none)'}", flush=True) 

    # Optionally forward LaRe-Path params from config (no-op when use_lare_path=false).
    lare_path_keys = [
        "use_lare_path",
        "use_lare_path_training",
        "lare_path_factor_dim",
        "lare_path_decoder_hidden_dim",
        "lare_path_decoder_n_layers",
        "lare_path_use_transformer",
        "lare_path_transformer_heads",
        "lare_path_transformer_depth",
        "lare_path_buffer_capacity",
        "lare_path_min_buffer",
        "lare_path_update_freq",
        "lare_path_batch_size",
        "lare_path_lr",
        "use_pretrained_lare_path",
        "pretrained_lare_path_model_name",
        "use_finetuning_lare_path",
        "finetuning_lare_path_model_name",
        "lare_path_autosave",
        "lare_path_autosave_path",
        "lare_path_save_dir",
        "lare_path_save_freq_steps",
        # LaRe-Task (System B)
        "use_lare_task",
        "use_lare_task_training",
        "lare_task_factor_dim",
        "lare_task_decoder_hidden_dim",
        "lare_task_decoder_n_layers",
        "lare_task_buffer_capacity",
        "lare_task_min_buffer",
        "lare_task_update_freq",
        "lare_task_batch_size",
        "lare_task_lr",
        "use_pretrained_lare_task",
        "pretrained_lare_task_model_name",
        "use_finetuning_lare_task",
        "finetuning_lare_task_model_name",
        "lare_task_autosave",
        "lare_task_autosave_path",
        "lare_task_save_dir",
        "lare_task_save_freq_steps",
        # タスク到着プロセスのランダム化 (学習用) / エピソード単位 seed (評価用)
        "randomize_task_arrival",
        "mmpp_ratio",
        "rand_p_min",
        "rand_p_max",
        "episode_seed_base",
    ]
    lare_kwargs = {k: getattr(config, k) for k in lare_path_keys if hasattr(config, k)}

    dynamic_agent_keys = [
        "use_dynamic_agents",
        "randomize_initial_active",
        "min_active_agents",
        "max_active_agents",
        "initial_active_num",
        "exclude_station_from_tasks",
    ]
    dynamic_agent_kwargs = {k: getattr(config, k) for k in dynamic_agent_keys if hasattr(config, k)}

    # path_planner が PBS のときだけ pbs_mode=True にする.
    # PBS は待機 agent の予定も path 計画に反映するため current_goal を非 None に
    # 保つ必要があるが、それ以外 (QMIX/IQL/VDN/MAA2C) では None のままにして
    # SafeEnv の保護を機能させる. 詳細は CLAUDE.md「SafeEnv と PBS のトレードオフ」.
    pbs_mode = (getattr(config, "path_planner", "") == "pbs")

    model_tag = getattr(config, "reassign_before_pickup", "base")

    training = bool(getattr(config, "train_task_assigner", False))
    if training and config.task_assigner != "ppo":
        raise ValueError("train_task_assigner is True but task_assigner is not 'ppo'.")
    
    if training:
        arrivals = [config.task_arrival]
        reassign_flags = [bool(config.allow_reassign_before_pickup)]
    else:
        arrivals = list(config.eval_arrivals)
        reassign_flags = list(config.eval_allow_reassign)
    for arrival in arrivals:
        if arrival.startswith("bernoulli") and ":" not in arrival:
            raise ValueError(f"'{arrival}' needs a probability, e.g. bernoulli:0.05")

    summary_columns = []   # ログの最後に出すまとめの表の列 (条件ごと)

    for arrival, reassign_flag in product(arrivals, reassign_flags):
        print(f"\n########## model={model_tag} arrival={arrival} allow_reassign_before_pickup={reassign_flag}##########", flush=True)
        mode, _, p = arrival.partition(":")
        config.task_arrival = mode
        if p:
            config.task_density = float(p)
        arrival_kwargs = {k: getattr(config, k) for k in 
                          ("task_arrival", "task_density", "task_p_high", "task_p_low", "task_switch_prob")
                          if hasattr(config, k)}
        config.allow_reassign_before_pickup = reassign_flag
        np.random.seed(config.seed if training else config.eval_seed) #シード値を固定するため
        torch.manual_seed(config.seed)
        env = gym.make(
            env_name,
            state_repre_flag="onehot_fov",
            reward_list=reward_list,
            time_limit=config.time_limit,
            task_flag=True,
            task_list=None,
            pbs_mode=pbs_mode,
            allow_reassign_before_pickup=reassign_flag,
            **lare_kwargs,
            **dynamic_agent_kwargs,
            **arrival_kwargs
        )
        """
        with open("./config/algo/" + config.algo + ".yaml", 'r') as file:
            config_dict = yaml.safe_load(file)
        config = Namespace(**config_dict)
        """
        runner = Runner(config, env, reward_list, training=training)
        runner.run()
        runner.finish()
        if getattr(runner, "summary", None):
            top = arrival.replace("bernoulli:", "bernoulli p=")
            bottom = "reassign" if reassign_flag else "no-reassign"
            summary_columns.append(((top, bottom), runner.summary))

    print_summary_table(
        f"[summary] {config.map_name} / {config.agent_num} agents / {config.path_planner} / "
        f"{config.task_assigner} / trained: {model_tag} / seed{getattr(config, 'model_seed', 0)}"
        f"   (top: task arrival / bottom: reassign at execution)",
        summary_columns)
