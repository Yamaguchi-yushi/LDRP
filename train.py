import yaml
import gym
import sys
import datetime
import numpy as np
from argparse import Namespace
from src.task_assign.runner import Runner


if __name__ == "__main__":
    reward_list = {
        "goal": 100,
        "collision": -100,
        "wait": -10.,
        "move": -1,
    }

    with open("./src/config/default.yaml", 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Namespace(**config_dict)

    env_name = config.env_name#"drp_env:drp-" + str(config.drone_num) + "agent_map_" + config.map_name + "-v2"

    env = gym.make(
        env_name,
        state_repre_flag="onehot_fov",
        reward_list=reward_list,
        time_limit=config.time_limit,
        task_flag=True
    )
    """
    with open("./config/algo/" + config.algo + ".yaml", 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Namespace(**config_dict)
    """

    print("train start",datetime.datetime.now())
    runner = Runner(config, env, reward_list, training=True)
    runner.run()
    runner.finish()