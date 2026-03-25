import yaml
import gym
import sys
import numpy as np
from argparse import Namespace
import argparse
from runner import Runner


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

    if len(sys.argv) > 1:
        #map,agent,path,task
        config.map_name = sys.argv[1]
        config.agent_num = int(sys.argv[2])
        config.path_planner = sys.argv[3]
        config.task_assigner = sys.argv[4]

    env_name = "drp_env:drp-" + str(config.agent_num) + "agent_" + config.map_name + "-v2"
    #env_name = "drp_env:drp_safe-" + str(config.agent_num) + "agent_" + config.map_name + "-v2"

    env = gym.make(
        env_name,
        state_repre_flag="onehot_fov",
        reward_list=reward_list,
        time_limit=config.time_limit,
        task_flag=True,
        task_list=None,
    )
    """
    with open("./config/algo/" + config.algo + ".yaml", 'r') as file:
        config_dict = yaml.safe_load(file)
    config = Namespace(**config_dict)
    """
    runner = Runner(config, env, reward_list)
    runner.run()
    runner.finish()