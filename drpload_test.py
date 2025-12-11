import gym
import numpy as np
import yaml
import time
from argparse import Namespace
from src.policy.pbs import PBS
from src.all_policy import all_policy 
#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'main'))) 
#from policy.policy import policy
"""
env=gym.make("drp_env:drp-2agent_map_3x3-v2", 
             state_repre_flag = "onehot_fov", 
             start_ori_array = [0,1],
             goal_array = [3,6],
             task_flag = False)
"""
env=gym.make("drp_env:drp-3agent_map_5x4-v2", 
             state_repre_flag = "onehot_fov", 
             task_flag = False)
#"""

n_obs=env.reset()
print("n_obs", n_obs)
print(env.obs)
#print("action_space", env.action_space)
#print("observation_space", env.observation_space)

#PBSのtest用
#"""
actions = []
with open("./src/config/default.yaml", 'r') as file:
    config_dict = yaml.safe_load(file)
args = Namespace(**config_dict)

PBS_agent = PBS(args)
#PBS_agent.culc_actions(n_obs, env)

print("obs", env.start_ori_array, env.goal_array)
print("actions", PBS_agent.schedule_actions)


for _ in range(50):
    #env.render()
    #time.sleep(0.5)
    #input()
    print("step", env.step_account)

    actions = PBS_agent.policy(n_obs, env)
    n_obs, reward, done, info = env.step(actions)

    if all(done):
        if info["goal"]:
            print("goal!!!")
        elif info["timeup"]:
            print("timeup!!!")
        elif info["collision"]:
            print("collision!!!")

        break

"""

#tasklistのtest用
actions = []

#print("obs", env.start_ori_array, env.goal_array)

for _ in range(50):
    env.render()
    input()

    #actions=tuple(map(int, input().split()))
    #task = tuple(map(int, input().split()))
    actions, task = all_policy(n_obs, env)
    joint_action = {"pass": actions, "task": task}
    n_obs, reward, done, info = env.step(joint_action)

"""