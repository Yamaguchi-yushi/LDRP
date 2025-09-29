import gym
import numpy as np
import yaml
from argparse import Namespace
from src.policy.pbs import PBS
#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'main'))) 
#from policy.policy import policy

env=gym.make("drp_env:drp-2agent_map_3x3-v2", state_repre_flag = "onehot_fov")

n_obs=env.reset()
#print("n_obs", n_obs, type(n_obs),)
#print("action_space", env.action_space)
#print("observation_space", env.observation_space)
actions = []
with open("./src/config/default.yaml", 'r') as file:
    config_dict = yaml.safe_load(file)
args = Namespace(**config_dict)

PBS_agent = PBS(args)
PBS_agent.culc_actions(n_obs, env)
print("initial actions", PBS_agent.schedule_actions)

for _ in range(50):
    env.render()

    #actions=tuple(map(int, input().split()))
    actions = PBS_agent.policy(n_obs, env)
    n_obs, reward, done, info = env.step(actions.pop(0))

    print(env.obs, env.obs_onehot, n_obs)
    #print("actions", actions, "reward", reward, done)
    #print("info", info)
    #print("n_obs", n_obs)
