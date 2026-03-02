import gym
import numpy as np
import yaml
import time
import sys
from argparse import Namespace
from src.policy.pbs import PBS
from src.all_policy import all_policy 
#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'main'))) 
#from policy.policy import policy
#"""
env=gym.make("drp_env:drp-3agent_map_3x3-v2", 
             state_repre_flag = "onehot_fov", 
             start_ori_array = [0, 8, 7],
             goal_array = [4, 3, 1],
             task_flag = False)
"""
#pbsのtest用の場合，設定ファイルも変える
#[0, 8, 7] [4, 3, 1]
#[3, 6, 8] [4, 2, 5]
env=gym.make("drp_env:drp-3agent_map_3x3-v2", 
             state_repre_flag = "onehot_fov", 
             task_flag = False)
"""

for _ in range(1000):

    n_obs=env.reset()
    #print("n_obs", n_obs)
    #print(env.obs)
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

    for i in range(1000):
        #env.render()
        #time.sleep(0.5)
        #input()
        #print("step", env.step_account)
        if i == 5:
            PBS_agent.schedule_actions = []

        actions = PBS_agent.policy(n_obs, env)

        #print("actions", actions)
        n_obs, reward, done, info = env.step(actions)

        #"""
        if 1==1:
            #print("############################")
            print("step:", i)
            print("current_start:", env.current_start)
            print("current_goal:", env.current_goal)
            print("goal_array:", env.goal_array)
            print("agents_action:", actions)
            print("############################")
        #"""

        if all(done):
            if info["goal"]:
                print("goal!!!")
            elif info["timeup"]:
                print("timeup!!!")
            elif info["collision"]:
                print("collision!!!")
                sys.exit()
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