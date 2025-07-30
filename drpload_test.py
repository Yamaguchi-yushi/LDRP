import gym
import numpy as np
#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'main'))) 
#from policy.policy import policy

env=gym.make("drp_env:drp-2agent_map_3x3-v2", state_repre_flag = "onehot_fov")
#"""
n_obs=env.reset()
#print("n_obs", n_obs, type(n_obs),)
#print("action_space", env.action_space)
#print("observation_space", env.observation_space)

for _ in range(50):
    env.render()

    actions=tuple(map(int, input().split()))
    n_obs, reward, done, info = env.step(actions)

    print(env.obs, env.obs_onehot, n_obs)
    #print("actions", actions, "reward", reward, done)
    #print("info", info)
    #print("n_obs", n_obs)
#"""
"""
steps = []
goal = []
timeup = []
for _ in range(100):
    n_obs=env.reset()
    #print("n_obs", n_obs, type(n_obs),)
    #print("action_space", env.action_space)
    #print("observation_space", env.observation_space)
    done = False
    while not done:
        #env.render()

        #actions=tuple(map(int, input().split()))
        actions = policy(n_obs, env)
        n_obs, reward, terminated_n, info = env.step(actions)

        done = all(terminated_n)
    steps.append(info["step"])
    goal.append(info["goal"])
    timeup.append(info["timeup"])
        #print("actions", actions, "reward", reward, done)
        #print("info", info)
        #print("n_obs", n_obs)

print("Average steps:", np.mean(steps))
print("Average goal:", np.mean(goal))
print("Average timeup:", np.mean(timeup))
"""