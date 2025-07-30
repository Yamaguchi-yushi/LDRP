import gym
from gym import error, spaces, utils
import numpy as np
import sys
import copy
import yaml
import os
from enum import Enum

from drp_env.EE_map import MapMake
from drp_env.drp_env import DrpEnv

class TaskEnv(DrpEnv):
	def step(self, joint_action):
		ri_tmp = []

		obs, ri_array, self.terminated, info = super().step(joint_action)

		for i in range(self.agent_num):
			ri_array[i] += ri_tmp[i]
	
		return obs, ri_array, self.terminated, info
	

	def reset(self, seed=None, options=None):
		obs = super().reset()
		return obs
	
	def valid_action_mask(self):
		pass