import gym
from gym import error, spaces, utils
import numpy as np
import sys
import copy
import yaml
import os
import math
from enum import Enum

from drp_env.EE_map import MapMake
from drp_env.drp_env import DrpEnv

class SafeEnv(DrpEnv):
	def _predict_pos (self, i, action_i, avail_i):
		cur = [self.obs[i][0], self.obs[i][1]]
		if action_i not in avail_i:
			return cur
		tgt = list(self.pos[int(action_i)])
		if tgt[0] == cur[0] and tgt[1] == cur[1]:
			return cur
		x, y = tgt[0] - cur[0], tgt[1] - cur[1]
		d = math.sqrt(x**2 + y**2)
		if d > self.speed:
			return [round(cur[0] + self.speed * x / d, 2), round(cur[1] + self.speed * y / d, 2)]
		return [round(tgt[0], 2), round(tgt[1], 2)]

		
	def reset(self):
		self.safety_intervention_count = 0
		return super().reset()

	def step(self, joint_action):
		task_assign = None
		if isinstance(joint_action, dict):
			task_assign = joint_action.get("task", None)
			joint_action = joint_action.get("pass", joint_action)

		if not hasattr(self, "safety_intervention_count"):
			self.safety_intervention_count = 0

		avail = [self._get_avail_agent_actions(k, self.n_actions)[1] for k in range(self.agent_num)]

		i = 0
		do = True

		while do:
			do = False
			for i in range(self.agent_num):
				if getattr(self, "use_dynamic_agents", False) and not self.active[i]:
					continue

				# one agent can reach the station node at the same time
				if getattr(self, "use_dynamic_agents", False) and self.pending_off[i] \
					and self.current_goal[i] == None:
					st = self.goal_array[i]
					busy = any(
						k != i and self.active[k] and (int(self.current_start[k]) == st 
									or (self.pending_off[k] and self.goal_array[k] == st
									and self.current_goal[k] != None))
									for k in range(self.agent_num))
					if busy:
						joint_action[i] = self.current_start[i]
						continue
						
				#act8，他のエージェントと向かう先が同じ場合
				#自分がノード上にいる時
				if self.current_goal[i] == None:
					for j in range(self.agent_num):
						if getattr(self, "use_dynamic_agents", False) and not self.active[j]:
							continue
						if j != i and joint_action[i] == joint_action[j]:
							if joint_action[i] != self.current_start[i]:
								self.safety_intervention_count += 1
								joint_action[i] = self.current_start[i]
								do = True #条件が変わる可能性があるため，もう一度ループを回す
							break

				#act9，正面衝突
				#自分がノード上にいる時
				if self.current_goal[i] == None:
					for j in range(self.agent_num):
						if getattr(self, "use_dynamic_agents", False) and not self.active[j]:
							continue
						if j != i and (joint_action[j] == self.current_start[i] and joint_action[i] == self.current_start[j]):
							if joint_action[i] != self.current_start[i]:
								self.safety_intervention_count += 1
								joint_action[i] = self.current_start[i]
								do = True
							break

			if not do:
				pred = [self._predict_pos(k, joint_action[k], avail[k]) for k in range(self.agent_num)]
				for i in range(self.agent_num):
					if getattr(self, "use_dynamic_agents", False) and not self.active[i]:
						continue
					hit = False
					for j in range(self.agent_num):
						if getattr(self, "use_dynamic_agents", False) and not self.active[j]:
							continue
						if j != i and math.dist(pred[i], pred[j]) < self.colli_distan_value:
							hit = True
							break
					if hit and joint_action[i] != self.current_start[i]:
						self.safety_intervention_count += 1
						joint_action[i] = self.current_start[i]
						do = True
						break
					
		joint_action = {"pass": joint_action, "task": task_assign} if task_assign is not None else joint_action
		obs, ri_array, self.terminated, info = super().step(joint_action)

		if isinstance(info, dict):
			info["safety_intervention_count"] = self.safety_intervention_count

		return obs, ri_array, self.terminated, info
