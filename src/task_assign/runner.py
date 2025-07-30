import torch
import numpy as np
from collections import deque
import math
import os
import matplotlib.pyplot as plt
from copy import deepcopy

from src.policy.policy import policy
from src.task_assign.task_agent import TaskAgent

class Runner():
    def __init__(self, args, env, reward_list, training=False):

        # Prepare directories
        self.args = args
        self.args.task_num = env.task_num
        self.args.node_num = env.n_nodes

        self.env = env
        self.reward_list = reward_list
        self.test_num = args.test_num
        self.training = training
        if training:
            self.test_mode = False
        else:
            self.test_mode = True

        self.episode_length = self.env.time_limit
        self.current_step = 0
        self.env_step = 0
        self.current_episode = 0
        self.max_step = args.running_steps
        self.batch_size = args.batch_size
        self.check_interval = 100000
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.info_buffer = deque(maxlen=self.test_num)
        self.task_Agent = TaskAgent(self.args.task_assigner, self.args)

    def get_avail_actions(self):
        avail_actions = []
        for i in range(self.env.n_agents):
            avail_actions.append(self.env.get_avail_agent_actions(i, self.env.n_actions)[0])
        avail_actions = torch.tensor(avail_actions, dtype=torch.int32)

        return avail_actions
    
    def run_episode(self):

        obs_n = self.env.reset()
        done = False
        task_info = {"Tasks": [], "Assigned": [[] for _ in range(self.env.n_agents)]}
        episode_score = 0
        env_step = 0
        
        while not done:
            agents_action = policy(obs_n, self.env)
            task_assign = self.task_Agent.assign_task(self.env, task_info)
            joint_action = {"agent": agents_action, "task": task_assign}

            #self.env.render()
            #input()

            next_obs_n, rew_n, terminated_n, info = self.env.step(joint_action)
            task_info = {"Tasks": self.env.current_tasklist, "Assigned": self.env.assigned_tasks}
            
            done = all(terminated_n)

            #報酬をバッファへ
            if self.training:
                    self.task_Agent.task_assigner.buffer_add_rewards(sum(rew_n), done)

            episode_score += np.mean(rew_n)
                            
            env_step += 1
            obs_n = deepcopy(next_obs_n)

        return episode_score, env_step, info

    def run(self):

        step_tmp = 0

        if self.training:
            self.task_Agent.task_assigner.set_test_mode(False)
            while self.current_step < self.max_step:
                episode_score, env_step, info = self.run_episode()
                self.info_buffer.append(info)
                self.current_step += env_step
                step_tmp += env_step

                self.task_Agent.task_assigner.process_end_episode()

                #training
                if self.task_Agent.task_assigner.update_ready():
                    a_loss, c_loss, e_loss = self.task_Agent.task_assigner.update()
                
                #log
                if step_tmp > self.check_interval:
                    print("Current step:", self.current_step)
                    print("a_loss:", a_loss, "c_loss:", c_loss, "e_loss:", e_loss)
                    print("Average task completion:", np.mean([info["task_completion"] for info in self.info_buffer]))
                    step_tmp = 0

            self.test_mode = True
            self.task_Agent.task_assigner.set_test_mode(True)


        if self.test_mode:
            self.info_buffer = deque(maxlen=self.test_num)
            for _ in range(self.test_num):
                episode_score, env_step, info = self.run_episode()
                self.info_buffer.append(info)
                #print(info["goal_account"])

        steps = [info["step"] for info in self.info_buffer]
        goal_account = [info["goal_account"] for info in self.info_buffer]
        task_completion = [info["task_completion"] for info in self.info_buffer]

        print("Total test episodes:", len(self.info_buffer))
        print("Average steps:", np.mean(steps))
        print("Average task completion:", np.mean(task_completion))

        return

    def finish(self):
        self.env.close()

