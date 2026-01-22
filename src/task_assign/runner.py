import torch
import numpy as np
from collections import deque
import math
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import time

from src.policy.policy import policy
from src.policy.policy_manager import PolicyManager 
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
        self.path_planner = PolicyManager(self.args)
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
        #実験用
        tmp_step = 0
        tmp_goal = self.env.goal_array.copy()
        self.tmp_flag = False

        #行動のログ
        #"""
        action_log = [[] for _ in range(self.env.n_agents)]
        #"""
        
        while not done:
            agents_action = self.path_planner.policy(obs_n, self.env)
            #agents_action = policy(obs_n, self.env)
            for i in range(self.env.n_agents):
                action_log[i].append(agents_action[i])

            task_assign = self.task_Agent.assign_task(self.env, task_info)
            joint_action = {"pass": agents_action, "task": task_assign}

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

            if self.env.goal_array != tmp_goal:
                tmp_goal = self.env.goal_array.copy()
                tmp_step = 0
            else:
                tmp_step += 1
            if tmp_step > 40:
                self.tmp_flag = True

            #"""
            if 1==0:
                #print("############################")
                print("step:", env_step)
                print("current_start:", self.env.current_start)
                print("current_goal:", self.env.current_goal)
                print("goal_array:", self.env.goal_array)
                #print("current_tasklist:", self.env.current_tasklist)
                #print("assigned_tasks:", self.env.assigned_tasks)
                #print("assigned_list:", self.env.assigned_list)
                print("agents_action:", agents_action)
                #print("task_assign:", task_assign)
                print("############################")
            #"""

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
                    print("a_loss:", a_loss.numpy(), "\nc_loss:", c_loss.numpy(), "\ne_loss:", e_loss.numpy())
                    print("Average task completion:", np.mean([info["task_completion"] for info in self.info_buffer]))
                    step_tmp = 0
                

            self.test_mode = True


        times = []
        tmp_list = []
        self.info_buffer = deque(maxlen=self.test_num)
        if self.test_mode:
            for i in range(self.test_num):
                start = time.perf_counter()
                episode_score, env_step, info = self.run_episode()
                end = time.perf_counter()

                self.info_buffer.append(info)
                #print(info["goal_account"])
                tmp_list.append(self.tmp_flag)

                times.append(end - start)
                #if (i+1) % 10 == 0:
                #    print(f"Test Episode {i+1}/{self.test_num} completed.")

        steps = [info["step"] for info in self.info_buffer]
        goal_account = [info["goal_account"] for info in self.info_buffer]
        task_completion = [info["task_completion"] for info in self.info_buffer]
        full_completion = [info["task_completion"] for info in self.info_buffer if not info["collision"]]
        non_lock_completion = [info["task_completion"] for idx, info in enumerate(self.info_buffer) if tmp_list[idx]==False]

        #print(full_completion)
        #print("衝突なし:",np.mean(full_completion),len(full_completion))
        #print(non_lock_completion)
        #print("ロックなし", np.mean(non_lock_completion), len(non_lock_completion))
        print("Total test episodes:", len(self.info_buffer))
        print("Average steps:", np.mean(steps))
        print("Average task completion:", np.mean(task_completion))
        print("最高値:",np.max(task_completion))
        print("最低値:",np.min(task_completion))
        print("実行時間:", np.sum(times), "秒")
        print("平均実行時間:", np.mean(times), "秒")

        return

    def finish(self):
        self.env.close()

