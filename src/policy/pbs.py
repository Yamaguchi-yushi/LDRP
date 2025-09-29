import gym
import math
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from collections import deque

@dataclass
class AgentInfo:
    pos: Tuple[float, float]
    current_start: int
    current_goal: int
    action_history: List[int]
    pos_history: List[Tuple[float,float]]

@dataclass
class OtherAgentsInfo:
    other_agents_pos: List[List[Tuple[float,float]]]

class PBS:
    def __init__(self, args):
        self.env = gym.make("drp_env:drp-" + str(1) + "agent_map_" + args.map_name + "-v2", state_repre_flag = "onehot_fov")
        self.num_agents = args.agent_num
        self.time_limit = args.time_limit
        self.schedule_actions = []

    #優先度を決める，ゴールまでの距離が長いものを優先（優先度リストを作る）
    #優先度順に最短経路を決定
    #最短経路の求め方
    #環境の初期化（goal_arrayの設定）
    #dequeの先頭のデータをとる(popleft)
    #agentのデータを環境にセットする
    #avail_actionの数だけstepを実行，ゴールしてたなら@へ
    #衝突判定がないものでdataclassのインスタンスを作り，データを更新，dequeへ(append)
    #@action_historyを参照してschedule_actionへ追加，pos_historyを参照してposを更新
    def culc_actions(self, obs ,env):
        self.schedule_actions = [[] for _ in range(self.num_agents)]
        other_agents_infos = OtherAgentsInfo(other_agents_pos=[[] for _ in range(self.time_limit - env.step_account)])
        
        priority_list = self.get_priority(obs, env)
        for i in priority_list:
            agent_info = AgentInfo(pos=(env.obs[i][0], env.obs[i][1]), 
                                   current_start=env.current_start[i], 
                                   current_goal=env.current_goal[i], 
                                   action_history=[], 
                                   pos_history=[(env.obs[i][0], env.obs[i][1])])
            agent_infos_deque = deque([agent_info])

            self.env.reset()
            self.env.goal_array[0] = env.goal_array[i]
            goal_flag = False
            while goal_flag == False and len(agent_infos_deque) > 0:
                current_agent_info = agent_infos_deque.popleft()
                self.set_env_info(current_agent_info)

                avail_actions = self.env.get_avail_agent_actions(0, self.env.n_actions)[1]
                
                for action in avail_actions:
                    obs, reward, done, info = self.env.step([action])
                    new_pos = (self.env.obs[0][0], self.env.obs[0][1])
                    collision_flag = self.collision_detect(current_agent_info, other_agents_infos)

                    if collision_flag == False:
                        new_action_history = current_agent_info.action_history + [action]
                        new_pos_history = current_agent_info.pos_history + [new_pos]
                        new_current_start = self.env.current_start[0]
                        new_current_goal = self.env.current_goal[0]

                        new_agent_info = AgentInfo(pos=new_pos, 
                                                   current_start=new_current_start, 
                                                    current_goal=new_current_goal, 
                                                   action_history=new_action_history, 
                                                   pos_history=new_pos_history)
                        
                        if all(done) is True:
                            if info["goal"] == True:
                                goal_flag = True
                                self.schedule_actions[i] = new_agent_info.action_history + new_agent_info.action_history[-1:]*(self.time_limit - len(new_agent_info.action_history))
                                for j in range(len(other_agents_infos.other_agents_pos)):
                                    other_agents_infos.other_agents_pos[j].append(new_agent_info.pos_history[j])
                                break

                            elif info["collision"] == True:
                                pass
                            elif info["timeup"] == True:
                                print("timeup!!!!!!")

                        else:
                            agent_infos_deque.append(new_agent_info)

                ###

    def get_priority(self, obs, env):
        priority_list = []
        for i in range(self.num_agents):
            #エッジ上のエージェントへの対応（未）
            path_length = env.get_path_length(env.current_start[i], env.goal_array[i])
            priority_list.append((i, path_length))
        
        priority_list.sort(key=lambda x: x[1], reverse=True)
        priority_list = [x[0] for x in priority_list]

        return priority_list
    
    #エージェントのxy座標(self.obs)環境にセットする
    def set_env_info(self, agent_infos):
        self.env.obs[0] = tuple(np.array([agent_infos.pos[0], agent_infos.pos[1], self.env.obs[0][2], self.env.obs[0][3]]))
        self.env.current_start[0] = agent_infos.current_start
        self.env.current_goal[0] = agent_infos.current_goal
        self.env.step_account = len(agent_infos.action_history)
        return
    
    #衝突判定
    def collision_detect(self, agent_infos, other_agents_infos):
        speed = 5
        collision_flag = False

        for other_agent_pos in (other_agents_infos.other_agents_pos):
            distance = math.dist(agent_infos.pos, other_agent_pos)
            if distance<speed:
                collision_flag = True

        return collision_flag
    

    def policy(self, obs, env):
        actions = []
        if len(self.schedule_actions) == 0:
            return [0 for _ in range(self.num_agents)]
            
        actions = self.schedule_actions.pop(0)

        return actions