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
    step_account: int = 0

@dataclass
class OtherAgentsInfo:
    #各エージェントの位置をstep数分格納
    other_agents_pos: List[List[Tuple[float,float]]]

class PBS:
    def __init__(self, args):
        self.env = gym.make("drp_env:drp-" + str(1) + "agent_" + args.map_name + "-v2", state_repre_flag = "onehot_fov")
        self.num_agents = args.agent_num
        self.time_limit = 20#args.time_limit
        self.schedule_actions = []
        self.goal_rec = []


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
        self.goal_rec = env.goal_array.copy()
        #今：step × agent_num
        #other_agents_infos = OtherAgentsInfo(other_agents_pos=[[] for _ in range(self.time_limit - env.step_account)])
        #改：agent_num × step
        tmp_list = [[] for _ in range(self.time_limit+1)]
        other_agents_infos = OtherAgentsInfo(other_agents_pos=[[] for _ in range(self.num_agents)])

        for i in range(self.num_agents):
            other_agents_infos.other_agents_pos[i] = tmp_list.copy()
            #道の途中のエージェントへの対応
            #ノード上にいない場合，ノードにつくまでのposをother_agents_infosに格納
            if env.pos[env.current_start[i]] != [env.obs[i][0], env.obs[i][1]]:
                agent_info = AgentInfo(pos=(env.obs[i][0], env.obs[i][1]), 
                                   current_start=env.current_start[i], 
                                   current_goal=env.current_goal[i], 
                                   action_history=[], 
                                   pos_history=[(env.obs[i][0], env.obs[i][1])])
                
                self.env.reset()
                self.set_env_info(agent_info, env.goal_array[i])
                while self.env.pos[self.env.current_start[0]] != [self.env.obs[0][0], self.env.obs[0][1]]:
                    self.set_env_info(agent_info, env.goal_array[i])
                    avail_actions = self.env.get_avail_agent_actions(0, self.env.n_actions)[1]
                    action = avail_actions[0]
                    obs, reward, done, info = self.env.step([action])
                    new_pos = (self.env.obs[0][0], self.env.obs[0][1]) 
                    new_action_history = agent_info.action_history + [action]
                    new_pos_history = agent_info.pos_history + [new_pos]
                    new_current_start = self.env.current_start[0]
                    new_current_goal = self.env.current_goal[0]   
                    agent_info = AgentInfo(pos=new_pos, 
                                                current_start=new_current_start, 
                                                current_goal=new_current_goal, 
                                               action_history=new_action_history, 
                                               pos_history=new_pos_history,
                                               step_account=0)
                
                for _ in range(self.time_limit+1):
                    agent_info.pos_history.append(agent_info.pos_history[-1])

                other_agents_infos.other_agents_pos[i] = agent_info.pos_history

        longest_step = 0
        
        priority_list = self.get_priority(obs, env)
        for i in priority_list:
            agent_info = AgentInfo(pos=(env.obs[i][0], env.obs[i][1]), 
                                   current_start=env.current_start[i], 
                                   current_goal=env.current_goal[i], 
                                   action_history=[], 
                                   pos_history=[(env.obs[i][0], env.obs[i][1])])
            
            near_goal_nodes = self.env.get_near_nodes(env.goal_array[i])
            while self.schedule_actions[i] == []:
                tmp_goal = near_goal_nodes.pop(0)
                
                if agent_info.current_start == tmp_goal:
                    self.schedule_actions[i] = [tmp_goal] * self.time_limit
                    for _ in range(self.time_limit+1):
                        agent_info.pos_history.append(agent_info.pos_history[-1])
                    other_agents_infos.other_agents_pos[i] = agent_info.pos_history
                    continue

                agent_infos_deque = deque([agent_info])
                current_agent_info = agent_info

                self.env.reset()
                goal_flag = False
                while not goal_flag and len(agent_infos_deque) > 0 and current_agent_info.step_account+1 < self.time_limit:
                    current_agent_info = agent_infos_deque.popleft()

                    self.set_env_info(current_agent_info, tmp_goal)

                    avail_actions = self.env.get_avail_agent_actions(0, self.env.n_actions)[1]
                    for action in avail_actions:

                        self.set_env_info(current_agent_info, tmp_goal)

                        obs, reward, done, info = self.env.step([action])
                        new_pos = (self.env.obs[0][0], self.env.obs[0][1])
                        #collision_flag = self.collision_detect(new_pos, other_agents_infos.other_agents_pos[current_agent_info.step_account+1])
                        collision_flag = self.collision_detect(new_pos, other_agents_infos.other_agents_pos, current_agent_info.step_account+1, i)
                        if collision_flag == True:#衝突とゴールが同時の場合にバグが起こるため
                            self.env.reset()

                        elif collision_flag == False:
                            new_action_history = current_agent_info.action_history + [action]
                            new_pos_history = current_agent_info.pos_history + [new_pos]
                            new_current_start = self.env.current_start[0]
                            new_current_goal = self.env.current_goal[0]

                            new_agent_info = AgentInfo(pos=new_pos, 
                                                    current_start=new_current_start, 
                                                        current_goal=new_current_goal, 
                                                    action_history=new_action_history, 
                                                    pos_history=new_pos_history,
                                                    step_account=current_agent_info.step_account + 1)
                            
                            if all(done) is True:
                                if info["goal"] == True:
                                    #ゴールした後，そこに止まったときに優先度の高いエージェントと衝突しないか見るのを実装
                                    #衝突する場合，continue
                                    print("ゴール発見")
                                    if self.check_after_collision(new_agent_info, other_agents_infos.other_agents_pos, i) == True:
                                        print("ゴール後衝突発見")
                                        continue
                                    else:
                                        print("ゴール後衝突なし")
                                        pass
                                    """
                                    print("AAAAAA",new_agent_info.step_account)
                                    print(current_agent_info.pos, other_agents_infos.other_agents_pos[current_agent_info.step_account])
                                    print(new_agent_info.pos_history)
                                    print(other_agents_infos.other_agents_pos)
                                    """
                                    #今：各ステップ毎に格納（）
                                    #改：各エージェント毎に格納（i番目に入れるだけ）
                                    self.schedule_actions[i] = new_agent_info.action_history + new_agent_info.action_history[-1:]*(self.time_limit - len(new_agent_info.action_history))
                                    """
                                    for j in range(len(other_agents_infos.other_agents_pos)):
                                        if j < len(new_agent_info.pos_history):
                                            other_agents_infos.other_agents_pos[j].append(new_agent_info.pos_history[j])
                                        else:
                                            other_agents_infos.other_agents_pos[j].append(new_agent_info.pos_history[-1])
                                    """

                                    for _ in range(self.time_limit - len(new_agent_info.pos_history)+1):
                                        new_agent_info.pos_history.append(new_agent_info.pos_history[-1])

                                    other_agents_infos.other_agents_pos[i] = new_agent_info.pos_history
                                    
                                    goal_flag = True
                                    
                                    break

                                elif info["collision"] == True:
                                    pass
                                elif info["timeup"] == True:
                                    print("timeup!!!!!!")

                            else:
                                agent_infos_deque.append(new_agent_info)

            ###
            #print(other_agents_infos.other_agents_pos)

    def get_priority(self, obs, env):
        priority_list = []
        for i in range(self.num_agents):
            #エッジ上のエージェントへの対応（未）
            path_length = env.get_path_length(env.current_start[i], env.goal_array[i])
            priority_list.append((i, path_length))
        
        #遠いものを優先
        #priority_list.sort(key=lambda x: x[1], reverse=True)
        #近いものを優先
        priority_list.sort(key=lambda x: x[1], reverse=False)

        priority_list = [x[0] for x in priority_list]

        return priority_list
    
    #エージェントのxy座標(self.obs)環境にセットする
    #obs=tuple(array[],array[]...)
    def set_env_info(self, agent_infos, goal_array):
        """
        self.env.obs = tuple(np.array([agent_infos.pos[0], agent_infos.pos[1], self.env.obs[0][2], self.env.obs[0][3]]) for _ in range(1))
        self.env.current_start[0] = agent_infos.current_start
        self.env.current_goal[0] = agent_infos.current_goal
        self.env.step_account = 10#len(agent_infos.action_history)
        """
        self.env.set_1agent_info(pos=agent_infos.pos, 
                                 current_start=agent_infos.current_start, 
                                 current_goal=agent_infos.current_goal, 
                                 goal_array = goal_array,)
        return
    
    #衝突判定
    """
    def collision_detect(self, agent_infos, other_agents_pos_t):
        speed = 5
        collision_flag = False
        for other_agent_pos in (other_agents_pos_t):
            distance = math.dist(agent_infos, other_agent_pos)
            if distance<speed:
                collision_flag = True

        return collision_flag
    """
    def collision_detect(self, agent_pos, other_agents_pos, step_t, agent_num):
        speed = 5
        collision_flag = False
        
        """
        for other_agent_pos in (other_agents_pos):
            if other_agent_pos[step_t] == []:
                continue
            
            distance = math.dist(agent_pos, other_agent_pos[step_t])
            if distance<speed:
                collision_flag = True
        """
        for i in range(len(other_agents_pos)):
            if len(other_agents_pos[i]) == step_t:
                print(step_t, other_agents_pos[i])
            if i==agent_num or other_agents_pos[i][step_t] == []:
                continue
            
            distance = math.dist(agent_pos, other_agents_pos[i][step_t])
            if distance<speed:
                collision_flag = True

        return collision_flag
    
    #passを見つけた後，その後に衝突する恐れがないか確認
    #まだ衝突がある
    def check_after_collision(self, agent_info, other_agents_pos, agent_num):
        #self.schedule_actions
        collision_flag = False
        for t in range(len(agent_info.pos_history)):
            if self.collision_detect(agent_info.pos_history[t], other_agents_pos, t,  agent_num):
                collision_flag = True
                break
        return collision_flag

    def policy(self, obs, env):
        actions = []
        #schedule_actionsが変なとき，初期化
        if self.schedule_actions != []:
            for i in range(self.num_agents):
                if self.schedule_actions[i] == []:
                    self.schedule_actions = []
                    break
            
        #だれかのゴールが変わったとき，計算し直す
        if self.goal_rec != env.goal_array:
            self.schedule_actions = []

        if self.schedule_actions == []:
            self.culc_actions(obs, env)
            print("新たに計算")
            print("schedule_actions", self.schedule_actions)
            #return [0 for _ in range(self.num_agents)]
            
        for i in range(self.num_agents):
            actions.append(self.schedule_actions[i].pop(0))

        return actions