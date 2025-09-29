import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

import copy

class Buffer():
    def __init__(self, buffer_size, n_envs, obs_shape, action_dim, device, args=None):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        #self.pos = 0
        self.agent_num = args.agent_num if args else 1 
        self.reset_buffer()

    def reset_buffer(self):
        self.steps = []
        self.states = []
        self.actions = []
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.act_rew = []
        #self.pos = 0

        self.returns = []

    def reset_rewards(self):
        self.rewards = []
        self.dones = []

    def add_actions(self, step_idx, state, action, log_prob, entropy, value):
        #いっぱいでなければ追加
        if len(self.steps) < self.buffer_size:
            self.steps.append(step_idx)
            self.states.append(state)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)
            #self.pos += 1

        else:
            pass

    def add_rewards(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def normalize_rewards(self):
        self.rewards = np.array(self.rewards, dtype=np.float32)
        self.rewards = (self.rewards - np.mean(self.rewards)) / (np.std(self.rewards) + 1e-8)

    def compute_returns(self, gamma=0.99):
        returns = []
        R = 0
        list_pos = len(self.steps) - 1
        for reward, done, i in zip(reversed(self.rewards), reversed(self.dones), range(len(self.rewards))):
            if done:
                R = 0

            R = reward + gamma * R
            for _ in range(self.agent_num):
                if self.steps[list_pos] == len(self.rewards) - i:
                    returns.insert(0, R)
                    list_pos -= 1
                else:
                    break

        self.returns.extend(returns)
        self.reset_rewards()  # Clear rewards after computing returns

    ###########################未実装
    def compute_returns_and_advantages(self, next_value, gamma=0.99, lam=0.95):
        """GAEを使って，行動ステップに対応する advantage/return を計算"""
        T = len(self.rewards)
        values_full = [0.0] * (T + 1)
        for i, t in enumerate(self.action_steps):
            values_full[t] = self.values[i].item()
        values_full.append(next_value)

        # GAE計算
        advantages_full = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            delta = self.rewards[t] + gamma * values_full[t + 1] * (1 - self.dones[t]) - values_full[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages_full[t] = gae

        # 行動ステップだけ抽出
        advantages = [advantages_full[t] for t in self.action_steps]
        returns = [adv + values_full[t] for t, adv in zip(self.action_steps, advantages)]
        return returns, advantages
    
    def compute_advantages(self, values, gamma=0.99, lam=0.95):
        advantages = []
        A = 0
        list_pos = len(self.steps) - 1
        for value, reward, done, i in zip(reversed(values), reversed(self.rewards), reversed(self.dones), range(len(self.rewards))):
            if done:
                A = 0

            delta = reward + gamma * value - value
            A = delta + gamma * lam * A
            for _ in range(self.agent_num):
                if self.steps[list_pos] == len(self.rewards) - i:
                    advantages.insert(0, A)
                    list_pos -= 1
                else:
                    break

        return advantages
    #############################

    def get_tensors(self, device):
        states = torch.stack(self.states).to(device)
        actions = torch.tensor(self.actions).to(device)
        log_probs = torch.stack(self.log_probs).detach().to(device)
        values = torch.stack(self.values).detach().to(device).squeeze()
        returns = torch.tensor(self.returns).to(device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return states, actions, log_probs, returns, advantages


class PPO(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPO, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.policy_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.apply(self.orthogonal_init)

    def forward(self, x):
        policy = self.policy_layer(x)
        value = self.value_layer(x)
        return policy, value
    
    def orthogonal_init(self,m):
        if isinstance(m, nn.Linear):
            init.orthogonal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

    def save_model(self, path):
        # Implement model saving logic here
        pass

    def load_model(self, path):
        # Implement model loading logic here
        pass

class PPOAgent_1():
    def __init__(self, args):
        input_dim = args.agent_num * args.node_num * 2 + args.task_num * args.node_num + args.agent_num 
        output_dim = args.task_num * args.agent_num 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PPO(input_dim, output_dim).to(self.device)
        self.buffer = Buffer(args.buffer_size, args.n_envs, (input_dim,), output_dim, self.device, args)
        self.args = args
        self.test_mode = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.model.policy_layer.parameters(), lr=args.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.model.value_layer.parameters(), lr=args.learning_rate)

    def update(self):
        # Implement the PPO update logic here
        states, actions, log_probs, returns, advantages = self.buffer.get_tensors(self.device)
        dataset = TensorDataset(states, actions, log_probs, returns, advantages)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

        for _ in range(self.args.epochs):
            for batch in loader:
                s, a, old_logp, r, adv = batch
                logits, values = self.model(s)
            dist = Categorical(logits=logits)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(a)

            ratio = (new_log_probs - old_logp).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.args.clip_epsilon, 1 + self.args.clip_epsilon) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (r - values.squeeze(-1)).pow(2).mean()

            #loss = actor_loss + self.args.value_loss_coef * critic_loss - self.args.entropy_coef * entropy
            #self.optimizer.zero_grad()
            #loss.backward()
            #self.optimizer.step()

            actor_loss -= self.args.entropy_coef * entropy
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        self.buffer_reset()

        return actor_loss.detach(), critic_loss.detach(), entropy.detach()


    def assign_task(self, env, current_tasklist, assigned_tasklist):
        current_tasklist = copy.deepcopy(current_tasklist)  
        assigned_tasklist = copy.deepcopy(assigned_tasklist) 

        task_assign = [-1 for _ in range(env.n_agents)]

        len_current_task = len(current_tasklist)
        for i in range(env.n_agents):
            if any(len(task) == 0 for task in assigned_tasklist) and len_current_task > 0:
                state = self.create_state(env, current_tasklist, assigned_tasklist)
                state = state.clone().detach().to(self.device)
                policy, value = self.model(state)
                #マスク
                #task持ちのエージェント
                mask = torch.zeros_like(policy)
                for i in range(env.n_agents):
                    if len(assigned_tasklist[i]) > 0:
                        agent_idx = env.task_num*i
                        mask[agent_idx:agent_idx+env.task_num] = 1
                #task数が少ないとき
                for i in range(env.task_num):
                    if i > len(current_tasklist)-1:
                        mask_idx = []
                        for j in range(env.n_agents):
                            mask_idx.append(env.task_num * j + i)
                        mask[mask_idx] = 1
                #使用済みのtask
                for i in range(len(current_tasklist)):
                    if current_tasklist[i][0] == -1:
                        mask_idx = []
                        for j in range(env.n_agents):
                            mask_idx.append(env.task_num * j + i)
                        mask[mask_idx] = 1

                policy[mask == 1] = float('-inf')
                policy = F.softmax(policy, dim=-1)

                if self.test_mode:# 実行用
                    action = policy.argmax().item()
                else:# 学習用
                    dist = Categorical(policy)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    entropy = dist.entropy()

                    action = action.item()

                    self.buffer.add_actions(env.step_account, state, action, log_prob, entropy, value)
            
                #actionをみてtask_assignを決定
                #currentとassignedを更新
                q, r = divmod(action, env.task_num)
                assigned_tasklist[q].append(current_tasklist[r])
                task_assign[q] = r
                current_tasklist[r][0] = -1  # タスクを割り当てたので、タスクリストから削除
                len_current_task -= 1

            else:
                pass

        return task_assign
    
    def create_state(self, env, current_tasklist, assigned_tasklist):
        current_tasklist = copy.deepcopy(current_tasklist)#[[4,2,-1],[1,5,-1]][s,g,time]->s,gのonehotに
        assigned_tasklist = copy.deepcopy(assigned_tasklist)#[[4,2,-1]]->エージェントごとのonehotに
        onehot_obs = copy.deepcopy(env.obs_onehot)
        tensor_list = [torch.tensor(lst) for lst in onehot_obs]
        state = torch.cat(tensor_list, dim=0)

        #current_tasklistのonehot化
        for _ in range(env.task_num):
            task_tensor = torch.zeros(env.n_nodes, dtype=torch.float32)
            if len(current_tasklist) > 0:
                task = current_tasklist.pop(0)
                if task[0] != -1:
                    task_tensor[task[0]] = 1
                    task_tensor[task[1]] = -1 
                else:
                    pass

            state = torch.cat((state, task_tensor), dim=0)
        #assignedされているエージェントを可視化
        assigned = []
        for i in range(env.n_agents):
            if len(assigned_tasklist[i]) > 0:
                assigned.append(1)
            else:
                assigned.append(0)
        assigned_tensor = torch.tensor(assigned, dtype=torch.float32)
        state = torch.cat((state, assigned_tensor), dim=0)

        return state
        
    def update_ready(self):
        if len(self.buffer.steps) < self.buffer.buffer_size:
            return False
        return True

    def buffer_add_rewards(self, reward, done):
        self.buffer.add_rewards(reward, done)

    def buffer_reset(self):
        self.buffer.reset_buffer()

    def set_test_mode(self, mode_tf):
        self.test_mode = mode_tf

    #エピソード終了時の処理
    def process_end_episode(self):
        self.buffer.normalize_rewards()
        self.buffer.compute_returns(self.args.gamma)

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
