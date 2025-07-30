import torch 
import numpy as np

#from epymarl.src.modules.agents.rnn_agent import RNNAgent  # EPyMARLそのまま使う
from .rnn_agent import RNNAgent  # ローカルのRNNAgentを使用


class DummyArgs:
    def __init__(self, hidden_dim=64, n_actions=None, use_rnn=False):
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.use_rnn = use_rnn


class PolicyRunner:
    def __init__(self, model_path, input_shape, n_actions, agent_num):
        self.args = DummyArgs(hidden_dim=64, n_actions=n_actions, use_rnn=False)
        self.agent = RNNAgent(input_shape, self.args)
        self.agent.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.agent.eval()
        self.hidden_states = [self.agent.init_hidden() for _ in range(agent_num)]

    def get_action(self, ag_idx, obs, avail_actions):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        h_in = self.hidden_states[ag_idx]

        q_values, h_out = self.agent(obs_tensor, h_in)
        self.hidden_states[ag_idx] = h_out.detach()  # detachでグラフ切って次に備える

        q_numpy = q_values.squeeze(0).detach().numpy()
        masked_q = [q_numpy[a] if a in avail_actions else -np.inf for a in range(len(q_numpy))]
        # if agents act bad behavior, give the agents the second-good action from masked_q
        #
        #
        #####################################
        return int(np.argmax(masked_q))