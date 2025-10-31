from Models.DQN import DQN_Agent  # ← 使用你的 DQN 学习器（含 online/target QNet）
from Agents.AgentUtils import *
import os, copy, math, random
import torch
import numpy as np


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class High_Level_Agent(object):
    """
    纯 DQN 版本：
    - 不再使用 Termination_Net/Option 终止逻辑
    - HL_option_dim 直接视为动作数 action_dim
    - 保留原方法名以减少外部代码改动
    """

    def __init__(self, agent_ID, device, args):
        self.ag_idx = agent_ID
        self.HL_state_dim = args['state_dim_ideal']  # int
        self.HL_option_dim = args['option_dim']  # 作为动作数 action_dim 使用
        self.device = device

        # ε-greedy（保持你原来的风格）
        self.sample_count = 0
        self.epsilon = 0.95
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 1000

        # === DQN 学习器 ===
        # 这里用 n_hiddens=256（你原来也是 256）
        # 其余超参从 args 里取，保持可配置
        dqn_args = dict(
            hidden_dims=(256, 128, 128),
            dueling=True,
            double_dqn=True,

            high_level_lr=2e-5,  # Q 网络学习率
            gamma=0.95,
            tau=1e-3,
            batch_size=128,
            memory_capacity=262144,
            max_steps=args.get('max_steps', 5000),
            eps_start=0.95,
            eps_end=0.01,
            eps_decay=5000,

            replay_type=args.get('replay_type', "Uniform"),  # 或 "PER"
        )

        self.learner = DQN_Agent(
            n_observations=self.HL_state_dim,
            n_hiddens=256,  # 仅占位，不影响结构
            n_actions=self.HL_option_dim,
            device=self.device,
            args=dqn_args
        )

    # 兼容原接口（DQN 不需要额外初始化）
    def init_training(self):
        pass

    # 写入回放缓存（字段按你的缓冲区来）
    def add2replay_memory(self, state_old, option, reward, state_new, done, step_idx=None, epoch_idx=None):
        # DQN_Agent.push_transition(s, a, r, ns, done)
        self.learner.push_transition(state_old.flatten(), option, reward, state_new, done, step_idx)

    # === 动作选择（训练时 ε-greedy）===
    def get_option_during_train(self, state, current_option=None):
        action = self.learner.select_action(np.array(state, dtype=np.float32))
        return int(action)

    # === 动作选择（测试时贪心）===
    @torch.no_grad()
    def get_option_during_test(self, state, current_option=None):
        # 测试：ε≈0，强制贪心
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.learner.online_net(state_t)
        action = int(q.argmax(dim=1).item())
        return action

    def policy_update(self):
        loss = self.learner.update()

        return 0.0, float(loss)

    def load_QNet_model(self, model_save_dir, file_str):
        DNN_model_save_dir = r"{}/QNet_models".format(model_save_dir)
        mkdir(DNN_model_save_dir)
        file_name = "agent_ID_" + str(self.ag_idx) + "_QNet" + "_" + str(file_str) + ".pt"
        DNN_model_save_path = os.path.join(DNN_model_save_dir, file_name)
        if os.path.exists(DNN_model_save_path):
            print('Load model:', DNN_model_save_path)
            loaded_paras = torch.load(DNN_model_save_path, map_location=self.device)
            self.learner.online_net.load_state_dict(loaded_paras)
            self.learner.target_net.load_state_dict(loaded_paras)  # 同步目标网
        else:
            raise ValueError("Error: No saved QNet model at {}".format(DNN_model_save_path))

    def save_QNet_model(self, model_save_dir, file_str):
        DNN_model_save_dir = r"{}/QNet_models".format(model_save_dir)
        mkdir(DNN_model_save_dir)
        file_name = "agent_ID_" + str(self.ag_idx) + "_QNet" + "_" + str(file_str) + ".pt"
        DNN_model_save_path = os.path.join(DNN_model_save_dir, file_name)
        model_state = copy.deepcopy(self.learner.online_net.state_dict())
        torch.save(model_state, DNN_model_save_path)
        print('Saved QNet to:', DNN_model_save_path)
