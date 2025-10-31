import torch
import torch.nn.functional as F
import torch.optim as optim
from utils.Replay_Memory import ReplayBuffer_HighOC, ReplayBuffer_HighOC_Reward_Backpropagation
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from Agents.AgentUtils import *


class Q_Net(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, max_step_idx=5000, embedding_dim=10):
        super(Q_Net, self).__init__()

        # 添加一个 Embedding 层，将 step_idx 转换为嵌入向量
        self.embedding = nn.Embedding(num_embeddings=max_step_idx, embedding_dim=embedding_dim)

        # 原有的网络部分
        self.fc1 = nn.Linear(state_dim + embedding_dim, hidden_dim)  # 修改：输入维度增加 embedding_dim
        self.layer1 = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.LayerNorm(hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.LayerNorm(hidden_dim)
        self.fc_A = nn.Linear(hidden_dim, action_dim)
        self.fc_V = nn.Linear(hidden_dim, 1)

        # 参数初始化
        nn.init.uniform_(self.fc1.weight, -1 / np.sqrt(256), 1 / np.sqrt(256))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.uniform_(self.layer2.weight, -1 / np.sqrt(256), 1 / np.sqrt(256))
        nn.init.constant_(self.layer2.bias, 0)
        nn.init.uniform_(self.layer4.weight, -1 / np.sqrt(256), 1 / np.sqrt(256))
        nn.init.constant_(self.layer4.bias, 0)
        nn.init.uniform_(self.fc_A.weight, -3e-3, 3e-3)
        nn.init.constant_(self.fc_A.bias, 0)
        nn.init.uniform_(self.fc_V.weight, -3e-3, 3e-3)
        nn.init.constant_(self.fc_V.bias, 0)

    def forward(self, x, step_idx):
        # 将 step_idx 通过 Embedding 层转换为稠密的向量
        step_embedding = self.embedding(step_idx)  # 输出形状 (batch_size, embedding_dim)

        # 拼接 state 和 step_embedding
        x = torch.cat((x, step_embedding), dim=1)  # 拼接后形状为 (batch_size, state_dim + embedding_dim)

        # 原有的网络部分
        x = F.relu(self.fc1(x))
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)

        A = self.fc_A(x)
        V = self.fc_V(x)

        # Q值由V值和A值计算得到
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))

        return Q

    def cal_Advantage_fun(self, x, step_idx):
        # 将 step_idx 通过 Embedding 层转换为稠密的向量
        step_embedding = self.embedding(step_idx)  # 输出形状 (batch_size, embedding_dim)

        # 拼接 state 和 step_embedding
        x = torch.cat((x, step_embedding), dim=1)  # 拼接后形状为 (batch_size, state_dim + embedding_dim)

        # 原有的网络部分
        x = F.relu(self.fc1(x))
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)

        A = self.fc_A(x)
        return A - A.mean()  # A值计算得到


class Termination_Net(nn.Module):
    """
    A network to predict the termination condition in a reinforcement learning task.
    """

    def __init__(self, n_states, n_hiddens, n_options, max_step_idx=5000, embedding_dim=10):
        super(Termination_Net, self).__init__()

        # Embedding layer to map step_idx to dense vector
        self.embedding = nn.Embedding(num_embeddings=max_step_idx, embedding_dim=embedding_dim)

        # Define the network layers
        self.fc1 = nn.Linear(n_states + embedding_dim, n_hiddens)  # Input dim increased by embedding_dim
        self.layer1 = nn.LayerNorm(n_hiddens)
        self.layer2 = nn.Linear(n_hiddens, n_hiddens)
        self.layer3 = nn.LayerNorm(n_hiddens)
        self.layer4 = nn.Linear(n_hiddens, n_hiddens)
        self.layer5 = nn.LayerNorm(n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_options)

        # Initialize the parameters
        nn.init.uniform_(self.fc1.weight, -1 / np.sqrt(256), 1 / np.sqrt(256))
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.uniform_(self.layer2.weight, -1 / np.sqrt(256), 1 / np.sqrt(256))
        nn.init.constant_(self.layer2.bias, 0)
        nn.init.uniform_(self.layer4.weight, -1 / np.sqrt(256), 1 / np.sqrt(256))
        nn.init.constant_(self.layer4.bias, 0)
        nn.init.uniform_(self.fc2.weight, -3e-3, 3e-3)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x, step_idx):
        # Map step_idx to embedding
        step_embedding = self.embedding(step_idx)  # (batch_size, embedding_dim)

        # Concatenate state (x) with the step embedding
        x = torch.cat((x, step_embedding), dim=1)  # (batch_size, n_states + embedding_dim)

        # Forward pass through the network
        x = F.relu(self.fc1(x))  # [batch_size, n_hiddens]
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)

        # Output layer with sigmoid activation
        x = torch.sigmoid(self.fc2(x))  # [batch_size, n_options]
        return x


class Policy_over_options_Learner:
    def __init__(self, n_observations, n_hiddens, n_options, device, args):
        super(Policy_over_options_Learner, self).__init__()
        self.max = 0
        self.state_dim = n_observations
        self.option_dim = n_options
        self.hidden_dim = n_hiddens
        self.replay_buffer_size = int(args['memory_capacity'])  # int
        self.gamma = 0.95
        self.batch_size = args['batch_size']
        self.learning_rate_init = args['high_level_lr']
        self.soft_tau = args['tau']
        self.Termination_lr = self.learning_rate_init * 0.5
        self.Q_lr = self.learning_rate_init
        self.sample_count = 0  # 用于epsilon的衰减计数
        self.epsilon = 0.95
        self.epsilon_start = 0.95
        self.epsilon_end = 0.1
        self.epsilon_decay = 5000
        self.device = device
        self.replay_buffer = None
        self.Q_Net_criterion = None
        self.Termination_Net_scheduler = None
        self.Termination_Net_optimizer = None
        self.Q_Net_scheduler = None
        self.Q_Net_optimizer = None
        self.target_Q_Net = Q_Net(self.state_dim, self.hidden_dim, self.option_dim).to(self.device)
        self.Q_Net = Q_Net(self.state_dim, self.hidden_dim, self.option_dim).to(self.device)
        self.Termination_Net = Termination_Net(self.state_dim, self.hidden_dim, self.option_dim).to(self.device)
        self.target_Termination_Net = Termination_Net(self.state_dim, self.hidden_dim, self.option_dim).to(self.device)
        self.if_set_finite_MDP = True
        self.reply_buffer_type = "PER"
        self.num_time_steps = args['max_steps']

    def init_training(self):
        for target_param, param in zip(self.target_Q_Net.parameters(), self.Q_Net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_Termination_Net.parameters(), self.Termination_Net.parameters()):
            target_param.data.copy_(param.data)

        self.Q_Net_optimizer = optim.Adam(self.Q_Net.parameters(), lr=self.Q_lr)
        # self.Q_Net_scheduler = ReduceLROnPlateau(self.Q_Net_optimizer, mode='min', factor=0.1, patience=5)
        self.Termination_Net_optimizer = optim.Adam(self.Termination_Net.parameters(), lr=self.Termination_lr)
        # self.Termination_Net_scheduler = ReduceLROnPlateau(self.Termination_Net_optimizer, mode='min', factor=0.1, patience=5)
        self.Q_Net_criterion = nn.MSELoss().to(self.device)
        if self.reply_buffer_type == "PER":
            self.replay_buffer = \
                ReplayBuffer_HighOC_Reward_Backpropagation(self.replay_buffer_size, self.num_time_steps)
        else:
            self.replay_buffer = ReplayBuffer_HighOC(self.replay_buffer_size)

    def policy_update(self):
        if self.reply_buffer_type == "PER":
            if len(self.replay_buffer) < 5000:
                return 0, 0
            b_memory = self.replay_buffer.sample(self.batch_size)
            # print([b_memory.step_idx,b_memory.epoch_idx])
            state = torch.FloatTensor(np.array(b_memory.state)).to(self.device)

            next_state = torch.FloatTensor(np.array(b_memory.next_state)).to(self.device)
            option = torch.LongTensor(np.array(b_memory.option)).to(self.device)
            reward = torch.FloatTensor(np.array(b_memory.reward)).to(self.device)
            done = torch.FloatTensor(np.float32(b_memory.done)).to(self.device)
            step_idx = torch.LongTensor(np.array(b_memory.step_idx)).to(self.device)
        else:
            if len(self.replay_buffer) < 2 * self.batch_size:  # 当memory中不满足一个批量时，不更新策略
                return 0, 0
            state, option, reward, next_state, done,step = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(np.array(state)).to(self.device)
            next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
            option = torch.LongTensor(np.array(option)).to(self.device)
            reward = torch.FloatTensor(np.array(reward)).to(self.device)
            done = torch.FloatTensor(np.float32(done)).to(self.device)
            step_idx = torch.LongTensor(np.array(step)).to(self.device)
        self.Q_Net.train()
        self.Termination_Net.train()

        """
        TP_Policy Net Loss Calculation
        """
        option_termination_prob = self.Termination_Net(state, step_idx).gather(1,
                                                                               option.unsqueeze(1))  # [batch_size, 1]
        # advantage = self.Q_Net.cal_Advantage_fun(state).gather(1, option.unsqueeze(1))  # size = [batch_size, 1]
        argmax_o = self.Q_Net(state,step_idx).argmax(1).unsqueeze(-1)
        a = self.Q_Net(state,step_idx).gather(1, argmax_o)
        b = self.Q_Net(state,step_idx).gather(1, option.unsqueeze(1))
        advantage = b - a
        # advantage = b - a
        c = option_termination_prob * advantage.detach() * (1 - done)
        Termination_loss = torch.mean(c)
        with torch.enable_grad():
            self.Termination_Net_optimizer.zero_grad()
            Termination_loss.backward()
            # self.Termination_Net_scheduler.step(Termination_loss)
            torch.nn.utils.clip_grad_norm_(self.Termination_Net.parameters(), max_norm=1.0)
            self.Termination_Net_optimizer.step()

        """
        DQN Loss Calculation
        """
        option_termination_prob = self.target_Termination_Net(next_state, step_idx).gather(1,
                                                                                           option.unsqueeze(
                                                                                               1))  # [batch_size, 1]
        argmax_o = self.Q_Net(next_state, step_idx).argmax(1).unsqueeze(-1)
        # next_state_option_values = self.Q_Net(next_state).detach()  # size = [batch_size, num_of_actions]
        # _, switch_option_batch = next_state_option_values.max(1)  # [batch_size]
        # switch_option_batch = switch_option_batch.unsqueeze(1)  # [batch_size, 1]
        with torch.no_grad():
            temp_a = torch.mul(self.target_Q_Net(next_state, step_idx).gather(1, argmax_o),
                               option_termination_prob.detach())  # [batch_size, 1]
            temp_b = torch.mul(self.target_Q_Net(next_state, step_idx).gather(1, option.unsqueeze(1)),
                               1.0 - option_termination_prob.detach())
            option_value_upon_arrival = temp_a + temp_b

        if self.if_set_finite_MDP:
            expected_state_option_values = (1 - done) * option_value_upon_arrival * self.gamma + reward
        else:
            expected_state_option_values = (option_value_upon_arrival * self.gamma) + reward

        state_option_values = self.Q_Net(state, step_idx).gather(1, option.unsqueeze(1))  # size = [batch_size, 1]
        value_loss = self.Q_Net_criterion(state_option_values, expected_state_option_values)

        with torch.enable_grad():
            self.Q_Net_optimizer.zero_grad()  # 清空之前的梯度
            value_loss.backward()  # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm_(self.Q_Net.parameters(), max_norm=100.0)

            # 打印每个参数的梯度大小（范数）
            # total_grad_norm = 0.0
            # for param in self.Q_Net.parameters():
            #     if param.grad is not None:
            #         # 计算每个参数的L2范数
            #         grad_norm = param.grad.data.norm(2).item()  # L2范数
            #         total_grad_norm += grad_norm ** 2
            # total_grad_norm = total_grad_norm ** 0.5
            # # if total_grad_norm > self.max:
            # #     self.max = total_grad_norm
            # print(f"Total gradient norm: {total_grad_norm}")

            # 更新参数
            self.Q_Net_optimizer.step()

        for target_param, param in zip(self.target_Q_Net.parameters(), self.Q_Net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_Termination_Net.parameters(), self.Termination_Net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        return Termination_loss.item(), value_loss.item()
