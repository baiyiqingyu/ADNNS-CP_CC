# -*- coding: utf-8 -*-
import os, copy, math, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.Replay_Memory import ReplayBuffer_HighOC, ReplayBuffer_HighOC_Reward_Backpropagation


# 如果没用 PER，可以删掉上面这行的 *_Reward_Backpropagation 导入

# -----------------------------
# Dueling DQN: [256,128,128], relu, relu, relu, linear
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim, dueling=True):
        """
        hidden_dims: (256, 128, 128)
        dueling: True -> 输出 V 与 A，再合成 Q；False -> 普通 DQN
        """
        super().__init__()
        self.dueling = dueling
        h1, h2, h3 = hidden_dims

        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)

        if dueling:
            self.fc_A = nn.Linear(h3, action_dim)  # Advantage 头
            self.fc_V = nn.Linear(h3, 1)  # Value 头
        else:
            self.fc_out = nn.Linear(h3, action_dim)

        # 初始化：和你以前风格一致
        def uni_init(layer, fan=256):
            nn.init.uniform_(layer.weight, -1 / np.sqrt(fan), 1 / np.sqrt(fan))
            nn.init.zeros_(layer.bias)

        uni_init(self.fc1, 256);
        uni_init(self.fc2, 256);
        uni_init(self.fc3, 256)
        if dueling:
            nn.init.uniform_(self.fc_A.weight, -3e-3, 3e-3);
            nn.init.zeros_(self.fc_A.bias)
            nn.init.uniform_(self.fc_V.weight, -3e-3, 3e-3);
            nn.init.zeros_(self.fc_V.bias)
        else:
            nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3);
            nn.init.zeros_(self.fc_out.bias)

    def features(self, x):
        # relu, relu, relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def forward(self, x):
        x = self.features(x)
        if self.dueling:
            A = self.fc_A(x)
            V = self.fc_V(x)
            Q = V + (A - A.mean(dim=-1, keepdim=True))
            return Q
        else:
            return self.fc_out(x)

    @torch.no_grad()
    def forward_components(self, x):
        """调试用：一次前向返回 (Q, A, V, A_centered)。"""
        x = self.features(x)
        if self.dueling:
            A = self.fc_A(x);
            V = self.fc_V(x)
            A_centered = A - A.mean(dim=-1, keepdim=True)
            return V + A_centered, A, V, A_centered
        else:
            Q = self.fc_out(x)
            return Q, None, None, None


class DQN_Agent:
    def __init__(self, n_observations, n_hiddens, n_actions, device, args):
        # n_hiddens 仅占位（兼容你的旧构造）；真正的结构用 hidden_dims
        self.device = device
        self.state_dim = n_observations
        self.action_dim = n_actions

        # -------- 表格里的超参 --------
        self.gamma = args.get('gamma', 0.95)
        self.batch_size = args.get('batch_size', 128)
        self.learning_rate = args.get('high_level_lr', 2e-5)
        self.tau = args.get('tau', 1e-3)
        self.buffer_size = int(args.get('memory_capacity', 5000))
        self.double_dqn = args.get('double_dqn', True)
        self.dueling = args.get('dueling', True)

        # ε-greedy: 0.95 -> 0.01, decay=500
        self.sample_count = 0
        self.eps_start = args.get('eps_start', 0.95)
        self.eps_end = args.get('eps_end', 0.01)
        self.eps_decay = args.get('eps_decay', 500)

        # 网络
        hidden_dims = args.get('hidden_dims', (256, 128, 128))
        self.online_net = DQN(self.state_dim, hidden_dims, self.action_dim, dueling=self.dueling).to(self.device)
        self.target_net = DQN(self.state_dim, hidden_dims, self.action_dim, dueling=self.dueling).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss().to(self.device)

        replay_type = args.get('replay_type', 'Uniform')
        if replay_type == 'PER':
            self.replay_buffer = ReplayBuffer_HighOC_Reward_Backpropagation(
                self.buffer_size, args.get('max_steps', 5000)
            )
        else:
            self.replay_buffer = ReplayBuffer_HighOC(self.buffer_size)

    # ---------- 交互 ----------
    @torch.no_grad()
    def select_action(self, state_np):
        self.sample_count += 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1.0 * self.sample_count / self.eps_decay)
        if np.random.rand() < eps:  # 随机探索
            return np.random.randint(self.action_dim)
        s = torch.as_tensor(np.asarray(state_np, np.float32), device=self.device).unsqueeze(0)
        q = self.online_net(s)
        q = q + 1e-6 * torch.randn_like(q)  # 打破平手
        return int(q.argmax(dim=1).item())

    def push_transition(self, s, a, r, ns, done, step, **kwargs):
        self.replay_buffer.push(s, a, r, ns, done, step)

    # ---------- 训练一步 ----------
    def update(self):
        # 样本不足不更新
        if len(self.replay_buffer) < 10 * self.batch_size:
            return 0.0

        # === 采样 ===
        s, a, r, ns, d,step = self.replay_buffer.sample(self.batch_size)

        # === 张量化 ===
        s = torch.as_tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.array(a), dtype=torch.long, device=self.device)
        r = torch.as_tensor(np.array(r), dtype=torch.float32, device=self.device).view(-1)
        ns = torch.as_tensor(np.array(ns), dtype=torch.float32, device=self.device)
        d = torch.as_tensor(np.array(d), dtype=torch.float32, device=self.device).view(-1)

        # 确保处于训练模式（外部若设成 eval() 会影响 LN/Dropout 等）
        self.online_net.train()
        self.target_net.train()

        # === 当前 Q(s,a) ===
        q_all = self.online_net(s)  # [B, A]
        q_sa = q_all.gather(1, a.view(-1, 1)).squeeze(1)  # [B]

        # === 目标 y ===
        with torch.no_grad():
            if self.double_dqn:
                na = self.online_net(ns).argmax(dim=1, keepdim=True)  # [B,1]
                nq = self.target_net(ns).gather(1, na).squeeze(1)  # [B]
            else:
                nq = self.target_net(ns).max(dim=1)[0]  # [B]
            target = r + (1.0 - d) * self.gamma * nq  # [B]

            # 目标数值检查（可选）
            if not torch.isfinite(target).all():
                # 打印一份诊断信息再早退出，避免把 NaN/Inf 送进优化器
                bad = ~torch.isfinite(target)
                print("[WARN] non-finite target detected:",
                      {"r": r[bad][:5].tolist(), "d": d[bad][:5].tolist(), "nq": nq[bad][:5].tolist()})
                return 0.0

        # === 损失（Huber 更稳；用 MSE 也可以）===
        # self.criterion 可换成 nn.SmoothL1Loss() 预先定义；这里演示直接建
        loss_fn = getattr(self, "criterion", nn.SmoothL1Loss().to(self.device))
        loss = loss_fn(q_sa, target)

        # === 反传 ===
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # === 软更新 ===
        with torch.no_grad():
            tau = self.tau
            for tp, p in zip(self.target_net.parameters(), self.online_net.parameters()):
                tp.copy_(tp * (1.0 - tau) + p * tau)

        return float(loss.item())

    # 兼容你旧接口
    def policy_update(self):
        return self.update()
