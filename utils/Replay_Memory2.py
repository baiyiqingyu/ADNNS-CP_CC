import math
import random
import numpy as np
from collections import deque, namedtuple
import torch

Transition = namedtuple('Transition', ('state', 'option', 'reward', 'next_state', 'done', 'step_idx', 'epoch_idx'))


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Store data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class ReplayBuffer_HighOC:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, np.array([reward]), next_state, np.array([done])))

    def sample(self, batch_size, sequential=False):
        if sequential:  # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done
        else:
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer_HighOC_Reward_Backpropagation(object):
    # 论文中的高优起点 β；λ 为回传衰减系数
    beta = 10.0  # 建议 3~10；原来 100 太大容易垄断
    lambda_decay = 0.3  # 论文中的 λ (0,1)
    p_min = 1.0  # 最小优先级，避免“采不到”
    p_max = 10.0  # 上限裁剪，防止垄断
    backprop_K = 1  # 每次采样向前推进的步数；想更激进可设 5/10

    def __init__(self, capacity, time_steps, transition=Transition):
        self.Transition = transition
        self.capacity = int(math.floor(capacity / time_steps) * time_steps)
        self.tree = SumTree(self.capacity)
        self.time_steps = time_steps

    # ---- 索引换算辅助 ----
    def _leaf_to_dataptr(self, tree_idx):
        # 叶子索引 -> data 指针（0..capacity-1）
        return tree_idx - (self.tree.capacity - 1)

    def _dataptr_to_leaf(self, data_ptr):
        # data 指针 -> 叶子索引（capacity-1 .. 2*capacity-2）
        return data_ptr + (self.tree.capacity - 1)

    # ============== 正确的 push（按 奖励/终止 触发高优）=============
    def push(self, state, action, reward, next_state, done, step_idx, epoch_idx):
        """
        论文要求：若 r != 0 或 terminal，则该 transition 初始优先级设为 β，否则为 1
        """
        tr = self.Transition(state, action, np.array([reward]), next_state, np.array([done]),
                             step_idx, epoch_idx)
        p0 = self.beta if (reward != 0 or done) else 1.0
        self.tree.add(p0, tr)

    def __len__(self):
        return self.tree.data_pointer

    # ============== 正确的 sample（RBP 回传核心逻辑）=============
    def sample(self, batch_size):
        """
        Stratified 采样；若采到 priority>1 的叶子：
          1) 将该叶子的优先级重置为 1
          2) 向同 episode 的前驱连续回传 K 步（几何衰减、裁剪）
          3) 若已到头（无前驱/跨 episode），则从当前点向前查找本 episode 的 奖励/终止点，
             在其上以 λ·p 作为新一轮回传的起点（论文中的“绕回”）
        """
        pri_seg = self.tree.total_p / batch_size
        batch = []
        upd_ops = []  # [(leaf_idx, new_priority), ...]

        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            tree_idx, p, tr = self.tree.get_leaf(v)
            if tr == 0:
                continue

            batch.append(tr)

            # 仅当采到高优样本时触发回传
            if p > 1.0:
                # ① 当前叶子重置为 1.0
                upd_ops.append((tree_idx, 1.0))
                # print(f"High priority sample: epoch={tr.epoch_idx}, step={tr.step_idx}, p={p:.2f}")
                # ② 向前回传 K 步（同 episode + 连续 step）
                data_ptr = self._leaf_to_dataptr(tree_idx)
                curr = tr
                advanced = False
                p_seed = float(p)  # 本轮回传的种子优先级

                for k in range(1, self.backprop_K + 1):
                    prev_ptr = data_ptr - k
                    if prev_ptr < 0:
                        break
                    prev = self.tree.data[prev_ptr]
                    if prev == 0:
                        break
                    # 同一 episode 且 step 连续
                    if (prev.epoch_idx == curr.epoch_idx) and (prev.step_idx == curr.step_idx - k):
                        prev_leaf = self._dataptr_to_leaf(prev_ptr)
                        p_k = max(self.p_min, min(p_seed * (self.lambda_decay ** k), self.p_max))
                        upd_ops.append((prev_leaf, p_k))
                        advanced = True
                    else:
                        break  # 跨 episode 或不连续则停止

                # ③ 若这次没能向前推进（已经在段首/episode 首），则“绕回”到本 episode 的奖励/终止点，置 λ·p
                if not advanced:
                    # 从当前点向后找本 episode 的第一个 (reward!=0 或 done==1) 的 transition
                    ptr = data_ptr + 1
                    while ptr < self.tree.capacity:
                        nxt = self.tree.data[ptr]
                        if nxt == 0 or nxt.epoch_idx != curr.epoch_idx:
                            break  # 已出本 episode
                        if (nxt.reward[0] != 0) or (nxt.done[0] == 1):
                            leaf = self._dataptr_to_leaf(ptr)
                            p_new = max(self.p_min, min(self.lambda_decay * p_seed, self.p_max))
                            upd_ops.append((leaf, p_new))
                            break
                        ptr += 1

        # 统一更新 SumTree
        for idx, val in upd_ops:
            self.tree.update(idx, float(val))

        return self.Transition(*zip(*batch))

    def save_to_disk(self, filename):
        """
            Save the memory to a file
        """
        with open(filename, 'wb') as f:
            pickle.dump((self.tree.capacity, self.tree.tree, self.tree.data, self.tree.data_pointer), f)

    def load_from_disk(self, filename):
        with open(filename, 'rb') as f:
            (capacity, tree, data, data_pointer) = pickle.load(f)
            self.tree.capacity = capacity
            self.tree.tree = tree
            self.tree.data = data
            self.tree.data_pointer = data_pointer
