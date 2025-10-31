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

    def push(self, state, action, reward, next_state, done, step):
        self.buffer.append((state, action, np.array([reward]), next_state, np.array([done]), np.array(step)))

    def sample(self, batch_size, sequential=False):
        if sequential:  # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            state, action, reward, next_state, done = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done
        else:
            batch = random.sample(self.buffer, batch_size)
            state, action, reward, next_state, done, step = map(np.stack, zip(*batch))
            return state, action, reward, next_state, done, step

    def __len__(self):
        return len(self.buffer)


class ReplayBuffer_HighOC_Reward_Backpropagation(object):
    beta = 100.0  # importance-sampling
    decay_factor = 0.1

    def __init__(self, capacity, time_steps, transition=Transition):
        self.Transition = transition
        self.capacity = int(math.floor(capacity / time_steps) * time_steps)

        self.tree = SumTree(self.capacity)
        self.temp_buffer = np.zeros(time_steps, dtype=object)  # for all transitions in the buffer
        self.temp_buffer_pointer = 0
        self.time_steps = time_steps

    def push(self, state, action, reward, next_state, done, step_idx, epoch_idx):
        """Save a transition"""
        transition_data = self.Transition(state, action, np.array([reward]), next_state, np.array([done]), step_idx,
                                          epoch_idx)

        self.temp_buffer[self.temp_buffer_pointer] = transition_data
        self.temp_buffer_pointer += 1
        if self.temp_buffer_pointer >= self.time_steps:  # if we store a whole trajectory of MDP
            self.temp_buffer_pointer = 0
            for i in range(self.time_steps):
                if i == self.time_steps - 1:
                    self.tree.add(self.beta, self.temp_buffer[i])  # set beta for end of traj
                else:
                    self.tree.add(1.0, self.temp_buffer[i])  # set 1.0 for others

    def __len__(self):
        return self.tree.data_pointer

    def sample(self, batch_size):
        b_idx, b_memory, priorities = [], [], []
        pri_seg = self.tree.total_p / batch_size  # priority segment

        update_tree_index = np.zeros(batch_size * 3, int)
        update_properties = np.zeros(batch_size * 3, int)

        update_data_pointer = 0
        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            tree_idx, priority, data = self.tree.get_leaf(v)
            if data != 0:
                b_memory.append(data)
                """
                        Renew of priorities
                """
                if priority > 1.0:
                    transition_tree_idx = tree_idx
                    transition_data_pointer = transition_tree_idx - self.tree.capacity + 1
                    transition_data_step_idx = self.tree.data[transition_data_pointer].step_idx
                    update_tree_index[update_data_pointer] = transition_data_step_idx
                    update_properties[update_data_pointer] = 1.0
                    update_data_pointer = update_data_pointer + 1
                    # self.tree.update(transition_tree_idx, 1.0)
                    if transition_data_step_idx > 0:
                        pre_transition_data_pointer = transition_data_pointer - 1
                        pre_transition_tree_idx = pre_transition_data_pointer + self.tree.capacity - 1
                        pre_transition_priority = priority
                        update_tree_index[update_data_pointer] = pre_transition_tree_idx
                        update_properties[update_data_pointer] = pre_transition_priority
                        self.tree.update(transition_tree_idx, 1)
                        self.tree.update(pre_transition_tree_idx, pre_transition_priority)
                        update_data_pointer = update_data_pointer + 1

                        # self.tree.update(pre_transition_tree_idx, pre_transition_priority)
                    else:
                        # print("111")
                        after_transition_data_pointer = transition_data_pointer + self.time_steps - 1
                        after_transition_tree_idx = after_transition_data_pointer + self.tree.capacity - 1
                        if self.tree.data[after_transition_data_pointer].step_idx == self.tree.data[
                            transition_data_pointer].step_idx + self.time_steps - 1 and \
                                self.tree.data[after_transition_data_pointer].epoch_idx == self.tree.data[
                            transition_data_pointer].epoch_idx:
                            after_transition_priority = max(self.decay_factor * priority, 1.0)

                            update_tree_index[update_data_pointer] = after_transition_tree_idx
                            update_properties[update_data_pointer] = after_transition_priority

                            self.tree.update(transition_tree_idx, 1)
                            self.tree.update(after_transition_tree_idx, after_transition_priority)

                            update_data_pointer = update_data_pointer + 1
                            # self.tree.update(after_transition_tree_idx, after_transition_priority)
                        else:
                            raise ValueError('wrong after transition')

        """
        Renew of Tree
        """
        # for i in range(update_data_pointer):
        #     self.tree.update(update_tree_index[i], update_properties[i])
        # print(self.tree.tree)
        return self.Transition(*zip(*b_memory))

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
