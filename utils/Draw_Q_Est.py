import matplotlib
import torch

from scipy import stats
import numpy as np
import os

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd


def sav_Q_val_data(times_array, rewards_array, Q_est_array, Q_sample_val_array, file_str, i_vehicle):
    dataframe = pd.DataFrame(
        {'times_array': times_array, 'rewards_array': rewards_array, 'Q_est_array': Q_est_array,
         'Q_sample_val_array': Q_sample_val_array})
    dataframe.to_csv('./train_data/Q_value/test_' + str(file_str) + '_vehicle_' + str(i_vehicle) + '.csv', index=False,
                     sep=',')


class draw_Q_value(object):
    def __init__(self, gamma, capacity=10000):
        self.gamma = gamma
        self.capacity = capacity
        self.buffer = []
        self.buffer_position = 0
        self.Q_est_array = []
        self.Q_sample_val_array = []
        self.times_array = []
        self.rewards_array = []
        self.position = 0
        self.tra_len = 82

    def push(self, time_idx, reward, Q_est, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.buffer_position] = (time_idx, reward, Q_est, done)
        self.buffer_position = (self.buffer_position + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.buffer_position = 0

    def cal_Q_sample_val(self, Q_vale_est_final_next_state):
        batch = self.buffer[0:10000]

        # print(self.buffer, batch)
        time_idx, rewards, Q_est, done = map(np.stack, zip(*batch))
        """
        Calculate cumulative reward
        """
        Q_sample_val = []
        temp = 0.0
        count = 0
        validation_sample_num = 0
        discounted_return = 0
        for i in range(10000):
            if validation_sample_num >= self.tra_len:
                self.Q_sample_val_array.append(discounted_return)
                validation_sample_num = 0
                discounted_return = 0
            if validation_sample_num == 0:
                self.Q_est_array.append(Q_est[i])
                self.times_array.append(time_idx[i])
                self.rewards_array.append(rewards[i])
            discounted_return += rewards[i] * (self.gamma ** validation_sample_num)
            validation_sample_num += 1
            if validation_sample_num == self.tra_len:
                discounted_return += Q_est[i + 1] * self.gamma ** validation_sample_num if done[i] == 0 else 0
            if done[i] == 1:
                validation_sample_num = self.tra_len

        self.Q_sample_val_array.append(discounted_return)

        # for reward in rewards[::-1]:
        #
        #     if count == 0:
        #         temp = temp * self.gamma + reward + Q_vale_est_final_next_state * self.gamma
        #     else:
        #         temp = temp * self.gamma + reward
        #     count += 1
        #     Q_sample_val.append(temp)
        # Q_sample_val.reverse()
        #
        # for i in range(10000):
        #     self.Q_sample_val_array.append(Q_sample_val[i])
        #     self.Q_est_array.append(Q_est[i])
        #     self.times_array.append(time_idx[i])
        #     self.rewards_array.append(rewards[i])
        #     self.position += 1

        """
        Reset Buffer
        """
        self.reset()

    def draw_scatter(self, file_str=None):
        if self.Q_est_array is not None and self.Q_sample_val_array is not None:
            max_value_Q_sample_val = max(self.Q_sample_val_array)  # Q_sample_val_array
            min_value_Q_sample_val = min(self.Q_sample_val_array)
            max_value_Q_est_array = max(self.Q_est_array)  # Q_est_array
            min_value_Q_est_array = min(self.Q_est_array)
            """
            Plot line y = x for calibration
            """
            plt.figure(figsize=(16, 9))
            max_value_y_x = max([max_value_Q_sample_val, max_value_Q_est_array])
            min_value_y_x = min([min_value_Q_sample_val, min_value_Q_est_array])
            x_array = np.linspace(min_value_y_x, max_value_y_x, 100)
            y_array = x_array
            plt.plot(x_array, y_array, color='red', linewidth='3.0', linestyle='--')
            """
            Scatter Plot
            """
            plt.scatter(self.Q_sample_val_array, self.Q_est_array, c='blue', s=30)
            """
            Format of figure
            """
            plt.xlabel('Return', fontsize=20)
            plt.ylabel('Estimation of Q value function', fontsize=20)
            if file_str is not None:
                title_str = 'Scenario: ' + file_str
                plt.title(title_str, fontsize=20)
            plt.grid()
            plt.tight_layout()
            if file_str is not None:
                plt.savefig('./train_data/' + file_str + '_return_vs_Q_val_est.png')
            else:
                plt.savefig('./train_data/Q_value/return_vs_Q_val_est.png')
        else:
            print("Error in plotting figure")

    def saving_data(self, vehicle_idx, file_str):
        sav_Q_val_data(self.times_array, self.rewards_array, self.Q_est_array, self.Q_sample_val_array,
                       file_str, vehicle_idx)
