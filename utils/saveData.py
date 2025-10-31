import pandas as pd
from Agents.AgentUtils import *


def save_data(step_array, CPA_rate, CTA_pls, CTA_num, syn,
              QOS_CPA, QOS_CTA, reward_array, complexity,
              i_episode, algo_name, option_array=None, reward_env_array=None, state=None,
              low_policy=None, low_level_algo=None, tag='test'):
    file_name = tag + '_' + str(i_episode) + '.csv'
    if low_policy is None:
        data_save_dir = r"./train_data/data/all_data/" + algo_name + "/" + low_level_algo
        mkdir(data_save_dir)
        data_save_path = os.path.join(data_save_dir, file_name)
        dataframe = pd.DataFrame(
            {
                'step': step_array, 'CPA_rate': CPA_rate, 'CTA_pls': CTA_pls,
                'CTA_num': CTA_num, 'syn': syn, 'QOS_CPA': QOS_CPA, 'QOS_CTA': QOS_CTA, 'complexity': complexity,
                'env_reward': reward_array, 'state': state, 'option': option_array,
            }
        )
    else:
        data_save_dir = r"./train_data/data/all_data/" + algo_name + "/" + low_policy
        mkdir(data_save_dir)
        data_save_path = os.path.join(data_save_dir, file_name)
        dataframe = pd.DataFrame(
            {
                'step': step_array, 'CPA_rate': CPA_rate, 'CTA_pls': CTA_pls,
                'CTA_num': CTA_num, 'syn': syn, 'QOS_CPA': QOS_CPA, 'QOS_CTA': QOS_CTA, 'env_reward': reward_array,
                'state': state, 'option': option_array,
            }
        )
    dataframe.to_csv(data_save_path, index=False, sep=',')


def save_reward_data(reward_array, i_episode, algo_name, low_policy=None, low_level_algo=None, tag='train'):
    if low_policy is None:
        data_save_dir = r"./train_data/data/reward_data/" + algo_name + '/' + low_level_algo
    else:
        data_save_dir = r"./train_data/data/reward_data/" + algo_name + '/' + low_policy
    mkdir(data_save_dir)
    file_name = tag + '_' + str(i_episode) + '.csv'
    data_save_path = os.path.join(data_save_dir, file_name)
    dataframe = pd.DataFrame(
        {'reward': reward_array})
    dataframe.to_csv(data_save_path, index=False, sep=',')


def save_pic_data(step_array, CPA_rate, CTA_pls,
                  env_vehicle_CPA_rate, env_vehicle_CTA_pls, i_episode, algo_name,
                  low_policy=None, low_level_algo=None, tag='test'):
    file_name = tag + '_' + str(i_episode) + '.csv'
    dataframe = pd.DataFrame(
        {
            'step': step_array, 'position_x': CPA_rate, 'position_y': CTA_pls,
            'env_vehicle_CPA_rate': env_vehicle_CPA_rate,
            'env_vehicle_CTA_pls': env_vehicle_CTA_pls
        }
    )
    if low_policy is None:
        data_save_dir = r"./train_data/data/pic_data/" + algo_name + "/" + low_level_algo
        mkdir(data_save_dir)
        data_save_path = os.path.join(data_save_dir, file_name)
    else:
        data_save_dir = r"./train_data/data/pic_data/" + algo_name + "/" + low_policy
        mkdir(data_save_dir)
        data_save_path = os.path.join(data_save_dir, file_name)

    dataframe.to_csv(data_save_path, index=False, sep=',')


def save_AoI_data(AoI_array, i_episode, algo_name, low_policy=None, low_level_algo=None, tag='train'):
    if low_policy is None:
        data_save_dir = r"./train_data/data/AoI_data/" + algo_name + '/' + low_level_algo
    else:
        data_save_dir = r"./train_data/data/AoI_data/" + algo_name + '/' + low_policy
    mkdir(data_save_dir)
    file_name = tag + '_' + str(i_episode) + '.csv'
    data_save_path = os.path.join(data_save_dir, file_name)
    dataframe = pd.DataFrame(
        {'AoI': AoI_array})
    dataframe.to_csv(data_save_path, index=False, sep=',')
