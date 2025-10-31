from utils.saveData import save_reward_data
from utils.drawPic import plot_rewards


def save_model(agent, algo_name, low_policy=None, low_level_algo=None, tag='best'):
    if low_policy is None:
        model_save_dir = './train_data/model/' + algo_name + '/' + low_level_algo
        file_str = tag
        if algo_name == 'Ns' or algo_name == 'HighOC_delay' or algo_name == 'HighOC_no_PER' or algo_name == 'HighOC_50ms':
            agent.save_Termination_model(model_save_dir, file_str)
            agent.save_QNet_model(model_save_dir, file_str)
        elif algo_name == 'HighOC_v2':
            agent.save_Option_Critic_model(model_save_dir, file_str)
        else:
            agent.save_T_Policy_model(model_save_dir, file_str)
            agent.save_QNet_model(model_save_dir, file_str)
    else:
        model_save_dir = './train_data/model/' + algo_name + '/' + low_policy
        file_str = tag
        if algo_name == 'TD3_delay' or algo_name == 'TD3':
            agent.save_Actor_model(model_save_dir, file_str)
            agent.save_Critic_1_model(model_save_dir, file_str)
            agent.save_Critic_2_model(model_save_dir, file_str)
        elif algo_name == "PPO":
            agent.save_Policy_model(model_save_dir, file_str)
            agent.save_Value_model(model_save_dir, file_str)
        else:
            agent.save_Actor_model(model_save_dir, file_str)
            agent.save_Critic_model(model_save_dir, file_str)


def save_and_plot_reward_data(reward_array, i_ep, algo_name, device, low_policy=None, low_level_algo=None, tag='train'):
    save_reward_data(reward_array=reward_array, i_episode=i_ep, algo_name=algo_name,
                     low_policy=low_policy, low_level_algo=low_level_algo, tag=tag)
    plot_rewards(rewards=reward_array, algo_name=algo_name, device=str(device),
                 low_policy=low_policy, low_level_algo=low_level_algo, i_ep=i_ep, tag=tag)
