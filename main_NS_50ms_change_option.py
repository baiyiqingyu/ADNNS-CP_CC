import numpy as np
import torch

from Environment.NS_env_V2X_baseline import NS_Env

from utils.parameters import args
from utils.init import init_device
from Agents.High_Level_Agent_DQN import High_Level_Agent
from utils.saveData import save_data
from utils.train_utils import save_and_plot_reward_data, save_model
from tensorboardX import SummaryWriter
from utils.Draw_Q_Est import draw_Q_value
import torch.nn.functional as F


# def get_cdf(values):
#     """
#     Computes the Cumulative Distribution Function (CDF) of the input vector
#     """
#     values = np.array(values).flatten()
#     n = len(values)
#     sorted_val = np.sort(values)
#     cumulative_prob = np.arange(1, n + 1) / n
#     return sorted_val, cumulative_prob

# 训练
def train(high_agent, environment):
    writer = SummaryWriter()
    print('训练开始...')
    episode_reward_array = []
    episode_reward_test_array = []

    # for i_ep in range(args['train_episode']):
    for i_ep in range(args['train_episode']):
        environment.reset()
        ep_reward = 0
        agent_state = environment.getState_agent()
        terminate_loss, q_loss = 0, 0
        option = high_agent.get_option_during_train(agent_state)
        if option == 0:
            environment.SCs_of_CPA_pro = 3
            environment.SCs_share = 6
            environment.SCs_of_CTA = 1
            environment.option = 0
            environment.w1 = 0.2
        elif option == 1:
            environment.SCs_of_CPA_pro = 2
            environment.SCs_share = 5
            environment.SCs_of_CTA = 3
            environment.option = 1
            environment.w1 = 0.194
        elif option == 2:
            environment.SCs_of_CPA_pro = 2
            environment.SCs_share = 6
            environment.SCs_of_CTA = 2
            environment.option = 1
            environment.w1 = 0.197

        for TTI in range(args['max_steps']):
            environment.step_slot()
            for mTTI in range(args['num_of_mslots']):
                environment.step_mini_slot(TTI * 7 + mTTI + 1)
            environment.calculate_slot(TTI + 1)
            if (TTI + 1) % 20 == 0 and TTI != 0:
                next_agent_state = environment.getState_agent()
                next_state = next_agent_state.flatten()
                reward_per_step = environment.reward
                ep_reward += reward_per_step
                done = environment.done
                high_agent.add2replay_memory(agent_state.copy(), option, reward_per_step, next_state.copy(), done=done,
                                             step_idx=TTI, epoch_idx=i_ep)
                # with torch.no_grad():
                #     s = torch.as_tensor(agent_state.copy(), dtype=torch.float32, device=high_agent.device).unsqueeze(0)
                #     Q, A, V, Acent = high_agent.learner.online_net.forward_components(s)
                #     print("Q:", Q.cpu().numpy())
                #     print("A-mean(A):", Acent.cpu().numpy())
                #     print("action:", int(Q.argmax(dim=1).item()))
                agent_state = next_agent_state
                termination_loss_per_step, q_loss_per_step = high_agent.policy_update()
                terminate_loss += termination_loss_per_step
                q_loss += q_loss_per_step
                # print(option)
                writer.add_scalar('Train/TerminateLoss', terminate_loss, i_ep)
                writer.add_scalar('Train/QLoss', q_loss, i_ep)
                option = high_agent.get_option_during_train(agent_state, option)
                if option == 0:
                    environment.SCs_of_CPA_pro = 3
                    environment.SCs_share = 6
                    environment.SCs_of_CTA = 1
                    environment.option = 0
                    environment.w1 = 0.2
                elif option == 1:
                    environment.SCs_of_CPA_pro = 2
                    environment.SCs_share = 5
                    environment.SCs_of_CTA = 3
                    environment.option = 1
                    environment.w1 = 0.194
                elif option == 2:
                    environment.SCs_of_CPA_pro = 2
                    environment.SCs_share = 6
                    environment.SCs_of_CTA = 2
                    environment.option = 1
                    environment.w1 = 0.197
                environment.step_20ms()

            if (TTI + 1) % args['Slidingwindow_slot'] == 0 and TTI != 0:
                environment.calculate_long_time(TTI + 1)
                environment.step_long_time((TTI + 1) // 50)
        writer.add_scalar('Train/Reward', ep_reward, i_ep)
        print(f"回合：{i_ep + 1}/{args['train_episode']}，奖励：{ep_reward:.2f}")
        episode_reward_array.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            (reward, CPA_rate, CTA_pls, CTA_num, syn, QOS_CPA, QOS_CTA, step_array,
             option_array, state_array, complexity) = (
                hwh_test(high_agent, environment))
            # save_model(agent=high_agent, algo_name=args['algo_name'], low_level_algo='Ns', tag=str(i_ep))
            save_data(step_array=step_array, CPA_rate=CPA_rate, CTA_pls=CTA_pls,
                      CTA_num=CTA_num, syn=syn,
                      QOS_CPA=QOS_CPA, QOS_CTA=QOS_CTA, reward_array=reward,
                      i_episode=i_ep, algo_name=args['algo_name'], state=state_array, option_array=option_array,
                      low_level_algo='Ns', complexity=complexity
                      )
            episode_reward_test_array.append(reward)

            writer.add_scalar('Test/Reward', reward, i_ep)
        if (i_ep + 1) % 100 == 0:
            save_and_plot_reward_data(reward_array=episode_reward_array, i_ep=i_ep, algo_name=args['algo_name'],
                                      device=str(device), low_level_algo='Ns')
            save_and_plot_reward_data(reward_array=episode_reward_test_array, i_ep=i_ep, algo_name=args['algo_name'],
                                      device=str(device), tag='test', low_level_algo='Ns')
    writer.close()
    print('训练结束...')


def hwh_test(high_agent, environment):
    print('测试开始...')
    high_agent.learner.online_net.eval()
    high_agent.learner.target_net.eval()
    max_reward = 0
    all_reward = 0
    step_array = []
    CTA_pls = []
    CTA_num = []
    complexity = []
    qos_CPA = []
    qos_CTA = []
    CPA_delay_dist = []
    CPA_rate = []
    option_array = []
    reward_array = []
    state_array = []
    syn = []

    q_visualizer = draw_Q_value(gamma=0.95)
    for i_ep_test in range(args['test_episode']):
        step_array.append([])
        option_array.append([])
        reward_array.append([])
        CPA_rate.append([])
        CTA_pls.append([])
        CTA_num.append([])
        complexity.append([])
        qos_CPA.append([])
        qos_CTA.append([])
        CPA_delay_dist.append([])
        state_array.append([])
        syn.append([])
    for i_ep_test in range(5):
        com = 0
        environment.reset()
        agent_state_test = environment.getState_agent()
        ep_reward_test = 0
        option_test = 0
        option_test = high_agent.get_option_during_test(agent_state_test, option_test)

        if option_test == 0:
            environment.SCs_of_CPA_pro = 3
            environment.SCs_share = 6
            environment.SCs_of_CTA = 1
            environment.option = 0
            environment.w1 = 0.2
        elif option_test == 1:
            environment.SCs_of_CPA_pro = 2
            environment.SCs_share = 5
            environment.SCs_of_CTA = 3
            environment.option = 1
            environment.w1 = 0.194
        elif option_test == 2:
            environment.SCs_of_CPA_pro = 2
            environment.SCs_share = 6
            environment.SCs_of_CTA = 2
            environment.option = 2
            environment.w1 = 0.197
        for TTI in range(args['max_steps']):
            environment.step_slot()
            for mTTI in range(args['num_of_mslots']):
                environment.step_mini_slot(TTI * 7 + mTTI + 1)
            environment.calculate_slot(TTI + 1)
            if (TTI + 1) % 20 == 0 and TTI != 0:
                next_agent_state_test = environment.getState_agent()
                agent_state_test = next_agent_state_test
                reward_test = environment.reward
                ep_reward_test += reward_test
                # print(option_test)
                step_array[0].append(TTI)
                a = []
                for i in range(environment.CP_num):
                    a.append(environment.CPA[i].rate_all)
                CPA_rate[0].append(a)
                CTA_pls[0].append(np.sum(environment.CTA_pls))
                CTA_num[0].append(np.sum(environment.windows.num_CTA))
                complexity[0].append(environment.windows.complexity)
                com += environment.windows.complexity
                option_array[0].append(option_test)
                reward_array[0].append(environment.au)
                state_array[0].append(0)
                qos_CPA[0].append(environment.qos_CP)
                qos_CTA[0].append(environment.qos_CC)

                if TTI != args['max_steps'] - 1:
                    syn[0].append(0)

                option_test = high_agent.get_option_during_test(agent_state_test, option_test)
                if option_test == 0:
                    environment.SCs_of_CPA_pro = 3
                    environment.SCs_share = 6
                    environment.SCs_of_CTA = 1
                    environment.option = 0
                    environment.w1 = 0.2
                elif option_test == 1:
                    environment.SCs_of_CPA_pro = 2
                    environment.SCs_share = 5
                    environment.SCs_of_CTA = 3
                    environment.option = 1
                    environment.w1 = 0.194
                elif option_test == 2:
                    environment.SCs_of_CPA_pro = 2
                    environment.SCs_share = 6
                    environment.SCs_of_CTA = 2
                    environment.option = 1
                    environment.w1 = 0.197

                environment.step_20ms()
            if (TTI + 1) % args['Slidingwindow_slot'] == 0 and TTI != 0:
                environment.calculate_long_time(TTI + 1)
                environment.step_long_time((TTI + 1) // 50)

                if TTI + 1 == args['max_steps']:
                    syn[0].append(environment.windows.CPA_Synchronicity / 100)

        print(f"回合：{i_ep_test + 1}/{args['test_episode']}，奖励：{ep_reward_test:.2f}")
        all_reward += ep_reward_test
        if i_ep_test == 0:
            max_reward = ep_reward_test
            max_index = i_ep_test
        if ep_reward_test > max_reward:
            max_reward = ep_reward_test
            max_index = i_ep_test
    print('测试结束...')
    return (
        all_reward / args['test_episode'], CPA_rate[0], CTA_pls[0], CTA_num[0],
        syn[0], qos_CPA[0], qos_CTA[0], step_array[0],
        option_array[0],
        [0 for i in range(1250)], complexity[0])


if __name__ == '__main__':
    device = init_device(0)
    low_level_agent = None
    args.update({'memory_capacity': 262144})
    args.update({'train_episode': 5000})
    env = NS_Env(args)

    args.update({'algo_name': 'HighOC_50ms'})

    high_level_agent = High_Level_Agent(0, device, args)
    high_level_agent.learner.reply_buffer_type = 'no_PER'
    high_level_agent.init_training()

    train(high_level_agent, env)
