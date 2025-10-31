import numpy as np
import torch
import torch.nn as nn

from Environment.NS_env_V2X import NS_Env
from utils.parameters import args
from utils.init import init_device
from Agents.High_Level_Agent import High_Level_Agent
from utils.saveData import save_data
from utils.train_utils import save_and_plot_reward_data, save_model
from tensorboardX import SummaryWriter
from utils.Draw_Q_Est import draw_Q_value


# 加载已训练模型
def load_model(Q_net, Termination_net, model_path_Q, model_path_Termination):
    # 加载Q_net和Termination_net的权重
    Q_net.load_state_dict(torch.load(model_path_Q, weights_only=True))
    Termination_net.load_state_dict(torch.load(model_path_Termination, weights_only=True))

    # 设置为评估模式
    Q_net.eval()
    Termination_net.eval()

    print(f"Q_Net模型从 {model_path_Q} 加载完成")
    print(f"Termination_Net模型从 {model_path_Termination} 加载完成")

    return Q_net, Termination_net


# 测试部分
def hwh_test(high_agent, environment, test_count=10):
    print('测试开始...')
    high_agent.learner.Q_Net.eval()
    high_agent.learner.Termination_Net.eval()

    all_reward = 0
    max_reward = 0
    max_index = 0
    step_array = []
    CPA_rate = []
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

    # 初始化结果存储
    for i_ep_test in range(test_count):
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

    for i_ep_test in range(test_count):
        environment.reset()
        agent_state_test = environment.getState_agent(reset=1, step=0)
        ep_reward_test = 0
        option_test = 0

        for TTI in range(args['max_steps']):
            option_test = high_agent.get_option_during_test(agent_state_test, option_test, step=TTI)
            if option_test == 0:
                args.update({'low_policy': 'CP'})
                environment.k1 = environment.CC_transmission_rate_CP
                environment.k2 = environment.CP_packet_loss_CP
                environment.k3 = environment.avg_CP
                environment.k4 = environment.complexity_CP
                environment.beta = 0.05
                environment.option = 0
                environment.complexity_max = 3
            elif option_test == 1:
                args.update({'low_policy': 'CC'})
                environment.k1 = environment.CC_transmission_rate_CC
                environment.k2 = environment.CP_packet_loss_CC
                environment.k3 = environment.avg_CC
                environment.k4 = environment.complexity_CC
                environment.beta = 0.05
                environment.option = 1
                environment.complexity_max = 0.5
            environment.step_slot()
            for mTTI in range(args['num_of_mslots']):
                environment.step_mini_slot((TTI * 7 + mTTI) % 350 + 1)
            if (TTI + 1) % args['Slidingwindow_slot'] == 0 and TTI != 0:
                environment.calculate_long_time()

            environment.calculate_slot(TTI + 1)

            done = environment.done
            with torch.no_grad():
                state = torch.from_numpy(agent_state_test).float().unsqueeze(0).to(device)
                a = TTI
                step_idx = torch.LongTensor([a]).to(device)
                option_tensor = torch.tensor([[option_test]], dtype=torch.long).to(device)
                Q_val = high_agent.learner.Q_Net(state, step_idx).gather(1, option_tensor).item()

            q_visualizer.push(time_idx=TTI, reward=environment.reward, Q_est=Q_val, done=done)

            next_agent_state_test = environment.getState_agent(step=TTI + 1)
            agent_state_test = next_agent_state_test
            reward_test = environment.reward
            ep_reward_test += reward_test

            step_array[i_ep_test].append(TTI)
            CPA_rate[i_ep_test].append(environment.CPA_rate)
            CTA_pls[i_ep_test].append(np.sum(environment.pls_CC_num))
            CTA_num[i_ep_test].append(np.sum(environment.act_CC_num))
            qos_CPA[i_ep_test].append(environment.qos_CP)
            qos_CTA[i_ep_test].append(environment.qos_CC)
            complexity[i_ep_test].append(environment.complexity)
            option_array[i_ep_test].append(option_test)
            reward_array[i_ep_test].append(reward_test)
            state_array[i_ep_test].append(0)

            if (TTI + 1) % args['Slidingwindow_slot'] == 0 and TTI != 0:
                environment.step_long_time((TTI + 1) // 50)

            if TTI == args['max_steps'] - 1:
                syn[i_ep_test].append(environment.windows.CPA_Synchronicity / 100)
            else:
                syn[i_ep_test].append(0)

        print(f"回合：{i_ep_test + 1}/{test_count}，奖励：{ep_reward_test:.2f}")
        all_reward += ep_reward_test
        if i_ep_test == 0:
            max_reward = ep_reward_test
            max_index = i_ep_test
        if ep_reward_test > max_reward:
            max_reward = ep_reward_test
            max_index = i_ep_test
    print('测试结束...')
    return (
        all_reward / test_count, CPA_rate, CTA_pls, CTA_num,
        syn, qos_CPA, qos_CTA, step_array, option_array,
        state_array, complexity, q_visualizer)


# 测试不执行训练，进行10次测试并保存
if __name__ == '__main__':
    device = init_device(0)
    low_level_agent = None
    args.update({'memory_capacity': 262144})
    args.update({'train_episode': 5000})  # 不再需要训练的episode
    env = NS_Env(args)

    args.update({'algo_name': 'HighOC_delay'})

    high_level_agent = High_Level_Agent(0, device, args)
    high_level_agent.init_training()

    # 加载已训练的模型
    model_path_Q = r'C:\Users\dzl_n\Desktop\ADNNS1\ns\train_data\model\HighOC_delay\Ns\QNet_models\agent_ID_0_QNet_369.pt'  # 替换为您的Q_net模型路径
    model_path_Termination = r'C:\Users\dzl_n\Desktop\ADNNS1\ns\train_data\model\HighOC_delay\Ns\Termination_models\agent_ID_0_Termination_369.pt'  # 替换为您的Termination_net模型路径
    high_level_agent.learner.Q_Net, high_level_agent.learner.Termination_Net = load_model(
        high_level_agent.learner.Q_Net, high_level_agent.learner.Termination_Net, model_path_Q, model_path_Termination
    )
    syn_all = []
    CTA_pls_all = []
    CTA_num_all = []
    # 直接进行10次测试
    for i in range(10):  # 测试10次
        print(f"开始第{i + 1}次测试...")
        reward, CPA_rate, CTA_pls, CTA_num, syn, QOS_CPA, QOS_CTA, step_array, option_array, state_array, complexity, q_visualizer = hwh_test(
            high_level_agent, env, test_count=1)
        syn_all.append(syn[0][4999]*100)
        CTA_pls_all.append(sum(CTA_pls[0]))
        CTA_num_all.append(sum(CTA_num[0]))

        # 保存每次测试的结果
        save_data(step_array=step_array, CPA_rate=CPA_rate, CTA_pls=CTA_pls,
                  CTA_num=CTA_num, syn=syn,
                  QOS_CPA=QOS_CPA, QOS_CTA=QOS_CTA, reward_array=reward,
                  i_episode=i, algo_name=args['algo_name'], state=state_array, option_array=option_array,
                  low_level_algo='Ns_test', complexity=complexity)
    print("任务完成率", sum(syn_all)/1000)
    print("丢包率", sum(CTA_pls_all) / sum(CTA_num_all))
