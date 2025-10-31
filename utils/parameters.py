import argparse

parser = argparse.ArgumentParser(description='网络切片示例程序说明')

parser.add_argument('--low_policy', type=str, default='CP', help='low level policy')

parser.add_argument('--gamma', default=0.95, type=float, help="discounted factor")
parser.add_argument('--low_level_lr', default=3e-5, type=float, help="learning rate of critic")
parser.add_argument('--high_level_lr', default=2e-5, type=float, help="learning rate of high level")
parser.add_argument('--target_update', default=2, type=int)
parser.add_argument('--tau', default=1e-3, type=float)
parser.add_argument('--critic_hidden_dim', default=256, type=int)
parser.add_argument('--actor_hidden_dim', default=256, type=int)

parser.add_argument('--state_dim', default=880, type=int, help="state dimension")
parser.add_argument('--state_dim_ideal', default=880, type=int)

parser.add_argument('--option_dim', default=2, type=int, help="option dimension")
parser.add_argument('--memory_capacity', default=131072, type=int, help="memory capacity")
parser.add_argument('--batch_size', default=128, type=int)

parser.add_argument('--seed', default=1, type=int, help="random seed")
parser.add_argument('--train_episode', default=15000, type=int, help="train episode")
parser.add_argument('--test_episode', default=2, type=int, help="test episode")
parser.add_argument('--max_steps', default=5000, type=float, help="max steps in each episode")  # 5s
parser.add_argument('--algo_name', default='DDPG', type=str, help="name of algorithm")
parser.add_argument('--rode_file_path', default='./trace_data/straight/sumoTrace_low.csv', type=str,
                    help="path of road file")

parser.add_argument('--num_of_mslots', default=7, type=int)
parser.add_argument('--Slidingwindow_slot', default=50, type=int)
parser.add_argument('--qos_CPA_s', default=50, type=int)
parser.add_argument('--windows_length', default=1, type=int)  # 50ms
parser.add_argument('--num_of_CTA', default=4, type=int)  # 50ms
parser.add_argument('--burst_probability', default=0.00005, type=float)  # 50ms

args = parser.parse_args([])
args = {**vars(args)}  # 将args转换为字典
