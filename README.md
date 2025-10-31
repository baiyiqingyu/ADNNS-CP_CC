# Overview
This is the Matlab and PyTorch implementation of the paper "Adaptive Non-Orthogonal RAN Slicing for
Collaborative Driving with Bi-Level Optimization:
```
@ARTICLE{10740600,
  author={Mei, Jie and Wang, Xianbin and Zheng, Kan},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Learning Aided Closed-loop Feedback: A Concurrent Dual Channel Information Feedback Mechanism for {Wi-Fi}}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TWC.2024.3481054}}

```
# Requirements
To implement this project, you need to ensure the following requirements are installed.
 * Matlab >= 2021b
 * Python = 3.9, please refere to [Versions of Python Compatible with MATLAB Products by Release](https://www.mathworks.com/support/requirements/python-compatibility.html)
 * Pytorch >= 1.2

# Project Preparation

## Project Tree Arrangement
We recommend you to arrange the project tree as follows.

```
home # The cloned repository of "two-concurrent-CSI-Feedback"
├── NS  
│   ├── Agents
│   │     ├── Agentutils.py
│   │     ├── High_Level_Agent.py
│   │     ├── High_Level_Agent_DQN.py
│   ├── Environment
│   │     ├── EnvUtils.py
│   │     ├── NS_env_V2X.py
│   │     ├── NS_env_V2X_baseline.py
│   │     ├── vehilcles.csv
│   ├── Models
│   │     ├── DQN.py
│   │     ├── High_OC.py
│   │     ├── Modelutils.py
│   ├── utils
│   │     ├── Draw_Q_Est.py
│   │     ├── init.py
│   │     ├── parameters.py
│   │     ├── Replay_Memory.py
│   │     ├── Replay_Memory2.py
│   │     ├── saveData.py
│   │     ├── train_utils.py
│   ├── train_utils.py
│   ├── main_NS_50ms_change_option.py
│   ├── main_NS_load_model.py
│   ├── main_NS_no_PER.py

...
```
# Run simulation
- CSI data generation: The channel state information (CSI) matrix is generated from COST2100 model. You can generate your own dataset according to the open source library of [COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.
- For detailed parameters, please refer to the "Parameter.py" in the folder of "utils".
- Training: Run "main_NS.py" .

# Contact
If you have any problem with this code, please feel free to contact meijie@nbu.eud.cn.
