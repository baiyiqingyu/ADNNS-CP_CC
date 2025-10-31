# Environment/EnvUtils.py
# Q: 为什么要新建这个文件？
# A: 为了引入一些库，定义一些游戏环境，State和Action都要用的值，可能也需要写一些方便的小函数

import numpy as np  # numpy库是非常强大的数学与数据处理库，我们处理矩阵和张量需要用到
# torch是PyTorch库，是很有名的深度学习库，它也能处理矩阵和张量，它的很多操作支持CUDA加速
# CUDA是配套Nvidia显卡的工具，能够最大化利用Nvidia显卡的优势加速大量的计算，深度学习最好使用显卡加速
import torch
# typing是python的类型注释库，它可以帮助你给函数以及变量标注类型，使得你的项目更加清晰和可维护
# 对于被标注了类型的变量，PyCharm可以更好地提供补全建议，因为它可以找到变量对应的类的定义
from typing import *
# cv2是计算机视觉处理库，因为我们的环境格式类似图像，同时我们还要有可视化，因此引入它
# import cv2
import yaml
from collections import deque
from shapely.geometry import LineString
from scipy.interpolate import interp1d
import random
import os
import math
import pandas as pd
import scipy.spatial as spt
from shapely.geometry import box
from shapely.affinity import rotate
from shapely import intersects


def all_seed(seed=0):
    """
    万能的seed函数
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
