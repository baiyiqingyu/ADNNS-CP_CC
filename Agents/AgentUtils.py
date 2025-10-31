import os
import copy
from typing import *
import random
import torch
import torch.nn as nn
from collections import deque
import math
import numpy as np


def mkdir(path):
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
    # https://blog.csdn.net/zdc1305/article/details/106138491
