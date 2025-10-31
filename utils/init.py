"""
Original Code is: https://github.com/SIJIEJI/CLNet/blob/08a264a2bc71525cab95d848fbd1d6dc939c7458/utils/init.py#L12
Revised by Meijie, 2023-11-30
"""

import os
import random
import torch


def init_device(seed=None, gpu=None):
    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        print("Running on GPU%d" % (gpu if gpu else 0))
    elif torch.backends.mps.is_available():
        # MACOS system with M1/M2/M3 core
        device = torch.device('mps')
        print("MPS is available because the current MacOS version is 12.3+ "
              "and an MPS-enabled device on this machine.")
        # https://pytorch.org/docs/stable/notes/mps.html
    else:
        device = torch.device('cpu')
        print("Running on CPU")
    """
    elif torch.backends.mps.is_available():
            # MACOS system with M1/M2/M3 core
            device = torch.device('mps')
            print("MPS is available because the current MacOS version is 12.3+ "
                  "and an MPS-enabled device on this machine.")
            # https://pytorch.org/docs/stable/notes/mps.html
    """

    return device
