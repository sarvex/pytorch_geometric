import os
import subprocess

import numpy as np
import torch

from torch_geometric.graphgym.config import cfg


def get_gpu_memory_map():
    '''Get the current gpu usage.'''
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    return np.array([int(x) for x in result.strip().split('\n')])


def get_current_gpu_usage():
    '''
    Get the current GPU memory usage.
    '''
    if not cfg.gpu_mem or cfg.device == 'cpu' or not torch.cuda.is_available():
        return -1
    result = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,used_memory',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    current_pid = os.getpid()
    used_memory = 0
    for line in result.strip().split('\n'):
        line = line.split(', ')
        if current_pid == int(line[0]):
            used_memory += int(line[1])
    return used_memory


def auto_select_device():
    r"""Auto select device for the current experiment."""
    if cfg.accelerator == 'auto':
        if torch.cuda.is_available():
            cfg.accelerator = 'cuda'
            cfg.devices = 1
        else:
            cfg.accelerator = 'cpu'
            cfg.devices = None
