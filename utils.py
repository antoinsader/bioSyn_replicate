
import logging
import time
import numpy as np
import torch
import random
import os
import torch.nn.functional as F
import pickle
from datetime import datetime

def init_logging(LOGGER, base_output_dir, logging_folder, minimize):


    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_folder_name = 'logs_min' if minimize else 'logs'
    log_dir = os.path.join(base_output_dir, logs_folder_name, logging_folder)
    
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{date_str}.log")
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    LOGGER.addHandler(file_handler)

    LOGGER.info(f"Logging file: {log_file_path}")
    return log_file_path

def init_seed(LOGGER, seed=None):
    if seed == None:
        seed = int(round(time.time() * 1000)) % 10000
    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def marginal_nll(score, target):
    """
    sum all scores among positive samples
    """
    predict = F.softmax(score, dim=-1)
    loss = predict * target
    loss = loss.sum(dim=-1)                   # sum all positive scores
    loss = loss[loss > 0]                     # filter sets with at least one positives
    loss = torch.clamp(loss, min=1e-9, max=1) # for numerical stability
    loss = -torch.log(loss)                   # for negative log likelihood
    if len(loss) == 0:
        loss = loss.sum()                     # will return zero loss
    else:
        loss = loss.mean()
    return loss



def save_pkl(ar, fp):
    with open(fp, 'wb') as f:
        pickle.dump(ar, f)
def get_pkl(fp):
    with open(fp, "rb") as f:
        return pickle.load(f)




def get_gpu_tier():
    """
        return (cpu_tier, has_fp16)
        cpu_tier is one of ["cpu", "gpu_s", "gpu_m", "gpu_l", "gpu_xl"]
    """
    if not torch.cuda.is_available():
        return  "cpu", False

    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)

    major, _ = torch.cuda.get_device_capability(0)
    has_fp16= major >= 7

    if vram_gb < 6:
        return "gpu_s", has_fp16
    elif vram_gb < 16:
        return "gpu_m", has_fp16
    elif vram_gb < 32:
        return "gpu_l" , has_fp16
    else:
        return "gpu_xl", has_fp16


def embed_dense_batch_size(len_names, gpu_tier):
    if gpu_tier == "cpu":
        base = 256
    elif gpu_tier == "gpu_s":
        base= 512
    elif gpu_tier == "gpu_m":
        base = 2048
    elif gpu_tier == "gpu_l":
        base = 4096
    else:
        base = 8192

    base = int(base * 1.3)
    base = max(1, min(base, int(len_names) ))
    return base

