import torch
import logging
import os
import psutil
import subprocess
import gc
import json
import csv
from pathlib import Path
from os.path import exists
import os
import random 
import numpy as np
#---------------------------------------------------------------------------
def check_if_exists(path, isDir=False, createIfNotExists=False): 
    if isDir and not path.endswith("/"):
        path += "/"
    pathExists = exists(path)
    if not pathExists:
        if createIfNotExists:
            os.makedirs(path) 
        else:
            raise ValueError(f"{path} is an invalid path!")
    if not isDir:
        filePath = Path(path)
        if not filePath.is_file():
            raise ValueError(f"{path} is not a file!")   
    return path
#---------------------------------------------------------------------------
def check_file(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"[check_file] {fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise RuntimeError(f"[check_file] {fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise RuntimeError(f"[check_file] {fileName} is not a file!")
#---------------------------------------------------------------------------
def read_file(fileName):
    data=None
    if fileName.endswith(".csv"):
        with open(fileName, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    elif fileName.endswith(".json"):
        with open(fileName, "r") as f:
            data = json.load(f)
    else: 
        raise RuntimeError("Unsupported file: {}".format(fileName))
    return data
#---------------------------------------------------------- 
def log_memory_usage(stage=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"{stage} - CPU Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB RSS")
    
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'])
    gpu_memory = [tuple(map(int, line.split(','))) for line in result.decode('utf-8').strip().split('\n')]
    for i, (used, total) in enumerate(gpu_memory):
        logging.info(f"GPU {i}: {used}/{total} MB")
#---------------------------------------------------------- 
def clear_memory(*vars_to_delete):
    for var in vars_to_delete:
        del var
    torch.cuda.empty_cache()
    gc.collect()
#---------------------------------------------------------- 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#---------------------------------------------------------- 