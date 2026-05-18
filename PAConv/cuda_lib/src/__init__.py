import importlib
import os
import sys

import torch
from torch.utils.cpp_extension import load

cwd = os.path.dirname(os.path.realpath(__file__))
gpu_path = os.path.join(cwd, 'gpu')

if torch.cuda.is_available():
    try:
        sys.path.insert(0, gpu_path)
        gpu = importlib.import_module('gpconv_cuda')
    except ImportError:
        gpu = load('gpconv_cuda', [
            os.path.join(gpu_path, 'operator.cpp'),
            os.path.join(gpu_path, 'assign_score_withk_gpu.cu'),
            os.path.join(gpu_path, 'assign_score_withk_halfkernel_gpu.cu'),
        ], build_directory=gpu_path, verbose=False,
           extra_cuda_cflags=['-gencode=arch=compute_86,code=sm_86'])
