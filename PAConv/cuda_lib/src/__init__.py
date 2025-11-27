"""
#원본코드
import os
import torch
from torch.utils.cpp_extension import load

cwd = os.path.dirname(os.path.realpath(__file__))
gpu_path = os.path.join(cwd, 'gpu')

if torch.cuda.is_available():
    gpu = load('gpconv_cuda', [
        os.path.join(gpu_path, 'operator.cpp'),
        os.path.join(gpu_path, 'assign_score_withk_gpu.cu'),
        os.path.join(gpu_path, 'assign_score_withk_halfkernel_gpu.cu'),
    ], build_directory=gpu_path, verbose=False)
"""

import os
import torch
from torch.utils.cpp_extension import load

cwd = os.path.dirname(os.path.realpath(__file__))
gpu_path = os.path.join(cwd, 'gpu')

if torch.cuda.is_available():
    gpu = load('gpconv_cuda', [
        os.path.join(gpu_path, 'operator.cpp'),
        os.path.join(gpu_path, 'assign_score_withk_gpu.cu'),
        os.path.join(gpu_path, 'assign_score_withk_halfkernel_gpu.cu'),
    ], build_directory=gpu_path, verbose=False,
       extra_cuda_cflags=['-gencode=arch=compute_86,code=sm_86'])  # 여기에만 추가
