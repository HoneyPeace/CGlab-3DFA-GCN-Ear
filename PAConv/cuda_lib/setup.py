"""
원래코드
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='assign_score_withk',
    ext_modules=[
        CUDAExtension(
            name='assign_score_withk',
            sources=[
                'operator.cpp',
                'assign_score_withk_gpu.cu',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='assign_score_withk',
    ext_modules=[
        CUDAExtension(
            name='assign_score_withk',
            sources=[
                'src/gpu/operator.cpp',
                'src/gpu/assign_score_withk_gpu.cu',
                'src/gpu/assign_score_withk_halfkernel_gpu.cu',  # ✅ 추가!
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)