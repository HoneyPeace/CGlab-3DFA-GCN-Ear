"""
@Author: Yuan Wang (Modified by Researcher 2)
@File: augmentations.py
@Description: 
    - Normalize: Compute stats using XYZ only, preserve Normal.
    - Scale & Translate: Apply to XYZ only, preserve Normal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from My_args import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''data normalization'''
def normalize_data(batch_data):
    """
    [수정됨]
    XYZ(앞 3채널)만 이용하여 중심(Centroid)과 스케일(m)을 계산해 정규화합니다.
    Normal(뒤 3채널)은 방향 벡터이므로 정규화 과정에서 값 변형 없이 그대로 붙여서 내보냅니다.
    """
    B, N, C = batch_data.shape
    
    # 1. XYZ 좌표 분리
    xyz = batch_data[:, :, :3]
    
    # 2. XYZ 기준 Centroid 이동
    centroid = torch.mean(xyz, axis=1)
    xyz = xyz - centroid.unsqueeze(1).repeat(1, N, 1)
    
    # 3. XYZ 기준 Scale Normalization
    m = torch.max(torch.sqrt(torch.sum(xyz ** 2, axis=2)), axis=1)[0]
    xyz = xyz / m.view(-1, 1, 1)
    
    # 4. [중요] 6채널(Normal 포함)인 경우, Normal을 원본 그대로 다시 합침
    if C > 3:
        others = batch_data[:, :, 3:] # Normal (B, N, 3)
        return torch.cat([xyz, others], dim=2) # (B, N, 6)
        
    return xyz


'''data Scale and Translate'''
class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        # pc: (B, N, C)
        bsize = pc.size()[0]
        for i in range(bsize):
            # Uniform Scale (단일 값) -> 공이 찌그러지지 않으므로 Normal 방향 유지됨
            scale = np.random.uniform(low=self.scale_low, high=self.scale_high)
            
            # Translation
            xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            
            # [기존 코드 유지] 0:3 (XYZ)에만 연산 적용
            # Normal(3:6)은 이동/균등스케일에 영향받지 않으므로 건드리지 않음
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], scale) + torch.from_numpy(xyz2).float().to(device)
            
        return pc

'''[NEW] Rotation augmentation (Small angle recommended for registered data)'''
class PointcloudRotation(object):
    def __init__(self, angle_range=np.pi/6): # 기본값: 30도 (이정도면 안전함)
        self.angle_range = angle_range

    def __call__(self, pc):
        # pc: (B, N, C)
        bsize = pc.size()[0]
        
        for i in range(bsize):
            # Y축(Up-axis) 기준 랜덤 회전 (귀 데이터 축에 따라 수정 가능)
            theta = np.random.uniform(low=-self.angle_range, high=self.angle_range)
            
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # 회전 행렬 (Y축 기준)
            rotation_matrix = torch.tensor([
                [cos_theta, 0, sin_theta],
                [0,         1, 0],
                [-sin_theta, 0, cos_theta]
            ]).float().to(device)

            # 1. XYZ 좌표 회전
            pc[i, :, 0:3] = torch.matmul(pc[i, :, 0:3], rotation_matrix)
            
            # 2. Normal 벡터 회전 (방향이므로 같이 돌려야 함!)
            if pc.shape[2] >= 6:
                 pc[i, :, 3:6] = torch.matmul(pc[i, :, 3:6], rotation_matrix)
                
        return pc