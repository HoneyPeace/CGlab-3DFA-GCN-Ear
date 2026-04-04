"""
@Author: Yuan Wang (Modified by Researcher 2)
@File: augmentations.py
@Description: 6-Channel (XYZ + Vectors) Aware Normalization & Augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''data normalization'''
def normalize_data(batch_data, landmark=None):
    B, N, C = batch_data.shape
    
    # 🔥 1. 좌표(XYZ)와 방향벡터(V) 분리
    xyz = batch_data[:, :, :3]
    
    # 2. 중심점(Centroid) 이동 및 균일 스케일(Uniform Scale) 정규화
    centroid = torch.mean(xyz, axis=1, keepdim=True)
    xyz = xyz - centroid
    m = torch.max(torch.sqrt(torch.sum(xyz ** 2, axis=2)), axis=1)[0]
    m = m.view(-1, 1, 1)
    xyz = xyz / m
    
    # 🔥 3. 6채널일 경우 벡터 다시 조립
    # 균일 스케일과 이동은 벡터의 '방향'에 영향을 주지 않으므로 V는 원본 그대로 붙입니다!
    if C == 6:
        v = batch_data[:, :, 3:]
        batch_data_out = torch.cat([xyz, v], dim=-1)
    else:
        batch_data_out = xyz
    
    if landmark is not None:
        landmark = (landmark - centroid) / m
        return batch_data_out, landmark
        
    return batch_data_out


'''data Scale and Translate'''
class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc, landmark=None): 
        B, N, C = pc.shape
        
        # 🔥 1. 좌표(XYZ) 추출
        xyz = pc[:, :, :3]
        
        # 랜덤 스케일(Anisotropic)과 이동량 결정
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[B, 1, 3])
        xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[B, 1, 3])
        
        scale = torch.from_numpy(xyz1).float().to(device)
        translate = torch.from_numpy(xyz2).float().to(device)
        
        # 2. 좌표(XYZ) 변환: 스케일 곱하기 + 이동량 더하기
        xyz = torch.mul(xyz, scale) + translate
        
        # 🔥 3. 6채널일 경우 벡터(V) 정밀 보정 
        if C == 6:
            v = pc[:, :, 3:]
            # [규칙 1] 모양이 찌그러지는 비율(scale)만큼 방향 벡터도 곱해서 휘어지게 함 (translate는 더하면 안 됨!)
            v = torch.mul(v, scale)
            # [규칙 2] 벡터의 본질인 '순수 방향'을 유지하기 위해 길이를 다시 1로 깎아줌 (L2 정규화)
            v = F.normalize(v, p=2, dim=-1)
            
            pc_out = torch.cat([xyz, v], dim=-1)
        else:
            pc_out = xyz
        
        if landmark is not None:
            landmark = torch.mul(landmark, scale) + translate
            return pc_out, landmark
            
        return pc_out