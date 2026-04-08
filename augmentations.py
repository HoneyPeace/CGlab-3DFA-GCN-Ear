"""
@Author: Yuan Wang (Modified by Researcher 2 & AI Assistant)
@File: augmentations.py
@Description: 7-Channel (XYZ + Principal Direction + Curvature) Aware Normalization & Augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''data normalization'''
def normalize_data(batch_data, landmark=None):
    B, N, C = batch_data.shape
    
    # 🔥 1. 좌표(XYZ)와 나머지 기하 피처(방향/곡률) 분리
    xyz = batch_data[:, :, :3]
    
    # 2. 중심점(Centroid) 이동 및 균일 스케일(Uniform Scale) 정규화
    centroid = torch.mean(xyz, axis=1, keepdim=True)
    xyz = xyz - centroid
    m = torch.max(torch.sqrt(torch.sum(xyz ** 2, axis=2)), axis=1)[0]
    m = m.view(-1, 1, 1)
    xyz = xyz / m
    
    # 🔥 3. 6채널/7채널일 경우 벡터 및 곡률 다시 조립
    # 균일 스케일과 중심점 이동은 3D 공간상의 '방향'이나 무차원 '곡률 비율'에 영향을 주지 않으므로 원본 그대로 붙입니다.
    if C == 7 or C == 6:
        geom = batch_data[:, :, 3:]
        batch_data_out = torch.cat([xyz, geom], dim=-1)
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
        
        # 🔥 3. 7채널(벡터+곡률) / 6채널(벡터) 정밀 보정 
        if C == 7:
            # 7채널 분리: 주방향(3), 곡률(1)
            v = pc[:, :, 3:6]
            curv = pc[:, :, 6:]
            
            # [방향 처리] 모양이 찌그러지는 비율만큼 방향 벡터도 곱해서 휘어지게 한 뒤 L2 정규화
            v = torch.mul(v, scale)
            v = F.normalize(v, p=2, dim=-1)
            
            # [곡률 처리] 무차원 스칼라 값이므로 어떠한 왜곡도 가하지 않고 원본 그대로 병합
            pc_out = torch.cat([xyz, v, curv], dim=-1)
            
        elif C == 6:
            v = pc[:, :, 3:]
            v = torch.mul(v, scale)
            v = F.normalize(v, p=2, dim=-1)
            pc_out = torch.cat([xyz, v], dim=-1)
            
        else:
            pc_out = xyz
        
        if landmark is not None:
            landmark = torch.mul(landmark, scale) + translate
            return pc_out, landmark
            
        return pc_out