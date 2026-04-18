"""
@Author: Yuan Wang (Modified by Researcher)
@File: augmentations.py
@Description: 
7채널(XYZ + Principal Direction + Curvature) 3D Point Cloud 데이터 증강 모듈.
절대 좌표 회귀(Coordinate Regression)의 정확도 하락을 방지하기 위해 회전(Rotation) 증강은 배제하고,
모델의 일반화 성능을 높이기 위한 스케일/이동(Scale&Translate)과 미세 노이즈(Jitter)만 엄격히 적용합니다.
"""

import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''1. data normalization (전역 정규화)'''
def normalize_data(batch_data, landmark=None):
    B, N, C = batch_data.shape
    xyz = batch_data[:, :, :3]
    
    centroid = torch.mean(xyz, axis=1, keepdim=True)
    xyz = xyz - centroid
    m = torch.max(torch.sqrt(torch.sum(xyz ** 2, axis=2)), axis=1)[0]
    m = m.view(-1, 1, 1)
    xyz = xyz / m
    
    if C == 7 or C == 6:
        geom = batch_data[:, :, 3:]
        batch_data_out = torch.cat([xyz, geom], dim=-1)
    else:
        batch_data_out = xyz
    
    if landmark is not None:
        landmark = (landmark - centroid) / m
        return batch_data_out, landmark
        
    return batch_data_out

'''2. data Scale and Translate (비등방성 크기 변환 및 이동)'''
class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc, landmark=None): 
        B, N, C = pc.shape
        xyz = pc[:, :, :3]
        
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[B, 1, 3])
        xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[B, 1, 3])
        
        scale = torch.from_numpy(xyz1).float().to(device)
        translate = torch.from_numpy(xyz2).float().to(device)
        
        xyz = torch.mul(xyz, scale) + translate
        
        if C == 7:
            v = pc[:, :, 3:6]
            curv = pc[:, :, 6:]
            v = torch.mul(v, scale)
            v = F.normalize(v, p=2, dim=-1)
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

'''3. 🌟 NEW: data Jitter (미세 센서 노이즈 시뮬레이션)'''
class PointcloudJitter(object):
    """
    [가우시안 지터링 (안전한 노이즈)]
    - 스캐너의 기계적 오차를 시뮬레이션하여 딥러닝 모델의 과적합(Overfitting)을 방지합니다.
    - std=0.001 수준의 극미세 노이즈이므로 랜드마크의 절대 위치를 훼손하지 않습니다.
    """
    def __init__(self, std=0.001, clip=0.005):
        self.std = std
        self.clip = clip

    def __call__(self, pc, landmark=None):
        B, N, C = pc.shape
        xyz = pc[:, :, :3]
        
        # 3D 좌표에만 가우시안 노이즈(미세 떨림) 적용
        noise = torch.randn(B, N, 3, device=device) * self.std
        noise = torch.clamp(noise, -self.clip, self.clip)
        xyz_jittered = xyz + noise
        
        # 🌟 기하학적 보존: 노이즈는 점의 위치만 미세하게 흔들 뿐, 표면의 주방향과 곡률은 훼손하지 않음
        if C >= 6:
            pc_out = torch.cat([xyz_jittered, pc[:, :, 3:]], dim=-1)
        else:
            pc_out = xyz_jittered
            
        # [주의]: 랜드마크(정답지)는 흔들면 안 됩니다! Ground Truth는 고정되어야 함.
        if landmark is not None:
            return pc_out, landmark
        return pc_out