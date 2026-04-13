"""
@Author: Yuan Wang (Modified by Researcher 2 & AI Assistant)
@File: augmentations.py
@Description: 
[NotebookLM을 위한 모듈 요약]
7채널(XYZ + Principal Direction + Curvature) 3D Point Cloud 데이터의 정규화 및 데이터 증강(Augmentation) 모듈.
핵심은 점의 위치(XYZ)를 물리적으로 왜곡시킬 때, 모델이 헷갈리지 않도록 
표면의 방향(Vector)과 굴곡(Curvature)도 기하학적 수학 규칙에 맞게 올바르게 연동시키는 것입니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''data normalization'''
def normalize_data(batch_data, landmark=None):
    """
    [전역 정규화 (Global Normalization)]
    - 목적: 모델이 귀/얼굴의 크기나 절대적인 위치에 영향을 받지 않도록, 중심을 (0,0,0)으로 맞추고 반지름 1인 구(Sphere) 안에 넣습니다.
    - 기하학적 특징 유지: 이 함수는 '균일 스케일링(Uniform Scale)'과 '평행 이동(Translation)'만 수행합니다.
      공간 전체를 균일하게 줄이거나 위치만 옮길 때는 표면이 바라보는 '방향'이나 '곡률 비율' 자체가 변하지 않으므로 부가 채널(3~6)은 계산 없이 그대로 유지합니다.
    """
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
        # 정답 랜드마크도 입력 데이터와 동일한 잣대로 정규화
        landmark = (landmark - centroid) / m
        return batch_data_out, landmark
        
    return batch_data_out


'''data Scale and Translate'''
class PointcloudScaleAndTranslate(object):
    """
    [비등방성 크기 변환 및 이동 증강 (Anisotropic Scale & Translate)]
    - 목적: 데이터 다양성을 위해 X, Y, Z 축을 각각 다르게 찌그러뜨리고(비등방성 스케일), 위치를 무작위로 옮깁니다.
    - 🌟 핵심 기하학적 보정 (NotebookLM 분석 포인트): 
      물체의 X축이 2배 늘어나면, 해당 표면의 주름 방향(Vector)도 X축으로 더 기울어져야 합니다.
      따라서 XYZ 좌표만 곱하는 것이 아니라, 방향 벡터(v)도 똑같이 스케일링한 뒤, 
      다시 길이가 1인 단위 벡터가 되도록 L2 정규화(F.normalize)를 거치는 고차원적 보정을 수행합니다.
    """
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc, landmark=None): 
        B, N, C = pc.shape
        
        # 🔥 1. 좌표(XYZ) 추출
        xyz = pc[:, :, :3]
        
        # 랜덤 스케일(Anisotropic)과 이동량 결정 (축마다 다른 값이 생성됨)
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
            
            # [방향 처리] 모양이 찌그러지는 비율(scale)만큼 방향 벡터도 곱해서 올바르게 휘어지게 한 뒤 L2 정규화
            v = torch.mul(v, scale)
            v = F.normalize(v, p=2, dim=-1)
            
            # [곡률 처리] 곡률은 무차원 스칼라 값이므로 어떠한 왜곡도 가하지 않고 원본 그대로 병합
            pc_out = torch.cat([xyz, v, curv], dim=-1)
            
        elif C == 6:
            v = pc[:, :, 3:]
            v = torch.mul(v, scale)
            v = F.normalize(v, p=2, dim=-1)
            pc_out = torch.cat([xyz, v], dim=-1)
            
        else:
            pc_out = xyz
        
        if landmark is not None:
            # 랜드마크 좌표도 모델과 동일하게 변환
            landmark = torch.mul(landmark, scale) + translate
            return pc_out, landmark
            
        return pc_out