"""
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: augmentations.py
@Time: 2021/12/02 10:03 AM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''data normalization'''
def normalize_data(batch_data, landmark=None):
    """
    포인트 클라우드와 랜드마크를 동일한 좌표계로 정규화합니다.
    batch_data : batch_size * num_points * 3
    landmark   : batch_size * num_landmarks * 3 (Optional)
    """
    B, N, C = batch_data.shape
    
    # 1. 중심점(Centroid) 계산 및 이동
    # 모든 점의 평균 위치를 (0,0,0)으로 맞춥니다.
    centroid = torch.mean(batch_data, axis=1, keepdim=True) # (B, 1, 3)
    batch_data = batch_data - centroid
    
    # 2. 스케일(Scale) 정규화
    # 가장 먼 점까지의 거리를 1로 맞추어 [-1, 1] 범위로 압축합니다.
    m = torch.max(torch.sqrt(torch.sum(batch_data ** 2, axis=2)), axis=1)[0]
    m = m.view(-1, 1, 1) # (B, 1, 1)
    batch_data = batch_data / m
    
    # [중요] 랜드마크가 있다면 포인트 클라우드와 똑같은 중심점과 배율을 적용합니다.
    if landmark is not None:
        landmark = (landmark - centroid) / m
        return batch_data, landmark
        
    return batch_data


'''data Scale and Translate'''
class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc, landmark=None): 
        B, N, C = pc.shape
        
        # 랜덤하게 스케일과 이동량 결정
        xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[B, 1, 3])
        xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[B, 1, 3])
        
        scale = torch.from_numpy(xyz1).float().to(device)
        translate = torch.from_numpy(xyz2).float().to(device)
        
        # 포인트 클라우드 변환
        pc = torch.mul(pc, scale) + translate
        
        # [중요] 랜드마크가 들어왔다면 포인트 클라우드와 완벽히 동기화하여 변환합니다.
        if landmark is not None:
            landmark = torch.mul(landmark, scale) + translate
            return pc, landmark
            
        return pc