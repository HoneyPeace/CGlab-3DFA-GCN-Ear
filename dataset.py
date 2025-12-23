'''
@Author: Researcher 5 (Fixed)
@File: dataset.py
@Description: Loads NPY files from '{partition}_npy' folder. Matches train.py arguments.
'''

import os
import torch
import numpy as np
from torch.utils.data import Dataset

def load_face_data(data_root, data_name):
    """
    저장된 .npy 파일들을 불러오는 함수
    경로: {data_root}/{data_name}_npy/  (언더바 _ 사용)
    """
    # [핵심 수정] util.py와 맞춤: -npy -> _npy
    base_path = os.path.join(data_root, f"{data_name}_npy")
    
    if not os.path.exists(base_path):
        # 혹시 몰라 하이픈도 체크 (호환성)
        alt_path = os.path.join(data_root, f"{data_name}-npy")
        if os.path.exists(alt_path):
            base_path = alt_path
        else:
            print(f"[Error] NPY folder not found: {base_path}")
            # 빈 배열 반환하여 충돌 방지 (디버깅용)
            return np.array([]), np.array([]), np.array([])
    
    print(f">> Loading NPY data from: {base_path}")
    
    # util.py에서 저장한 파일명 그대로 로드
    # 파일명: Heat_data_{data_name}.npy 등
    try:
        Heat_data_sample = np.load(os.path.join(base_path, f'Heat_data_{data_name}.npy'), allow_pickle=True)
        Shape_sample = np.load(os.path.join(base_path, f'shape_{data_name}.npy'), allow_pickle=True)
        landmark_position_select_all = np.load(os.path.join(base_path, f'landmark_{data_name}.npy'), allow_pickle=True)
        
        # 이름 데이터도 로드 (있으면)
        names_path = os.path.join(base_path, f'names_{data_name}.npy')
        if os.path.exists(names_path):
            names_sample = np.load(names_path, allow_pickle=True)
        else:
            names_sample = np.array(["unknown"] * len(Shape_sample))
            
    except FileNotFoundError as e:
        print(f"[Error] Missing NPY file in {base_path}: {e}")
        return np.array([]), np.array([]), np.array([])
    
    return Shape_sample, landmark_position_select_all, Heat_data_sample, names_sample


class FaceLandmarkData(Dataset):
    # [핵심 수정] 'data' 인자 삭제 -> partition 이름 그대로 사용
    def __init__(self, data_root, partition='train'):
        self.data_root = data_root
        self.partition = partition
        
        # 해당 폴더(train_npy 또는 test_npy) 로드
        self.points, self.landmarks, self.heatmaps, self.names = load_face_data(self.data_root, self.partition)
        
        # 로드 실패 시 에러 처리
        if len(self.points) == 0:
            # raise RuntimeError(f"Failed to load data for partition: {partition}")
            pass # train.py에서 처리하도록 패스

    def __getitem__(self, item):
        # Tensor 변환
        face = torch.from_numpy(self.points[item]).float()
        landmark = torch.from_numpy(self.landmarks[item]).float()
        heatmap = torch.from_numpy(self.heatmaps[item]).float()
        name = str(self.names[item])

        return face, landmark, heatmap, name

    def __len__(self):
        return self.points.shape[0]
    
    # 편의 속성 (train.py의 process_data_storage 함수 호환용)
    @property
    def npy_dir(self):
        return os.path.join(self.data_root, f"{self.partition}_npy")