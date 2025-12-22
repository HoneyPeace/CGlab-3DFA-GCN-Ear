'''
@Author: Yuan Wang (Modified by Researcher 5)
@File: dataset.py
@Description: Loads NPY files including 'names' for tracking original filenames.
'''

import os
import torch
import numpy as np
from torch.utils.data import Dataset

class FaceLandmarkData(Dataset):
    def __init__(self, data_root, partition='train'):
        """
        data_root: ../../Data (Args에서 받아옴)
        partition: 'train' or 'test' (폴더 구분용)
        """
        self.partition = partition
        
        # train.py가 생성해둔 _npy 폴더 경로 (예: ../../Data/train_npy)
        self.npy_dir = os.path.join(data_root, f"{partition}_npy")
        
        try:
            print(f"[{partition.upper()}] Loading data from {self.npy_dir}...")
            
            # 1. Shape (Point Cloud)
            self.points = np.load(os.path.join(self.npy_dir, f"shape_{partition}.npy"), allow_pickle=True)
            # 2. Landmark (Ground Truth)
            self.landmarks = np.load(os.path.join(self.npy_dir, f"landmark_{partition}.npy"), allow_pickle=True)
            # 3. Heatmap (Gaussian)
            self.heatmaps = np.load(os.path.join(self.npy_dir, f"Heat_data_{partition}.npy"), allow_pickle=True)
            # 4. Names (파일명 추적용) - [추가됨]
            self.names = np.load(os.path.join(self.npy_dir, f"names_{partition}.npy"), allow_pickle=True)
            
            print(f"[{partition.upper()}] Successfully loaded {len(self.points)} samples.")
            
        except FileNotFoundError:
            # 아직 npy를 굽지 않았을 때 (최초 실행 시) 에러 방지용 빈 배열 처리
            print(f"\n[Warning] NPY files not found in {self.npy_dir}")
            print(f"Please run 'python train.py --need_resample True' to generate them.\n")
            self.points, self.landmarks, self.heatmaps, self.names = [], [], [], []

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        # 텐서 변환
        point = torch.from_numpy(self.points[index]).float()
        landmark = torch.from_numpy(self.landmarks[index]).float()
        heatmap = torch.from_numpy(self.heatmaps[index]).float()
        
        # [중요] 파일명(문자열) 반환
        name = str(self.names[index])
        
        # 4개 리턴: (Point, Landmark, Heatmap, Name)
        return point, landmark, heatmap, name