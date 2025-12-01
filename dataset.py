import os
import torch
import numpy as np
from torch.utils.data import Dataset

def load_face_data(data_root, data_name):
    """
    저장된 .npy 파일들을 불러오는 함수
    경로: {data_root}/{data_name}-npy/
    """
    # 저장된 npy 경로 설정
    base_path = os.path.join(data_root, f"{data_name}-npy")
    
    print(f">> Loading NPY data from: {base_path}")
    
    Heat_data_sample = np.load(os.path.join(base_path, 'Heat_data_sample.npy'), allow_pickle=True)
    Shape_sample = np.load(os.path.join(base_path, 'shape_sample.npy'), allow_pickle=True)
    landmark_position_select_all = np.load(os.path.join(base_path, 'landmark_sample.npy'), allow_pickle=True)
    
    return Shape_sample, landmark_position_select_all, Heat_data_sample


class FaceLandmarkData(Dataset):
    def __init__(self, data_root, partition='train', data='Ear296_Korean'):
        # [수정] data_root 인자 추가
        self.data_root = data_root
        self.partition = partition
        self.DATA = data
        
        # 데이터 로드
        self.data, self.landmark, self.seg = load_face_data(self.data_root, self.DATA)

    def __getitem__(self, item):
        # Tensor 변환
        face = torch.from_numpy(self.data[item]).float()
        landmark = torch.from_numpy(self.landmark[item]).float()
        heatmap = torch.from_numpy(self.seg[item]).float()

        # [주의] 학습(Train) 단계에서만 셔플이 필요하면 DataLoader에서 shuffle=True를 씁니다.
        # 여기서는 데이터 자체를 섞지 않고 그대로 반환하는 것이 일반적입니다.
        
        return face, landmark, heatmap

    def __len__(self):
        return self.data.shape[0]