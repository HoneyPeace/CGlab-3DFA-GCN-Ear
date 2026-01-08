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
    
    # [수정 포인트 1] 이름 데이터(name_sample.npy) 로드 추가
    name_path = os.path.join(base_path, 'name_sample.npy')
    
    if os.path.exists(name_path):
        Name_sample = np.load(name_path, allow_pickle=True)
    else:
        # 혹시 이름 파일이 아직 안 만들어졌을 경우를 대비한 안전 장치 (인덱스를 이름으로 사용)
        print("   [Warning] name_sample.npy not found. Using indices as names.")
        Name_sample = np.array([str(i) for i in range(len(Shape_sample))])
    
    return Shape_sample, landmark_position_select_all, Heat_data_sample, Name_sample


class FaceLandmarkData(Dataset):
    def __init__(self, data_root, partition='train', data='Ear296_Korean'):
        self.data_root = data_root
        self.partition = partition
        self.DATA = data
        
        # [수정 포인트 2] self.names에 이름 리스트 저장
        # load_face_data가 4개를 반환하도록 바뀌었음
        self.data, self.landmark, self.seg, self.names = load_face_data(self.data_root, self.DATA)

    def __getitem__(self, item):
        # Tensor 변환
        face = torch.from_numpy(self.data[item]).float()
        landmark = torch.from_numpy(self.landmark[item]).float()
        heatmap = torch.from_numpy(self.seg[item]).float()

        # [주의] __getitem__에서는 이름을 반환하지 않습니다.
        # 이유는 train.py의 학습 루프(for a,b,c in loader) 구조를 깨지 않기 위함입니다.
        # 이름은 train.py의 저장 함수(process_data_storage)에서 self.names[item]으로 직접 접근해서 씁니다.
        return face, landmark, heatmap

    def __len__(self):
        return self.data.shape[0]