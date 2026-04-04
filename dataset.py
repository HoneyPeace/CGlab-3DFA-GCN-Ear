import os
import torch
import numpy as np
from torch.utils.data import Dataset

def load_face_data(data_root, data_name, partition, in_channels=3):
    """
    저장된 .npy 파일들을 불러오는 함수
    경로: {data_root}/{data_name}-npy/
    파일명: Heat_data_{suffix}.npy 형태로 자동 변경 (suffix: train, test, sample)
    """
    # 저장된 npy 경로 설정
    base_path = os.path.join(data_root, f"{data_name}-npy")
    
    # Partition에 따라 파일명 접미사(suffix) 결정
    if partition == 'train':
        suffix = 'train'
    elif partition == 'test':
        suffix = 'test'
    else:
        suffix = 'sample' # 기본값

    print(f">> Loading NPY data from: {base_path} (Suffix: _{suffix})")
    
    # 🌟 [신규 추가] 채널 수에 따라 읽어올 데이터 파일명 자동 스위칭!
    if in_channels == 6:
        shape_path = os.path.join(base_path, f'shape_6ch_{suffix}.npy')
        print(f"   [INFO] 🎯 6-Channel Mode: Loading Geometric Features from {shape_path}")
    else:
        shape_path = os.path.join(base_path, f'shape_{suffix}.npy')
        print(f"   [INFO] 🧊 3-Channel Mode: Loading Standard XYZ from {shape_path}")

    heat_path = os.path.join(base_path, f'Heat_data_{suffix}.npy')
    land_path  = os.path.join(base_path, f'landmark_{suffix}.npy')
    name_path  = os.path.join(base_path, f'name_{suffix}.npy')

    # 파일 존재 여부 확인 (디버깅용)
    if not os.path.exists(heat_path):
        raise FileNotFoundError(f"File not found: {heat_path}\nMake sure 'train.py' generated the NPY files correctly.")

    Heat_data_sample = np.load(heat_path, allow_pickle=True)
    Shape_sample = np.load(shape_path, allow_pickle=True)
    landmark_position_select_all = np.load(land_path, allow_pickle=True)
    
    # 이름 데이터 로드
    if os.path.exists(name_path):
        Name_sample = np.load(name_path, allow_pickle=True)
    else:
        print("   [Warning] name_sample.npy not found. Using indices as names.")
        Name_sample = np.array([str(i) for i in range(len(Shape_sample))])
    
    return Shape_sample, landmark_position_select_all, Heat_data_sample, Name_sample


class FaceLandmarkData(Dataset):
    # 🌟 [수정] in_channels 인자를 받을 수 있도록 파라미터 추가
    def __init__(self, data_root, partition='train', data='Ear296_Korean', in_channels=3):
        self.data_root = data_root
        self.partition = partition
        self.DATA = data
        self.in_channels = in_channels
        
        # 🌟 self.in_channels 정보를 load_face_data에 전달
        self.data, self.landmark, self.seg, self.names = load_face_data(self.data_root, self.DATA, self.partition, self.in_channels)

    def __getitem__(self, item):
        face = torch.from_numpy(self.data[item]).float()
        landmark = torch.from_numpy(self.landmark[item]).float()
        heatmap = torch.from_numpy(self.seg[item]).float()

        return face, landmark, heatmap

    def __len__(self):
        return self.data.shape[0]