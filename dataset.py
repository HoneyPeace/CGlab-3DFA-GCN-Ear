import os
import torch
import numpy as np
from torch.utils.data import Dataset

'''
[Module: PyTorch Dataset Loader]
- 역할: util.py에서 전처리(Baking)하여 저장한 NPY 파일들을 읽어와 PyTorch의 모델 학습용 Tensor 묶음(Batch)으로 공급합니다.
- 핵심 변경: 6채널 로직 완전 폐기. in_channels가 7(곡률 포함)인지 3(기본)인지만 직관적으로 판단합니다.
'''

def load_face_data(data_root, data_name, partition, in_channels=3):
    """
    [데이터 로드 및 채널 라우터(Router)]
    - partition: 'train', 'test', 'sample' 중 하나.
    - in_channels: 7 또는 3. 이 값에 따라 불러오는 shape_*.npy 파일의 종류가 달라집니다.
    """
    base_path = os.path.join(data_root, f"{data_name}-npy")
    
    if partition == 'train': suffix = 'train'
    elif partition == 'test': suffix = 'test'
    else: suffix = 'sample' 

    print(f">> Loading NPY data from: {base_path} (Suffix: _{suffix})")
    
    # 🌟 7채널 / 3채널 직관적 이분법 (6채널 찌꺼기 완벽 제거)
    if in_channels == 7:
        # [7채널 모드] util.py에서 XYZ(3) + 방향(3) + 곡률(1)을 병합한 파일
        shape_path = os.path.join(base_path, f'shape_{suffix}.npy') 
        print(f"   [INFO] 🎯 7-Channel Mode: Loading Geometric Features (XYZ+Dir+Curv) from {shape_path}")
    else:
        # [3채널 모드] 순수 XYZ 좌표만 불러옴 (비교군/기본형)
        shape_path = os.path.join(base_path, f'shape_3ch_{suffix}.npy')
        print(f"   [INFO] 🧊 3-Channel Mode: Loading Standard XYZ from {shape_path}")

    heat_path = os.path.join(base_path, f'Heat_data_{suffix}.npy')
    land_path  = os.path.join(base_path, f'landmark_{suffix}.npy')
    name_path  = os.path.join(base_path, f'name_{suffix}.npy')

    if not os.path.exists(heat_path):
        raise FileNotFoundError(f"File not found: {heat_path}\nMake sure 'util.py' generated the NPY files correctly.")

    # RAM 사용량 최적화 및 float32 캐스팅
    Heat_data_sample = np.load(heat_path, allow_pickle=True).astype(np.float32)
    Shape_sample = np.load(shape_path, allow_pickle=True).astype(np.float32)
    landmark_position_select_all = np.load(land_path, allow_pickle=True).astype(np.float32)
    
    # 데이터 이름 로드
    if os.path.exists(name_path):
        Name_sample = np.load(name_path, allow_pickle=True)
    else:
        Name_sample = np.array([str(i) for i in range(len(Shape_sample))])
    
    return Shape_sample, landmark_position_select_all, Heat_data_sample, Name_sample


class FaceLandmarkData(Dataset):
    """
    [PyTorch 표준 Dataset 클래스 상속]
    """
    def __init__(self, data_root, partition='train', data='Ear296_Korean', in_channels=3):
        self.data_root = data_root
        self.partition = partition
        self.DATA = data
        self.in_channels = in_channels
        
        self.data, self.landmark, self.seg, self.names = load_face_data(
            self.data_root, self.DATA, self.partition, self.in_channels
        )

    def __getitem__(self, item):
        # Zero-copy 텐서 변환 (초고속 GPU 업로드 준비)
        face = torch.as_tensor(self.data[item])
        landmark = torch.as_tensor(self.landmark[item])
        heatmap = torch.as_tensor(self.seg[item])  
        
        return face, landmark, heatmap

    def __len__(self):
        return self.data.shape[0]