import os
import torch
import numpy as np
from torch.utils.data import Dataset

'''
[Module: PyTorch Dataset Loader]
- 역할: util.py에서 전처리(Baking)하여 저장한 NPY 파일들을 읽어와 PyTorch의 모델 학습용 Tensor 묶음(Batch)으로 공급합니다.
- NotebookLM 핵심 분석 포인트: in_channels 인자를 통해, 모델이 일반적인 3D 좌표(XYZ)만 요구하는지, 아니면 기하학적 특징(방향/곡률)이 포함된 7채널 데이터를 요구하는지 판단하여 그에 맞는 NPY 파일을 동적으로 로드합니다.
'''

def load_face_data(data_root, data_name, partition, in_channels=3):
    """
    [데이터 로드 및 채널 라우터(Router)]
    - partition: 'train', 'test', 'sample' 중 하나.
    - in_channels: My_args.py에서 설정한 값. 이 값에 따라 불러오는 shape_*.npy 파일의 종류가 달라집니다.
    """
    base_path = os.path.join(data_root, f"{data_name}-npy")
    
    if partition == 'train': suffix = 'train'
    elif partition == 'test': suffix = 'test'
    else: suffix = 'sample' 

    print(f">> Loading NPY data from: {base_path} (Suffix: _{suffix})")
    
    # 🌟 7채널 / 6채널 / 3채널 자동 인식 및 로드 (Data Multiplexing)
    if in_channels == 7:
        # [7채널 모드] util.py에서 XYZ(3) + 방향(3) + 곡률(1)을 병합하여 shape_{suffix}.npy에 덮어썼으므로 이를 불러옴
        shape_path = os.path.join(base_path, f'shape_{suffix}.npy') 
        print(f"   [INFO] 🎯 7-Channel Mode: Loading Geometric Features (XYZ+Dir+Curv) from {shape_path}")
    elif in_channels == 6:
        # [6채널 모드] XYZ(3) + 방향(3) 만 사용할 경우 (설계 유연성)
        shape_path = os.path.join(base_path, f'shape_6ch_{suffix}.npy')
        print(f"   [INFO] 🎯 6-Channel Mode: Loading Geometric Features from {shape_path}")
    else:
        # [3채널 모드] 기본 PointNet/PAConv 파이프라인. 순수 XYZ 좌표만 불러옴.
        shape_path = os.path.join(base_path, f'shape_{suffix}.npy')
        print(f"   [INFO] 🧊 3-Channel Mode: Loading Standard XYZ from {shape_path}")

    heat_path = os.path.join(base_path, f'Heat_data_{suffix}.npy')
    land_path  = os.path.join(base_path, f'landmark_{suffix}.npy')
    name_path  = os.path.join(base_path, f'name_{suffix}.npy')

    # NPY 파일이 없을 경우 친절한 에러 메시지 출력
    if not os.path.exists(heat_path):
        raise FileNotFoundError(f"File not found: {heat_path}\nMake sure 'train.py' generated the NPY files correctly.")

    # 🌟 [디펜스 포인트: 메모리 최적화]
    # RAM 사용량을 반토막 내고 CPU 병목을 없애기 위해 로드 즉시 float32로 캐스팅합니다.
    Heat_data_sample = np.load(heat_path, allow_pickle=True).astype(np.float32)
    Shape_sample = np.load(shape_path, allow_pickle=True).astype(np.float32)
    landmark_position_select_all = np.load(land_path, allow_pickle=True).astype(np.float32)
    
    # 데이터 이름 로드
    if os.path.exists(name_path):
        Name_sample = np.load(name_path, allow_pickle=True)
    else:
        print("   [Warning] name_sample.npy not found. Using indices as names.")
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
        
        # 전체 데이터를 RAM에 한 번에 로드 (위에서 float32로 최적화됨)
        self.data, self.landmark, self.seg, self.names = load_face_data(
            self.data_root, self.DATA, self.partition, self.in_channels
        )

    def __getitem__(self, item):
        # 🌟 [디펜스 포인트: Zero-copy 텐서 변환]
        # 데이터가 이미 float32이므로 torch.as_tensor()를 사용하면 메모리 복사 없이 초고속으로 GPU로 날아갈 준비를 합니다.
        face = torch.as_tensor(self.data[item])
        landmark = torch.as_tensor(self.landmark[item])
        heatmap = torch.as_tensor(self.seg[item])  
        
        return face, landmark, heatmap

    def __len__(self):
        # 전체 데이터 개수 반환
        return self.data.shape[0]