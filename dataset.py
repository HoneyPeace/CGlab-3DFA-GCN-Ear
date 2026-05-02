import os
import torch
import numpy as np
from torch.utils.data import Dataset

'''
[Module: PyTorch Dataset Loader]
- 역할: util.py에서 전처리(Baking)하여 저장한 NPY 파일들을 읽어와 모델 학습용 Tensor로 공급합니다.
- 수정 사항: 'sample' 이라는 모호한 파일명 대신, partition('train'/'test')에 맞춰 파일을 정확히 로드합니다.
'''

def load_face_data(data_root, data_name, partition, in_channels=7):
    """
    [데이터 로드 및 채널 라우터]
    - partition: 'train' 또는 'test'. (util.py 저장 파일명과 동기화)
    - in_channels: 7(XYZ+Dir+Curv) 또는 3(XYZ).
    """
    base_path = os.path.join(data_root, f"{data_name}-npy")
    
    # 🌟 [수정 핵심] 'sample'로 퉁치지 않고, 요청받은 partition 이름을 그대로 suffix로 사용합니다.
    # util.py가 'shape_train.npy', 'shape_test.npy'로 저장하기 때문에 이와 일치시킵니다.
    suffix = partition 

    print(f">> [Dataset] Loading NPY data from: {base_path} (Suffix: _{suffix})")
    
    # 7채널(기하 특징 포함) / 3채널(순수 XYZ) 경로 설정
    if in_channels == 7:
        shape_path = os.path.join(base_path, f'shape_{suffix}.npy') 
        print(f"   [INFO] 🎯 7-Channel Mode: Loading Geometric Features (XYZ+Dir+Curv) from {shape_path}")
    else:
        shape_path = os.path.join(base_path, f'shape_3ch_{suffix}.npy')
        print(f"   [INFO] 🧊 3-Channel Mode: Loading Standard XYZ from {shape_path}")

    heat_path = os.path.join(base_path, f'Heat_data_{suffix}.npy')
    land_path  = os.path.join(base_path, f'landmark_{suffix}.npy')
    name_path  = os.path.join(base_path, f'name_{suffix}.npy')

    # 파일 존재 여부 체크 (에러 방지)
    if not os.path.exists(heat_path):
        raise FileNotFoundError(f"File not found: {heat_path}\n"
                                f"현재 요청된 partition: {partition}\n"
                                f"Make sure 'util.py' generated the NPY files correctly.")

    # 데이터 로드 및 float32 변환
    Heat_data = np.load(heat_path, allow_pickle=True).astype(np.float32)
    Shape_data = np.load(shape_path, allow_pickle=True).astype(np.float32)
    landmark_all = np.load(land_path, allow_pickle=True).astype(np.float32)
    
    if os.path.exists(name_path):
        Name_list = np.load(name_path, allow_pickle=True)
    else:
        Name_list = np.array([str(i) for i in range(len(Shape_data))])
    
    return Shape_data, landmark_all, Heat_data, Name_list


class FaceLandmarkData(Dataset):
    """
    [PyTorch 표준 Dataset 클래스]
    """
    def __init__(self, data_root, partition='train', data='Ear296_Korean', in_channels=7):
        self.data_root = data_root
        self.partition = partition
        self.DATA = data
        self.in_channels = in_channels
        
        # 로드 함수 호출
        self.data, self.landmark, self.seg, self.names = load_face_data(
            self.data_root, self.DATA, self.partition, self.in_channels
        )

    def __getitem__(self, item):
        # 텐서 변환
        face = torch.as_tensor(self.data[item])
        landmark = torch.as_tensor(self.landmark[item])
        heatmap = torch.as_tensor(self.seg[item])  
        
        return face, landmark, heatmap

    def __len__(self):
        return self.data.shape[0]