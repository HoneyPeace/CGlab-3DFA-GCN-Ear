# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: dataset.py
# @Description:
# [PyTorch Dataset Loader]
# - 역할: util.py에서 전처리(Baking)하여 저장한 NPY 파일들을 읽어와 모델 학습용 Tensor로 공급
# - 🌟 3채널(XYZ), 6채널(XYZ+Dir), 7채널(XYZ+Dir+Curv) 완벽 라우팅 적용
# ==============================================================================

import os
import torch
import numpy as np
from torch.utils.data import Dataset

def load_face_data(data_root, data_name, partition, in_channels=7):
    """
    [데이터 로드 및 채널 라우터]
    - partition: 'train' 또는 'test'. (util.py 저장 파일명과 동기화)
    - in_channels: 3, 6, 7 중 하나를 받아 알맞은 NPY 파일을 로드합니다.
    """
    base_path = os.path.join(data_root, f"{data_name}-npy")
    
    # 🌟 'sample' 대신 partition 이름을 그대로 suffix로 사용합니다.
    suffix = partition 

    print(f">> [Dataset] Loading NPY data from: {base_path} (Suffix: _{suffix})")
    
    # 🌟 [핵심 수정] 3, 6, 7 채널 분기 완벽 대응 및 예외 처리
    if in_channels == 7:
        shape_path = os.path.join(base_path, f'shape_{suffix}.npy') 
        print(f"   [INFO] 🎯 7-Channel Mode: Loading Geometric Features (XYZ+Dir+Curv) from {shape_path}")
    elif in_channels == 6:
        shape_path = os.path.join(base_path, f'shape_6ch_{suffix}.npy')
        print(f"   [INFO] 📐 6-Channel Mode: Loading Geometric Features (XYZ+Dir) from {shape_path}")
    elif in_channels == 3:
        shape_path = os.path.join(base_path, f'shape_3ch_{suffix}.npy')
        print(f"   [INFO] 🧊 3-Channel Mode: Loading Standard XYZ from {shape_path}")
    else:
        raise ValueError(f"❌ [Error] 지원하지 않는 in_channels 값입니다: {in_channels}. (허용값: 3, 6, 7)")

    heat_path = os.path.join(base_path, f'Heat_data_{suffix}.npy')
    land_path  = os.path.join(base_path, f'landmark_{suffix}.npy')
    name_path  = os.path.join(base_path, f'name_{suffix}.npy')

    # 파일 존재 여부 체크 (에러 방지)
    if not os.path.exists(heat_path) or not os.path.exists(shape_path):
        raise FileNotFoundError(f"❌ 파일 로드 실패!\n"
                                f"경로: {base_path}\n"
                                f"현재 요청된 partition: {partition}, 채널: {in_channels}ch\n"
                                f"먼저 'util.py'를 실행하여 NPY 파일들을 구워주세요(Baking).")

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
        # 텐서 변환 (메모리 카피 최소화를 위해 as_tensor 사용)
        face = torch.as_tensor(self.data[item])
        landmark = torch.as_tensor(self.landmark[item])
        heatmap = torch.as_tensor(self.seg[item])  
        
        return face, landmark, heatmap

    def __len__(self):
        return self.data.shape[0]