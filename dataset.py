import os
import torch
import numpy as np
from torch.utils.data import Dataset

def load_face_data(data_root, data_name, partition):
    """
    저장된 .npy 파일들을 불러오는 함수
    경로: {data_root}/{data_name}-npy/
    파일명: Heat_data_{suffix}.npy 형태로 자동 변경 (suffix: train, test, sample)
    """
    # 저장된 npy 경로 설정
    base_path = os.path.join(data_root, f"{data_name}-npy")
    
    # [수정] Partition에 따라 파일명 접미사(suffix) 결정
    # util.py에서 저장한 규칙과 정확히 일치시켜야 함
    if partition == 'train':
        suffix = 'train'
    elif partition == 'test':
        suffix = 'test'
    else:
        suffix = 'sample' # 기본값 (혹은 기존 데이터 호환용)

    print(f">> Loading NPY data from: {base_path} (Suffix: _{suffix})")
    
    # [수정] 접미사를 붙여서 파일 경로 완성
    heat_path = os.path.join(base_path, f'Heat_data_{suffix}.npy')
    shape_path = os.path.join(base_path, f'shape_{suffix}.npy')
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
    def __init__(self, data_root, partition='train', data='Ear296_Korean'):
        self.data_root = data_root
        self.partition = partition
        self.DATA = data
        
        # [수정] self.partition 정보를 load_face_data에 전달
        self.data, self.landmark, self.seg, self.names = load_face_data(self.data_root, self.DATA, self.partition)

    def __getitem__(self, item):
        face = torch.from_numpy(self.data[item]).float()
        landmark = torch.from_numpy(self.landmark[item]).float()
        heatmap = torch.from_numpy(self.seg[item]).float()

        return face, landmark, heatmap

    def __len__(self):
        return self.data.shape[0]