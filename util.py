'''
@Author: Yuan Wang (Modified by Researcher 2 & AI Assistant)
@File: util.py
@Description: 
[NotebookLM을 위한 모듈 요약]
이 파일은 3D 랜드마크 검출 모델의 '오프라인 데이터 전처리 파이프라인'을 담당합니다.
핵심 역할은 3가지입니다:
1) 원본 3D Point Cloud(ply/obj)와 정답 랜드마크(asc/mat) 매칭 및 로드
2) 공간적 균일성을 유지하는 FPS(Farthest Point Sampling) 다운샘플링 및 가우시안 히트맵 생성
3) 🌟 7-Channel Geometric Feature (Eigenvector + Eigenvalue) 사전 연산(Baking) 및 NPY 저장
'''

import os 
import glob
import numpy as np
import scipy.io as sio
import torch
import h5py

from plyfile import PlyData
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from functools import reduce

# ==========================================================
# 🚀 데이터 전처리용 초고속 C++ FPS 커널 로드
# ==========================================================
# [의도] FPS(가장 먼 점 샘플링)는 점의 개수 N에 대해 O(N^2)의 연산량을 가집니다.
# 파이썬(PyTorch) 네이티브 구현으로는 대용량 Point Cloud 처리 시 병목이 발생하므로,
# C++로 컴파일된 PointNet++의 CUDA 커널을 직접 호출하여 연산 속도를 극대화합니다.
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "utils" / "pointnet2_ops_lib"))

try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cpp
    USE_CPP_FPS_UTIL = True
    print(">>> [SUCCESS] 🚀 util.py에서 C++ 초고속 FPS 커널을 성공적으로 로드했습니다!")
except ImportError:
    USE_CPP_FPS_UTIL = False
    print(">>> [WARNING] ⚠️ util.py에서 C++ FPS 커널을 찾지 못했습니다. 기존 방식으로 진행합니다.")
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# [Helper] 파일 읽기 함수들 (Name 반환 유지)
# -----------------------------------------------------------------------------
def read_ply_files_from_folder(folder_path):
    files = sorted(glob.glob(os.path.join(folder_path, "*.ply")))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(folder_path, "*.obj"))) 
    
    if len(files) == 0:
        return [], []
    
    print(f"   Found {len(files)} shapes in [{os.path.basename(folder_path)}]. Reading...")
    shape_list = []
    name_list = []

    for f in tqdm(files, desc="Loading Shapes", unit="file"):
        try:
            basename = os.path.splitext(os.path.basename(f))[0]
            name_list.append(basename)

            plydata = PlyData.read(f)
            vertex = plydata['vertex']
            points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
            shape_list.append(points)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return shape_list, name_list

def read_asc_files_from_folder(folder_path):
    files = sorted(glob.glob(os.path.join(folder_path, "*.asc")))
    if len(files) == 0:
        return []
    
    print(f"   Found {len(files)} ASC files in [{os.path.basename(folder_path)}]. Reading...")
    lm_list = []
    for f in tqdm(files, desc="Loading Landmarks", unit="file"):
        try:
            try: points = np.loadtxt(f, delimiter=',')
            except ValueError: points = np.loadtxt(f)
            lm_list.append(points)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return lm_list

# -----------------------------------------------------------------------------
# 1. Shape Data Load (partition 인자 유지)
# -----------------------------------------------------------------------------
def load_shape_data(dataset, data_root, partition=None):
    """ 다양한 데이터셋 포맷(Ply 폴더, Mat 파일, H5 파일)을 파싱하여 3D 좌표 형태로 통일함 """
    target_folder_name = None
    if partition == 'train':
        target_folder_name = 'train' 
    elif partition == 'test':
        target_folder_name = 'test'
        
    if target_folder_name:
        target_path = os.path.join(data_root, target_folder_name)
        if os.path.exists(target_path):
            shapes, names = read_ply_files_from_folder(target_path)
            if len(shapes) > 0:
                return shapes, names

    dataset_dir = os.path.join(data_root, dataset)

    if dataset == 'Ear296_Korean':
        ply_path = os.path.join(dataset_dir, 'template-registered_data')
        if os.path.exists(ply_path): 
            return read_ply_files_from_folder(ply_path)
        
        mat_path = os.path.join(dataset_dir, 'template_registered_data_mat', 'Ear296_Korean.mat')
        if os.path.exists(mat_path):
            data = sio.loadmat(mat_path)
            shapes = [s for s in data['shape_all'][0]]
            names = [f"mat_{i:03d}" for i in range(len(shapes))]
            return shapes, names

    elif dataset == 'Ear296_Korean_arg':
        aug_path = os.path.join(dataset_dir, 'augmented_samples_Random_12_HIGH')
        if os.path.exists(aug_path): return read_ply_files_from_folder(aug_path)

    elif dataset == 'BU-3DFE':
        path = os.path.join(data_root, 'BU-3DFE-dataset-mat', 'vertics_landmark_refine_Read.mat')
        if os.path.exists(path):
            f = h5py.File(path, 'r')
            shapes = [np.array(f[f['shape_all'][i][0]][()].transpose()) for i in range(len(f['shape_all']))]
            names = [f"BU3DFE_{i:03d}" for i in range(len(shapes))]
            return shapes, names
    
    return [], []

# -----------------------------------------------------------------------------
# 2. Landmark Position Load (partition 인자 유지)
# -----------------------------------------------------------------------------
def load_landmark_position(dataset, data_root, shape_all=None, partition=None):
    """ 정답(GT) 랜드마크의 3D 좌표를 추출함. (인덱스 매핑 방식과 직접 좌표 로드 방식 모두 지원) """
    target_folder_name = None
    if partition == 'train':
        target_folder_name = 'train'
    elif partition == 'test':
        target_folder_name = 'test'

    if target_folder_name:
        target_path = os.path.join(data_root, target_folder_name)
        if os.path.exists(target_path):
            lms = read_asc_files_from_folder(target_path)
            if len(lms) > 0:
                print(f">> Loaded {len(lms)} landmarks from [{target_folder_name}] directly.")
                return lms

    dataset_dir = os.path.join(data_root, dataset)
    
    if 'Ear' in dataset:
        target_folder = None
        if dataset == 'Ear296_Korean':
            target_folder = os.path.join(dataset_dir, 'template-registered_data')
        elif dataset == 'Ear296_Korean_arg':
            target_folder = os.path.join(dataset_dir, 'augmented_samples_Random_12_HIGH')
            
        if target_folder and os.path.exists(target_folder):
            landmarks = read_asc_files_from_folder(target_folder)
            if landmarks: return landmarks

        print(">> .asc files missing. Falling back to Index-based derivation.")
        index_path = os.path.join(data_root, 'Ear296_Korean', 'template_registered_data_mat', 'Ear296_Korean.mat')
        if not os.path.exists(index_path):
             index_path = os.path.join(data_root, dataset, 'template_registered_data_mat', 'Ear296_Korean.mat')
             
        if os.path.exists(index_path):
            data = sio.loadmat(index_path)
            raw = data['landmark_index_select_all'][0]
            common_indices = raw[0].flatten() - 1 
            
            landmark_positions = []
            
            shapes_only = shape_all
            if isinstance(shape_all, tuple):
                 shapes_only = shape_all[0]
            
            if shapes_only is None or len(shapes_only) == 0:
                 s_temp, _ = load_shape_data(dataset, data_root, partition)
                 shapes_only = s_temp

            for shape in shapes_only:
                landmark_positions.append(shape[common_indices, :])
            return landmark_positions
        else:
            raise FileNotFoundError("Neither .asc files nor .mat index file found!")

    elif dataset == 'BU-3DFE':
        path = os.path.join(data_root, 'BU-3DFE-dataset-mat', 'landmark_select_refine.mat')
        data = sio.loadmat(path)
        raw = data['landmark_position_select_all'][0]
        return [np.array(raw[k]) for k in range(len(raw))]
        
    return []

# -----------------------------------------------------------------------------
# Core Processing (가우시안 히트맵 및 샘플링)
# -----------------------------------------------------------------------------
def Gaussian_Heatmap(Distance, sigma):
    """
    [히트맵 생성 수학적 로직]
    이산적인 점(Discrete point) 하나를 정답으로 주면 모델이 학습하기 매우 어렵습니다.
    따라서 정답 랜드마크를 중심으로 반경(sigma)만큼 퍼져나가는 
    연속적인 가우시안 확률 분포(Continuous Probability Distribution)를 생성합니다.
    공식: exp(-(d^2) / (2 * sigma^2))
    """
    D2 = Distance * Distance
    S2 = 2.0 * sigma * sigma
    Exponent = D2 / S2
    heatmap = np.exp(-Exponent)
    return heatmap

def calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma):
    """ 모든 Point Cloud에 대해 각 랜드마크 위치로부터의 유클리디안 거리를 구해 히트맵을 생성합니다. """
    Heat_data_all = []
    
    for i in tqdm(range(len(shape_all)), desc="   Calc Heatmaps", unit="shape"):
        shape_i = shape_all[i]
        lm_i = landmark_position_sample[i]
        
        diff  = shape_i[:, np.newaxis, :] - lm_i[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        heat  = Gaussian_Heatmap(dists, sigma)
        
        Heat_data_all.append(heat)
    return Heat_data_all

def compute_sample_index(Heat_data, num_points, landmark_index, rand_seed):
    np.random.seed(rand_seed)
    point_num = np.array(Heat_data).shape[0]
    index_1 = np.arange(point_num)
    index = np.random.choice(index_1, size=num_points, replace=False)
    return index

def get_dists(points1, points2):
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists)
    return torch.sqrt(dists).float()

def fps(xyz, M):
    """
    [Farthest Point Sampling (FPS) 로직]
    수만 개의 3D 점들을 모델 입력 크기(예: 2048개)로 줄일 때, 
    무작위로 뽑지 않고 서로 가장 멀리 떨어져 있는 점들을 순차적으로 뽑는 방식입니다.
    이로 인해 3D 표면 전체의 기하학적 형태(Geometry)를 균일하게 유지하며 샘플링할 수 있습니다.
    """
    if USE_CPP_FPS_UTIL and xyz.is_cuda:
        xyz = xyz.contiguous()
        idx = fps_cpp(xyz, M) # C++ 커널 호출
        return idx.long()
    else:
        # PyTorch 기반 폴백(Fallback) 로직
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
        dists = torch.ones(B, N).to(device) * 1e5
        inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
        batchlists = torch.arange(0, B, dtype=torch.long).to(device)
        for i in range(M):
            centroids[:, i] = inds
            cur_point = xyz[batchlists, inds, :] 
            cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
            dists[cur_dist < dists] = cur_dist[cur_dist < dists]
            inds = torch.max(dists, dim=1)[1]
        return centroids

def random_sample(shape_all, Heat_data_all, num_points, rand_seed, sample_way, dataset, data_root):    
    print('   Start sampling...')
    if sample_way == 'FPS':
        FPS_matrix = [fps(torch.from_numpy(shape_all[i]).float().unsqueeze(0).to(device), num_points)
                      for i in tqdm(range(len(Heat_data_all)), desc="   FPS Sampling", unit="shape")]

        Heat_data_sample = [np.array(Heat_data_all[j])[FPS_matrix[j].squeeze(0).cpu(), :]
                            for j in range(len(Heat_data_all))]
        shape_sample     = [np.array(shape_all[j])[FPS_matrix[j].squeeze(0).cpu(), :]
                            for j in range(len(shape_all))]
        return Heat_data_sample, shape_sample

    elif sample_way == 'Random':
        try:
             pass 
        except:
            return [], []
            
    return [], []

# 🔥 복구된 평가/회귀 함수들 (절대 삭제 금지!)
def get_rigid(src, dst):
    """
    [Rigid Transformation (ICP 방식) 계산]
    SVD(특이값 분해)를 사용하여 두 점군 간의 최적의 회전 행렬(R)과 평행 이동 벡터(T)를 찾습니다.
    """
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    H = reduce(lambda s, p: s + np.outer(p[0], p[1]), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    try: u, s, v = np.linalg.svd(H)
    except: pass
    R = v.T.dot(u.T)
    T = - R.dot(src_mean) + dst_mean
    return np.hstack((R, T[:, np.newaxis]))

def landmark_regression(shape, Heatmap, regression_point_num, idx=None):
    """
    [예측된 히트맵에서 3D 랜드마크 복원]
    모델이 예측한 히트맵(확률)을 기반으로, 가장 확률이 높은 이웃 점들을 모은 뒤
    MDS(다차원 척도법)와 Rigid Transform을 사용하여 소수점 단위의 정밀한 3D 좌표를 역산해냅니다.
    단순한 Soft-argmax보다 표면 제약을 더 잘 유지하는 포스트 프로세싱 기법입니다.
    """
    shape   = shape.cpu().numpy()
    Heatmap = Heatmap.cpu().numpy()

    sortIdx = np.argsort(Heatmap, 0)

    shape_sort_select = np.array([shape[sortIdx[-regression_point_num:, ld]]
                                  for ld in range(Heatmap.shape[1])])
    Heatmap_sort_select = np.array([Heatmap[sortIdx[-regression_point_num:, ld], ld]
                                    for ld in range(Heatmap.shape[1])]).reshape(-1, regression_point_num, 1)

    shape_sort_select_rep = np.expand_dims(shape_sort_select, axis=-1).repeat(regression_point_num, axis=-1)
    shape2_exp_eer = shape_sort_select_rep.transpose(0, 1, 3, 2) - shape_sort_select_rep.transpose(0, 3, 1, 2)
    D_Matrix = np.linalg.norm(shape2_exp_eer, axis=3)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    shape_MDS = np.array([mds.fit_transform(D_Matrix[i]) for i in range(Heatmap.shape[1])])
    shape_MDS = np.concatenate((shape_MDS, np.zeros((Heatmap.shape[1], regression_point_num, 1))), axis=2)

    landmark2D = np.sum(Heatmap_sort_select.repeat(3, axis=2) * shape_MDS, axis=1) / \
                 (Heatmap_sort_select.sum(1) + 1e-6)

    N_neighbors = min(regression_point_num, 6)
    neigh = NearestNeighbors(n_neighbors=N_neighbors)
    IDX = []
    for i in range(Heatmap.shape[1]):
        neigh.fit(shape_MDS[i])
        IDX.append(neigh.kneighbors(landmark2D[i].reshape(1,-1))[1])
    IDX = np.array(IDX)

    shape_ext = np.array([shape_MDS[i, IDX[i], :].reshape(-1,3) -
                          landmark2D[i].reshape(1,-1).repeat(N_neighbors, axis=0)
                          for i in range(Heatmap.shape[1])])

    shape_ext_T = np.array([shape_sort_select[i, IDX[i], :]
                            for i in range(Heatmap.shape[1])]).reshape(-1, N_neighbors, 3)

    w1 = shape_ext - np.repeat(shape_ext.mean(1, keepdims=True), N_neighbors, axis=1)
    w2 = shape_ext_T - np.repeat(shape_ext_T.mean(1, keepdims=True), N_neighbors, axis=1)
    w1 = np.linalg.norm(w1.reshape(Heatmap.shape[1], -1), axis=1).reshape(-1, 1, 1)
    w2 = np.linalg.norm(w2.reshape(Heatmap.shape[1], -1), axis=1).reshape(-1, 1, 1)
    w1[w1 < 1e-6] = 1e-6
    w2[w2 < 1e-6] = 1e-6
    shape_ext = shape_ext * w2 / w1

    landmark3D = np.array([get_rigid(shape_ext[i], shape_ext_T[i])[:, 3]
                           for i in range(Heatmap.shape[1])])

    return torch.from_numpy(landmark3D).unsqueeze(0).to(device)

def get_3D_FAN_NME(pred_landmark, gt_landmark):
    """ NME(Normalized Mean Error) 기반 평가 지표 계산 """
    NME_single = torch.sum(torch.norm(pred_landmark - gt_landmark, dim=2), 0)
    NME = torch.mean(NME_single)
    return NME, NME_single

# =============================================================================
# 🌟 [신규 업데이트] Offline 7-Channel 피처 생성기 (Eigenvector 3 + Eigenvalue 1)
# =============================================================================
def compute_geometric_features_7ch(shapes, k=15):
    """
    [PCA / Eigen Decomposition 기반 기하학적 특징 추출기]
    - 목적: 학습 중 loss.py에서 매번 계산하면 GPU 부하가 너무 크기 때문에,
            전처리 단계에서 미리 표면의 '주방향(뼈대 방향)'과 '곡률(뾰족한 정도)'을 계산해 둡니다.
    - 과정:
      1. 각 점마다 K-NN으로 주변 이웃 점들을 모아 공분산 행렬(Covariance Matrix)을 만듭니다.
      2. 이 행렬을 고유 분해(Eigen Decomposition)합니다.
      3. 가장 큰 고유값에 해당하는 고유벡터(Eigenvector)는 그 부위가 뻗어있는 주방향(3D)을 나타냅니다.
      4. 가장 작은 고유값을 고유값의 합으로 나누면 곡률(Curvature Magnitude, 1D)을 얻을 수 있습니다.
    - 결과: 반환된 4채널 특징은 나중에 기존 XYZ(3)와 합쳐져 총 7채널이 됩니다.
    """
    # GPU VRAM OOM 방지를 위해 배치(Batch) 단위로 쪼개서 연산합니다.
    geom_list = []
    batch_size = 32 # VRAM 안전선
    
    for i in tqdm(range(0, len(shapes), batch_size), desc="   Calc 7-Ch Geometrics"):
        batch_shapes = shapes[i:i+batch_size]
        shapes_tensor = torch.tensor(np.array(batch_shapes), dtype=torch.float32).to(device)
        B, N, _ = shapes_tensor.shape
        
        # 1. K-NN 거리 계산 및 이웃 추출
        dist_matrix = torch.cdist(shapes_tensor, shapes_tensor)
        _, knn_indices = torch.topk(dist_matrix, k, dim=2, largest=False)
        
        idx_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        shapes_expanded = shapes_tensor.unsqueeze(1).expand(-1, N, -1, -1)
        knn_points = torch.gather(shapes_expanded, 2, idx_expanded)
        
        # 2. 공분산 행렬 생성 및 아이겐 분해 (Eigen Decomposition)
        center = knn_points.mean(dim=2, keepdim=True)
        centered = knn_points - center
        cov = torch.matmul(centered.transpose(2, 3), centered)
        eigval, eigvec = torch.linalg.eigh(cov)
        
        # 🔥 3. 아이겐벡터: 가장 긴 축(주방향) 추출 (3채널)
        principal_dir = eigvec[..., 2]
        dot_product = torch.sum(principal_dir * center.squeeze(2), dim=-1, keepdim=True)
        principal_dir = principal_dir * torch.sign(dot_product) # PCA 벡터의 방향성(부호) 통일
        
        # 🔥 4. 아이겐밸류: 가장 작은 값을 이용해 표면 변동성(Surface Variation), 즉 곡률 추출 (1채널)
        sum_eig = torch.sum(eigval, dim=-1) + 1e-6
        curvature = (eigval[..., 0] / sum_eig).unsqueeze(-1) 
        
        # 5. 방향(3) + 곡률(1) 병합하여 4채널 피처 생성
        geom_features = torch.cat([principal_dir, curvature], dim=-1)
        
        geom_list.extend(geom_features.cpu().numpy())
        
    return geom_list

# -----------------------------------------------------------------------------
# Main Sampling Function (7채널 병합 및 저장 로직 적용)
# -----------------------------------------------------------------------------
def main_sample(num_points, seed, sigma, sample_way, dataset, data_root='../Data', partition=None):
    """
    [데이터 전처리 파이프라인의 오케스트레이터]
    1. 원본 데이터 로드 -> 2. 히트맵 생성 -> 3. FPS 다운샘플링 -> 4. 기하학적 특징 계산 -> 5. NPY 파일로 저장
    """
    suffix = "sample" 
    if partition == 'train': suffix = "train"
    elif partition == 'test': suffix = "test"

    print(f'\n--- Processing: {dataset} [Partition: {partition if partition else "ALL"}] ---')
    
    shape_all, name_all = load_shape_data(dataset, data_root, partition)
    if len(shape_all) == 0:
        print(f"   [Warning] No data found for partition '{partition}'. Skipping.")
        return

    print(f'   Loaded {len(shape_all)} shapes.')

    landmark_position_sample = load_landmark_position(dataset, data_root, (shape_all, name_all), partition)
    print(f'   Loaded {len(landmark_position_sample)} landmarks.')

    print('   Calculating Heatmaps...')
    Heat_data_all = calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma)

    Heat_data_sample, shape_sample = random_sample(shape_all, Heat_data_all,
                                                   num_points, seed, sample_way, dataset, data_root)
    
    if len(Heat_data_sample) == 0:
        print("Sampling failed or empty.")
        return

    # 🌟 [핵심 파트] 7채널(XYZ 3 + 주방향 3 + 곡률 1) 데이터 오프라인 베이킹
    # 학습 시간을 단축하기 위해 무거운 연산을 여기서 미리 수행합니다.
    print('   Baking 7-Channel Geometric Features (Eigenvectors & Eigenvalues)...')
    geom_features = compute_geometric_features_7ch(shape_sample, k=15)
    
    shape_7ch_sample = []
    for i in range(len(shape_sample)):
        # 공간 좌표 (2048, 3) 과 미리 계산한 특징 (2048, 4) 를 이어붙여 최종 (2048, 7) 텐서로 결합
        shape_7ch = np.concatenate([shape_sample[i], geom_features[i]], axis=-1)
        shape_7ch_sample.append(shape_7ch)

    save_base_dir = os.path.join(data_root, f"{dataset}-npy")
    os.makedirs(save_base_dir, exist_ok=True)
    
    print(f"   Saving to: {save_base_dir} (Suffix: _{suffix})")
    np.save(os.path.join(save_base_dir, f'Heat_data_{suffix}.npy'), Heat_data_sample)
    
    # 모델(train.py, eval_all.py)에서 읽어들일 수 있도록 shape_{suffix}.npy 에 7채널 데이터를 덮어쓰기 형태로 저장
    np.save(os.path.join(save_base_dir, f'shape_{suffix}.npy'),      shape_7ch_sample) 
    
    np.save(os.path.join(save_base_dir, f'landmark_{suffix}.npy'),   landmark_position_sample)
    np.save(os.path.join(save_base_dir, f'name_{suffix}.npy'),       np.array(name_all))

    print("--- Done ---\n")