'''
@Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
@File: util.py
@Description: 
[S2G 3D 랜드마크 검출 모델 - 오프라인 데이터 전처리 파이프라인]
1) 원본 3D Point Cloud(ply/obj)와 정답 랜드마크(asc/mat) 매칭 및 로드
2) 공간적 균일성을 유지하는 FPS 다운샘플링 및 가우시안 히트맵 생성
3) 🌟 7-Channel Geometric Feature 사전 연산 (PyTorch3D 초고속 C++ KNN 탑재 & VRAM 최적화)
4) 예외 처리 및 로깅 강화
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
import sys
from pathlib import Path

# ==========================================================
# 🚀 1. 데이터 전처리용 초고속 C++ FPS 커널 로드
# ==========================================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "utils" / "pointnet2_ops_lib"))

try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cpp
    USE_CPP_FPS_UTIL = True
    print(">>> [SUCCESS] 🚀 util.py에서 C++ 초고속 FPS 커널을 성공적으로 로드했습니다!")
except ImportError:
    USE_CPP_FPS_UTIL = False
    print(">>> [WARNING] ⚠️ util.py에서 C++ FPS 커널을 찾지 못했습니다. 기존 방식(PyTorch)으로 진행합니다.")

# ==========================================================
# 🚀 2. 초고속 C++ KNN 커널 (PyTorch3D) 로드 (여기가 추가된 부분입니다!)
# ==========================================================
try:
    from pytorch3d.ops import knn_points
    USE_PYTORCH3D_KNN = True
    print(">>> [SUCCESS] 🚀 util.py: PyTorch3D C++ KNN 커널 장착 완료! (데이터 굽기 초고속화)")
except ImportError:
    USE_PYTORCH3D_KNN = False
    print(">>> [WARNING] ⚠️ util.py: PyTorch3D가 없습니다. 메모리 최적화(Chunk) 모드로 안전하게 굽습니다.")
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# [Helper] 파일 읽기 함수들
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
            print(f"[Error] Error reading {f}: {e}")
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
            print(f"[Error] Error reading {f}: {e}")
    return lm_list

# -----------------------------------------------------------------------------
# 1. Shape Data Load
# -----------------------------------------------------------------------------
def load_shape_data(dataset, data_root, partition=None):
    target_folder_name = None
    if partition == 'train': target_folder_name = 'train' 
    elif partition == 'test': target_folder_name = 'test'
    elif partition in ['val', 'validation', 'valiation']:
        target_folder_name = dataset if dataset else partition
        
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
# 2. Landmark Position Load
# -----------------------------------------------------------------------------
def load_landmark_position(dataset, data_root, shape_all=None, partition=None):
    target_folder_name = None
    if partition == 'train': target_folder_name = 'train'
    elif partition == 'test': target_folder_name = 'test'
    elif partition in ['val', 'validation', 'valiation']:
        target_folder_name = dataset if dataset else partition

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
# Core Processing
# -----------------------------------------------------------------------------
def Gaussian_Heatmap(Distance, sigma):
    D2 = Distance * Distance
    S2 = 2.0 * sigma * sigma
    Exponent = D2 / S2
    heatmap = np.exp(-Exponent)
    return heatmap

def calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma):
    Heat_data_all = []
    for i in tqdm(range(len(shape_all)), desc="   Calc Heatmaps", unit="shape"):
        shape_i = shape_all[i]
        lm_i = landmark_position_sample[i]
        
        diff  = shape_i[:, np.newaxis, :] - lm_i[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        heat  = Gaussian_Heatmap(dists, sigma)
        
        Heat_data_all.append(heat)
    return Heat_data_all

def compute_sample_index(Heat_data, num_points, landmark_index):
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
    if USE_CPP_FPS_UTIL and xyz.is_cuda:
        xyz = xyz.contiguous()
        idx = fps_cpp(xyz, M)
        return idx.long()
    else:
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

def random_sample(shape_all, Heat_data_all, num_points, sample_way):    
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
        try: pass 
        except: return [], []
    return [], []

def get_rigid(src, dst):
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    H = reduce(lambda s, p: s + np.outer(p[0], p[1]), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    try: 
        u, s, v = np.linalg.svd(H)
        R = v.T.dot(u.T)
        T = - R.dot(src_mean) + dst_mean
        return np.hstack((R, T[:, np.newaxis]))
    except np.linalg.LinAlgError as e:
        print(f"[Warning] get_rigid SVD computation failed: {e}. Returning Identity matrix.")
        return np.hstack((np.eye(3), np.zeros((3, 1)))) 

def landmark_regression(shape, Heatmap, regression_point_num, idx=None):
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
    NME_single = torch.sum(torch.norm(pred_landmark - gt_landmark, dim=2), 0)
    NME = torch.mean(NME_single)
    return NME, NME_single

# =============================================================================
# 🌟 [메모리 최적화] Chunk 기반 k-NN 연산 (OOM 원천 차단 & 100% 동일 결과)
# =============================================================================
def chunked_knn(points, k, chunk_size=512):
    if USE_PYTORCH3D_KNN:
        # 🚀 C++ 커널이 있으면 청크로 쪼갤 필요도 없이 8192개 한 번에 1초 컷 계산!
        _, idx, _ = knn_points(points, points, K=k)
        return idx
    else:
        # 🧊 C++ 커널이 없으면 기존에 만든 안전한 청크 모드로 작동
        B, N, C = points.shape
        knn_indices = torch.zeros(B, N, k, dtype=torch.long, device=points.device)
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            chunk = points[:, i:end_i, :] 
            dist_matrix = torch.cdist(chunk, points) 
            _, chunk_knn_idx = torch.topk(dist_matrix, k, dim=2, largest=False)
            knn_indices[:, i:end_i, :] = chunk_knn_idx
            
        return knn_indices

def compute_geometric_features_7ch(shapes, k=15, batch_size=8):
    geom_list = []
    
    for i in tqdm(range(0, len(shapes), batch_size), desc="   Calc Geometrics"):
        batch_shapes = shapes[i:i+batch_size]
        shapes_tensor = torch.tensor(np.array(batch_shapes), dtype=torch.float32).to(device)
        B, N, _ = shapes_tensor.shape
        
        try:
            with torch.no_grad():
                knn_indices = chunked_knn(shapes_tensor, k, chunk_size=512)
                
                idx_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
                shapes_expanded = shapes_tensor.unsqueeze(1).expand(-1, N, -1, -1)
                knn_points = torch.gather(shapes_expanded, 2, idx_expanded)
                
                center = knn_points.mean(dim=2, keepdim=True)
                centered = knn_points - center
                
                cov = torch.matmul(centered.transpose(2, 3), centered) / (k - 1)
                eigval, eigvec = torch.linalg.eigh(cov)
                
                principal_dir = eigvec[..., 2]
                dot_product = torch.sum(principal_dir * center.squeeze(2), dim=-1, keepdim=True)
                principal_dir = principal_dir * torch.sign(dot_product)
                
                sum_eig = torch.sum(eigval, dim=-1) + 1e-6
                curvature = (eigval[..., 0] / sum_eig).unsqueeze(-1) 
                
                geom_features = torch.cat([principal_dir, curvature], dim=-1)
                geom_list.extend(geom_features.cpu().numpy())
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[Error] VRAM 부족! Chunk Size 혹은 Batch Size({batch_size})를 줄여주세요.")
                torch.cuda.empty_cache()
            else:
                print(f"\n[Error] 기하 피처 연산 실패: {e}")
            raise e
            
    return geom_list

# -----------------------------------------------------------------------------
# Main Sampling Function 
# -----------------------------------------------------------------------------
def main_sample(num_points, seed, sigma, sample_way, dataset, data_root='../Data', partition=None):
    suffix = "sample" 
    if partition == 'train': suffix = "train"
    elif partition == 'test': suffix = "test"
    elif partition in ['val', 'validation', 'valiation']: suffix = partition

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

    Heat_data_sample, shape_sample = random_sample(shape_all, Heat_data_all, num_points, sample_way)
    
    if len(Heat_data_sample) == 0:
        print("Sampling failed or empty.")
        return

    print('   Baking 6-Ch & 7-Ch Geometric Features...')
    geom_features = compute_geometric_features_7ch(shape_sample, k=15, batch_size=8)
    
    shape_3ch_sample, shape_6ch_sample, shape_7ch_sample = [], [], []
    for i in range(len(shape_sample)):
        xyz = shape_sample[i]
        vector = geom_features[i][:, :3] 
        curv = geom_features[i][:, 3:]   
        
        shape_3ch_sample.append(xyz)
        shape_6ch_sample.append(np.concatenate([xyz, vector], axis=-1))
        shape_7ch_sample.append(np.concatenate([xyz, vector, curv], axis=-1))

    save_base_dir = os.path.join(data_root, f"{dataset}-npy")
    os.makedirs(save_base_dir, exist_ok=True)
    
    print(f"   Saving to: {save_base_dir} (Suffix: _{suffix})")
    np.save(os.path.join(save_base_dir, f'Heat_data_{suffix}.npy'), Heat_data_sample)
    np.save(os.path.join(save_base_dir, f'landmark_{suffix}.npy'),  landmark_position_sample)
    np.save(os.path.join(save_base_dir, f'name_{suffix}.npy'),      np.array(name_all))
    
    np.save(os.path.join(save_base_dir, f'shape_3ch_{suffix}.npy'),  shape_3ch_sample) 
    np.save(os.path.join(save_base_dir, f'shape_6ch_{suffix}.npy'),  shape_6ch_sample) 
    np.save(os.path.join(save_base_dir, f'shape_{suffix}.npy'),      shape_7ch_sample) 

    print("--- Done ---\n")
