'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: util.py
@Description: Completely Separated Processing for Train and Test + Full Legacy Support.
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# [Helper] 파일 읽기 함수들 (수정 없음 - 원본 유지)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# [Core 1] 안전한 데이터 로드 (파일 이름 매칭)
# -----------------------------------------------------------------------------
def load_paired_data(mode, data_root):
    """
    .ply(Shape)와 .asc(Landmark) 파일의 이름을 비교하여
    정확히 쌍이 맞는 데이터만 로드하여 반환합니다.
    Args:
        mode (str): 'train' or 'test'
        data_root (str): 데이터 루트 경로 (../Data)
    Returns:
        shape_list, lm_list (List of np.array)
    """
    folder_path = os.path.join(data_root, mode)
    
    # 1. 파일 목록 검색
    ply_files = sorted(glob.glob(os.path.join(folder_path, "*.ply")))
    # ply가 없으면 obj도 찾아봄 (백업)
    if len(ply_files) == 0:
        ply_files = sorted(glob.glob(os.path.join(folder_path, "*.obj")))
        
    asc_files = sorted(glob.glob(os.path.join(folder_path, "*.asc")))

    # 2. 파일명(확장자 제외)을 기준으로 딕셔너리 생성
    ply_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in ply_files}
    asc_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in asc_files}

    # 3. 교집합(둘 다 존재하는 파일명)만 추출 및 정렬
    common_names = sorted(list(set(ply_dict.keys()) & set(asc_dict.keys())))

    if len(common_names) == 0:
        # 폴더가 없거나 파일이 없으면 빈 리스트 반환 (에러 방지)
        print(f"⚠️ [Warning] No paired data found in {folder_path}")
        return [], []

    print(f"   Matched {len(common_names)} pairs in '{mode}'. Loading...")

    shape_list = []
    lm_list = []

    for name in tqdm(common_names, desc=f"Loading {mode}", unit="file"):
        try:
            # --- Load Shape ---
            ply_path = ply_dict[name]
            plydata = PlyData.read(ply_path)
            vertex = plydata['vertex']
            points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
            
            # --- Load Landmark ---
            asc_path = asc_dict[name]
            try: 
                lm_points = np.loadtxt(asc_path, delimiter=',')
            except ValueError: 
                lm_points = np.loadtxt(asc_path)
            
            # 둘 다 성공했을 때만 리스트에 추가 (데이터 무결성 보장)
            shape_list.append(points)
            lm_list.append(lm_points)

        except Exception as e:
            print(f"Error loading pair for '{name}': {e}")
            continue

    return shape_list, lm_list


# -----------------------------------------------------------------------------
# Core Processing & Sampling (수정 없음 - 원본 유지)
# -----------------------------------------------------------------------------
def Gaussian_Heatmap(Distance, sigma):
    D2 = Distance * Distance
    S2 = 2.0 * sigma * sigma
    Exponent = D2 / S2
    heatmap = np.exp(-Exponent)
    return heatmap

def calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma):
    Heat_data_all = []
    if len(shape_all) != len(landmark_position_sample):
        print(f"[Warning] Mismatch! Shapes: {len(shape_all)}, Landmarks: {len(landmark_position_sample)}")
        min_len = min(len(shape_all), len(landmark_position_sample))
        shape_all = shape_all[:min_len]
        landmark_position_sample = landmark_position_sample[:min_len]

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
        print(f"   [FPS] Processing {len(Heat_data_all)} shapes... (This may take a while)")
        FPS_matrix = [fps(torch.from_numpy(shape_all[i]).float().unsqueeze(0).to(device), num_points)
                      for i in tqdm(range(len(Heat_data_all)), desc="   FPS Sampling", unit="shape")]

        Heat_data_sample = [np.array(Heat_data_all[j])[FPS_matrix[j].squeeze(0).cpu(), :]
                            for j in range(len(Heat_data_all))]
        shape_sample     = [np.array(shape_all[j])[FPS_matrix[j].squeeze(0).cpu(), :]
                            for j in range(len(shape_all))]
        return Heat_data_sample, shape_sample

    elif sample_way == 'Random':
        pass
    return [], []

def get_rigid(src, dst):
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    H = reduce(lambda s, p: s + np.outer(p[0], p[1]), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    try: u, s, v = np.linalg.svd(H)
    except: pass
    R = v.T.dot(u.T)
    T = - R.dot(src_mean) + dst_mean
    return np.hstack((R, T[:, np.newaxis]))

# [중요] 기존 코드의 긴 로직 그대로 유지
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

# -----------------------------------------------------------------------------
# Main Sampling Function (수정됨: 분리 로직 적용 및 파일명 저장)
# -----------------------------------------------------------------------------
def main_sample(num_points, seed, sigma, sample_way, dataset, data_root='../Data', partition=None):
    """
    partition: 'train' | 'test' | None
    - 'train'이 들어오면 Train 폴더만 읽고 shape_train.npy로 저장
    - 'test'가 들어오면 test 폴더만 읽고 shape_test.npy로 저장
    """
    # 접미사 결정
    suffix = "sample" # 기본값 (기존 호환)
    if partition == 'train': suffix = "train"
    elif partition == 'test': suffix = "test"

    print(f'\n--- Processing: {dataset} [Partition: {partition if partition else "ALL"}] ---')
    
    # 1. 데이터 로드 (partition 전달하여 분리)
    shape_all, name_all = load_shape_data(dataset, data_root, partition)
    
    if len(shape_all) == 0:
        print(f"   [Warning] No data found for partition '{partition}'. Skipping.")
        return

    print(f'   Loaded {len(shape_all)} shapes.')

    # 2. 랜드마크 로드 (partition 전달)
    landmark_position_sample = load_landmark_position(dataset, data_root, shape_all, partition)
    print(f'   Loaded {len(landmark_position_sample)} landmarks.')

    print('   Calculating Heatmaps...')
    Heat_data_all = calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma)

    Heat_data_sample, shape_sample = random_sample(shape_all, Heat_data_all,
                                                   num_points, seed, sample_way, dataset, data_root)
    
    # 3. 저장 (접미사 사용)
    save_base_dir = os.path.join(data_root, f"{dataset}-npy")
    os.makedirs(save_base_dir, exist_ok=True)
    print(f"   Saving to: {save_base_dir} (Suffix: _{suffix})")
    
    np.save(os.path.join(save_base_dir, f'Heat_data_{suffix}.npy'), Heat_data_sample)
    np.save(os.path.join(save_base_dir, f'shape_{suffix}.npy'),      shape_sample)
    np.save(os.path.join(save_base_dir, f'landmark_{suffix}.npy'),   landmark_position_sample)
    np.save(os.path.join(save_base_dir, f'name_{suffix}.npy'),       np.array(name_all))
    
    print("--- Done (Memory Cleared) ---\n")