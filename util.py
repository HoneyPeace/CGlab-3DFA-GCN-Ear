'''
@Author: Yuan Wang (Modified by Researcher 5)
@File: util.py
@Description: Fixed folder name mismatch ('-npy' -> '_npy'). Restored all original functions.
'''

import os 
import glob
import numpy as np
import torch
from plyfile import PlyData
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from functools import reduce

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# [Helper] 파일 읽기 함수들
# -----------------------------------------------------------------------------
def read_ply_files_from_folder(folder_path):
    """ .ply 파일들을 읽어서 Shape(N, 3) 리스트와 Name 리스트 반환 """
    files = sorted(glob.glob(os.path.join(folder_path, "*.ply")))
    
    if len(files) == 0:
        raise FileNotFoundError(f"No .ply files found in {folder_path}")
    
    print(f"   Found {len(files)} PLY files. Reading...")
    shape_list = []
    name_list = []
    
    for f in tqdm(files, desc="Loading Shapes", unit="file"):
        try:
            name = os.path.splitext(os.path.basename(f))[0] # 파일명 추출
            plydata = PlyData.read(f)
            vertex = plydata['vertex']
            points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
            
            shape_list.append(points)
            name_list.append(name)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return shape_list, name_list

def load_landmarks_by_names(folder_path, names):
    """ Shape와 짝이 맞는 .asc 파일 로드 """
    print(f"   Loading matching .asc files...")
    lm_list = []
    valid_names = []
    valid_indices = []
    
    for idx, name in enumerate(tqdm(names, desc="Loading Landmarks")):
        asc_path = os.path.join(folder_path, name + ".asc")
        if os.path.exists(asc_path):
            try:
                try: points = np.loadtxt(asc_path, delimiter=',')
                except ValueError: points = np.loadtxt(asc_path)
                
                lm_list.append(points)
                valid_names.append(name)
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error reading {asc_path}: {e}")
        else:
            pass
            
    return lm_list, valid_names, valid_indices

# -----------------------------------------------------------------------------
# 1. Shape Data Load
# -----------------------------------------------------------------------------
def load_shape_data(dataset, data_root):
    dataset_dir = os.path.join(data_root, dataset)
    if os.path.exists(dataset_dir):
        return read_ply_files_from_folder(dataset_dir)
    else:
        raise FileNotFoundError(f"Folder not found: {dataset_dir}")

# -----------------------------------------------------------------------------
# 2. Landmark Load
# -----------------------------------------------------------------------------
def load_landmark_position(dataset, data_root, names_all):
    dataset_dir = os.path.join(data_root, dataset)
    return load_landmarks_by_names(dataset_dir, names_all)

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
    min_len = min(len(shape_all), len(landmark_position_sample))
    
    for i in tqdm(range(min_len), desc="Calculating Heatmaps"):
        shape_i = shape_all[i]
        lm_i = landmark_position_sample[i]

        diff  = shape_i[:, np.newaxis, :] - lm_i[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        heat  = Gaussian_Heatmap(dists, sigma)

        Heat_data_all.append(heat)
    return Heat_data_all

# [복구] 원본에 있던 함수 복구
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

def random_sample(shape_all, Heat_data_all, names_all, num_points, seed, sample_way):    
    print('   Start sampling...')
    min_len = min(len(shape_all), len(Heat_data_all))
    shape_all = shape_all[:min_len]
    Heat_data_all = Heat_data_all[:min_len]
    names_all = names_all[:min_len]

    if sample_way == 'FPS':
        FPS_matrix = []
        for i in tqdm(range(len(Heat_data_all)), desc="   Sampling (FPS)"):
            FPS_matrix.append(fps(torch.from_numpy(shape_all[i]).float().unsqueeze(0).to(device), num_points))

        Heat_data_sample = [np.array(Heat_data_all[j])[FPS_matrix[j].squeeze(0).cpu(), :]
                            for j in range(len(Heat_data_all))]
        shape_sample     = [np.array(shape_all[j])[FPS_matrix[j].squeeze(0).cpu(), :]
                            for j in range(len(shape_all))]
        return Heat_data_sample, shape_sample, names_all

    elif sample_way == 'Random':
        np.random.seed(seed)
        Heat_data_sample = []
        shape_sample = []
        
        for i in tqdm(range(len(shape_all)), desc="   Sampling (Random)"):
            total_points = shape_all[i].shape[0]
            if total_points >= num_points:
                idx = np.random.choice(total_points, num_points, replace=False)
            else:
                idx = np.random.choice(total_points, num_points, replace=True)
                
            Heat_data_sample.append(Heat_data_all[i][idx, :])
            shape_sample.append(shape_all[i][idx, :])
            
        return Heat_data_sample, shape_sample, names_all

    return [], [], []

def get_rigid(src, dst):
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    H = reduce(lambda s, p: s + np.outer(p[0], p[1]), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    try: u, s, v = np.linalg.svd(H)
    except: pass
    R = v.T.dot(u.T)
    T = - R.dot(src_mean) + dst_mean
    return np.hstack((R, T[:, np.newaxis]))

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
# Main Processing Function (수정됨: 이름 추적 + NPY 저장)
# -----------------------------------------------------------------------------
def main_sample(num_points, seed, sigma, sample_way, dataset, data_root='../Data'):
    print(f'\n--- Processing: {dataset} ---')
    
    # 1. Shape & Name Load
    shape_all, names_all = load_shape_data(dataset, data_root) 
    print(f'   Loaded {len(shape_all)} shapes.')

    # 2. Landmark Load (Name 매칭)
    landmark_position_sample, valid_names, valid_indices = load_landmark_position(dataset, data_root, names_all)
    print(f'   Loaded {len(landmark_position_sample)} landmarks.')
    
    shape_all = [shape_all[i] for i in valid_indices]
    names_all = valid_names

    print('   Calculating Heatmaps...')
    Heat_data_all = calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma)

    # 3. Sampling
    Heat_data_sample, shape_sample, names_sample = random_sample(
        shape_all, Heat_data_all, names_all, num_points, seed, sample_way
    )

    # 4. Save (★ 핵심 수정: -npy -> _npy)
    save_base_dir = os.path.join(data_root, f"{dataset}_npy") # 하이픈(-)을 언더바(_)로 변경
    os.makedirs(save_base_dir, exist_ok=True)
    print(f"   Saving to: {save_base_dir}")
    
    np.save(os.path.join(save_base_dir, f'Heat_data_{dataset}.npy'), Heat_data_sample)
    np.save(os.path.join(save_base_dir, f'shape_{dataset}.npy'),     shape_sample)
    np.save(os.path.join(save_base_dir, f'landmark_{dataset}.npy'),  landmark_position_sample)
    np.save(os.path.join(save_base_dir, f'names_{dataset}.npy'),     names_sample)
    
    print("--- Done ---\n")