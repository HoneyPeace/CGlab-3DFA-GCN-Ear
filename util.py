'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: util.py
@Description: Loads .ply for Shape AND .asc for Landmarks correctly.
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
# [Helper] 파일 읽기 함수들
# -----------------------------------------------------------------------------
def read_ply_files_from_folder(folder_path):
    """ .ply 파일들을 읽어서 Shape(N, 3) 리스트로 반환 """
    files = sorted(glob.glob(os.path.join(folder_path, "*.ply")))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(folder_path, "*.obj"))) # obj 백업
    
    if len(files) == 0:
        raise FileNotFoundError(f"No .ply files found in {folder_path}")
    
    print(f"   Found {len(files)} PLY files (Shape). Reading...")
    shape_list = []
    for f in tqdm(files, desc="Loading Shapes", unit="file"):
        try:
            plydata = PlyData.read(f)
            vertex = plydata['vertex']
            points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
            shape_list.append(points)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    return shape_list

def read_asc_files_from_folder(folder_path):
    """ .asc 파일들을 읽어서 Landmark(L, 3) 리스트로 반환 """
    files = sorted(glob.glob(os.path.join(folder_path, "*.asc")))
    if len(files) == 0:
        # 파일이 없으면 빈 리스트 반환 (나중에 Index 방식 시도 위해)
        return None 
    
    print(f"   Found {len(files)} ASC files (Landmark). Reading...")
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
# 1. Shape Data Load (PLY)
# -----------------------------------------------------------------------------
def load_shape_data(dataset, data_root):
    dataset_dir = os.path.join(data_root, dataset)
    
    # 원본 데이터
    if dataset == 'Ear296_Korean':
        ply_path = os.path.join(dataset_dir, 'template-registered_data')
        if os.path.exists(ply_path): return read_ply_files_from_folder(ply_path)
        
        # fallback: mat
        mat_path = os.path.join(dataset_dir, 'template_registered_data_mat', 'Ear296_Korean.mat')
        if os.path.exists(mat_path):
            data = sio.loadmat(mat_path)
            return [s for s in data['shape_all'][0]]

    # 증강 데이터
    elif dataset == 'Ear296_Korean_arg':
        aug_path = os.path.join(dataset_dir, 'augmented_samples_Random_12_HIGH')
        if os.path.exists(aug_path): return read_ply_files_from_folder(aug_path)
        else: raise FileNotFoundError(f"Folder not found: {aug_path}")

    # 기존 데이터셋
    elif dataset == 'BU-3DFE':
        path = os.path.join(data_root, 'BU-3DFE-dataset-mat', 'vertics_landmark_refine_Read.mat')
        f = h5py.File(path, 'r')
        return [np.array(f[f['shape_all'][i][0]][()].transpose()) for i in range(len(f['shape_all']))]
    
    return []

# -----------------------------------------------------------------------------
# 2. Landmark Position Load (ASC 우선 -> Index 차선)
# -----------------------------------------------------------------------------
def load_landmark_position(dataset, data_root, shape_all=None):
    dataset_dir = os.path.join(data_root, dataset)
    
    if 'Ear' in dataset:
        # [전략 1] .asc 파일이 있으면 그걸 정답지로 쓴다 (가장 정확함)
        target_folder = None
        if dataset == 'Ear296_Korean':
            target_folder = os.path.join(dataset_dir, 'template-registered_data')
        elif dataset == 'Ear296_Korean_arg':
            target_folder = os.path.join(dataset_dir, 'augmented_samples_Random_12_HIGH')
            
        if target_folder and os.path.exists(target_folder):
            landmarks = read_asc_files_from_folder(target_folder)
            if landmarks is not None:
                print(">> Valid .asc landmarks found. Using them as Ground Truth.")
                return landmarks

        # [전략 2] .asc가 없으면 원본의 Index 정보를 빌려와서 계산한다 (Fallback)
        print(">> .asc files missing. Falling back to Index-based derivation.")
        index_path = os.path.join(data_root, 'Ear296_Korean', 'template_registered_data_mat', 'Ear296_Korean.mat')
        if not os.path.exists(index_path):
             index_path = os.path.join(data_root, dataset, 'template_registered_data_mat', 'Ear296_Korean.mat')
             
        if os.path.exists(index_path):
            data = sio.loadmat(index_path)
            raw = data['landmark_index_select_all'][0]
            # Matlab(1-based) -> Python(0-based)
            common_indices = raw[0].flatten() - 1 
            
            landmark_positions = []
            if shape_all is None: shape_all = load_shape_data(dataset, data_root)
            for shape in shape_all:
                landmark_positions.append(shape[common_indices, :])
            return landmark_positions
        else:
            raise FileNotFoundError("Neither .asc files nor .mat index file found!")

    # 기존 데이터셋
    elif dataset == 'BU-3DFE':
        path = os.path.join(data_root, 'BU-3DFE-dataset-mat', 'landmark_select_refine.mat')
        data = sio.loadmat(path)
        raw = data['landmark_position_select_all'][0]
        return [np.array(raw[k]) for k in range(len(raw))]
        
    return []

# -----------------------------------------------------------------------------
# Core Processing & Sampling (기존 유지)
# -----------------------------------------------------------------------------
def Gaussian_Heatmap(Distance, sigma):
    D2 = Distance * Distance                 # (Ni, L)
    S2 = 2.0 * sigma * sigma                 # scalar
    Exponent = D2 / S2                       # (Ni, L)
    heatmap = np.exp(-Exponent)              # (Ni, L)
    return heatmap

def calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma):
    Heat_data_all = []                       # list of (Ni, L)
    if len(shape_all) != len(landmark_position_sample):
        print(f"[Warning] Mismatch! Shapes: {len(shape_all)}, Landmarks: {len(landmark_position_sample)}")
        min_len = min(len(shape_all), len(landmark_position_sample))
        shape_all = shape_all[:min_len]
        landmark_position_sample = landmark_position_sample[:min_len]

    for i in range(len(shape_all)):
        shape_i = shape_all[i]              # (Ni, 3)
        lm_i = landmark_position_sample[i]  # (L, 3)

        diff  = shape_i[:, np.newaxis, :] - lm_i[np.newaxis, :, :]  # (Ni, L, 3)
        dists = np.linalg.norm(diff, axis=2)                        # (Ni, L)
        heat  = Gaussian_Heatmap(dists, sigma)                      # (Ni, L)

        Heat_data_all.append(heat)          # append (Ni, L)
    return Heat_data_all                    # list length S, each (Ni, L)

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
        FPS_matrix = [fps(torch.from_numpy(shape_all[i]).float().unsqueeze(0).to(device), num_points)
                      for i in range(len(Heat_data_all))]                         # each: (1, N)

        Heat_data_sample = [np.array(Heat_data_all[j])[FPS_matrix[j].squeeze(0).cpu(), :]
                            for j in range(len(Heat_data_all))]                   # list of (N, L)
        shape_sample     = [np.array(shape_all[j])[FPS_matrix[j].squeeze(0).cpu(), :]
                            for j in range(len(shape_all))]                       # list of (N, 3)
        return Heat_data_sample, shape_sample

    elif sample_way == 'Random':
        landmark_index_select_all = load_landmark_index(dataset, data_root)       # list of landmark indices
        np.random.seed(rand_seed)
        random_matrix = [compute_sample_index(Heat_data_all[i], num_points,
                                              landmark_index_select_all[i]-1, rand_seed)
                         for i in range(len(Heat_data_all))]                      # each: (N,)

        Heat_data_sample = [np.array(Heat_data_all[j])[random_matrix[j], :]
                            for j in range(len(Heat_data_all))]                   # list of (N, L)
        shape_sample     = [np.array(shape_all[j])[random_matrix[j], :]
                            for j in range(len(shape_all))]                       # list of (N, 3)
        return Heat_data_sample, shape_sample
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

def landmark_regression(shape, Heatmap, regression_point_num, idx=None):
    shape   = shape.cpu().numpy()             # (N, 3)
    Heatmap = Heatmap.cpu().numpy()           # (N, L)

    sortIdx = np.argsort(Heatmap, 0)          # (N, L)

    shape_sort_select = np.array([shape[sortIdx[-regression_point_num:, ld]]
                                  for ld in range(Heatmap.shape[1])])              # (L, r, 3)
    Heatmap_sort_select = np.array([Heatmap[sortIdx[-regression_point_num:, ld], ld]
                                    for ld in range(Heatmap.shape[1])]).reshape(-1, regression_point_num, 1)  # (L, r, 1)

    shape_sort_select_rep = np.expand_dims(shape_sort_select, axis=-1).repeat(regression_point_num, axis=-1)  # (L, r, 3, r)
    shape2_exp_eer = shape_sort_select_rep.transpose(0, 1, 3, 2) - shape_sort_select_rep.transpose(0, 3, 1, 2)  # (L, r, r, 3)
    D_Matrix = np.linalg.norm(shape2_exp_eer, axis=3)                        # (L, r, r)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
    shape_MDS = np.array([mds.fit_transform(D_Matrix[i]) for i in range(Heatmap.shape[1])])  # (L, r, 2)
    shape_MDS = np.concatenate((shape_MDS, np.zeros((Heatmap.shape[1], regression_point_num, 1))), axis=2)  # (L, r, 3)

    landmark2D = np.sum(Heatmap_sort_select.repeat(3, axis=2) * shape_MDS, axis=1) / \
                 (Heatmap_sort_select.sum(1) + 1e-6)                           # (L, 3)

    N_neighbors = min(regression_point_num, 6)
    neigh = NearestNeighbors(n_neighbors=N_neighbors)
    IDX = []
    for i in range(Heatmap.shape[1]):
        neigh.fit(shape_MDS[i])                                                # (r, 3)
        IDX.append(neigh.kneighbors(landmark2D[i].reshape(1,-1))[1])
    IDX = np.array(IDX)                                                        # (L, 1, N_neighbors)

    shape_ext = np.array([shape_MDS[i, IDX[i], :].reshape(-1,3) -
                          landmark2D[i].reshape(1,-1).repeat(N_neighbors, axis=0)
                          for i in range(Heatmap.shape[1])])                   # (L, N_neighbors, 3)

    shape_ext_T = np.array([shape_sort_select[i, IDX[i], :]
                            for i in range(Heatmap.shape[1])]).reshape(-1, N_neighbors, 3)  # (L, N_neighbors, 3)

    w1 = shape_ext - np.repeat(shape_ext.mean(1, keepdims=True), N_neighbors, axis=1)       # (L, N_neighbors, 3)
    w2 = shape_ext_T - np.repeat(shape_ext_T.mean(1, keepdims=True), N_neighbors, axis=1)   # (L, N_neighbors, 3)
    w1 = np.linalg.norm(w1.reshape(Heatmap.shape[1], -1), axis=1).reshape(-1, 1, 1)         # (L,1,1)
    w2 = np.linalg.norm(w2.reshape(Heatmap.shape[1], -1), axis=1).reshape(-1, 1, 1)         # (L,1,1)
    w1[w1 < 1e-6] = 1e-6
    w2[w2 < 1e-6] = 1e-6
    shape_ext = shape_ext * w2 / w1                                                         # (L, N_neighbors, 3)

    landmark3D = np.array([get_rigid(shape_ext[i], shape_ext_T[i])[:, 3]
                           for i in range(Heatmap.shape[1])])           # (L, 3)

    return torch.from_numpy(landmark3D).unsqueeze(0).to(device)         # (1, L, 3)

#def get_3D_FAN_NME(pred, gt):
#    if pred.dim() == 2: pred = pred.unsqueeze(0)
#    if gt.dim() == 2: gt = gt.unsqueeze(0)
#    return torch.mean(torch.norm(pred - gt, dim=2)), torch.norm(pred - gt, dim=2)

def get_3D_FAN_NME(pred_landmark, gt_landmark):
    NME_single = torch.sum(torch.norm(pred_landmark - gt_landmark, dim=2), 0)
    NME = torch.mean(NME_single)
    return NME, NME_single

def main_sample(num_points, seed, sigma, sample_way, dataset, data_root='../Data'):
    print(f'\n--- Processing: {dataset} ---')
    shape_all = load_shape_data(dataset, data_root)                               # list of (Ni, 3)
    print(f'   Loaded {len(shape_all)} shapes.')

    landmark_position_sample = load_landmark_position(dataset, data_root, shape_all)  # list of (L, 3)
    print(f'   Loaded {len(landmark_position_sample)} landmarks.')

    print('   Calculating Heatmaps...')
    Heat_data_all = calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma)  # list of (Ni, L)

    Heat_data_sample, shape_sample = random_sample(shape_all, Heat_data_all,
                                                   num_points, seed, sample_way, dataset, data_root)
    # Heat_data_sample: list of (N, L), shape_sample: list of (N, 3)

    save_base_dir = os.path.join(data_root, f"{dataset}-npy")
    os.makedirs(save_base_dir, exist_ok=True)
    print(f"   Saving to: {save_base_dir}")
    np.save(os.path.join(save_base_dir, 'Heat_data_sample.npy'), Heat_data_sample)       # (S, N, L)
    np.save(os.path.join(save_base_dir, 'shape_sample.npy'),      shape_sample)          # (S, N, 3)
    np.save(os.path.join(save_base_dir, 'landmark_sample.npy'),   landmark_position_sample) # (S, L, 3)
    print("--- Done ---\n")