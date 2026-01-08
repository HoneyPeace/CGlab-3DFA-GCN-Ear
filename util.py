'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: util.py
@Description: Loads .ply and .asc from 'Train' and 'test' folders directly + Saves Filenames.
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
# [Helper] 파일 읽기 함수들 (수정됨: 이름 반환 추가)
# -----------------------------------------------------------------------------
def read_ply_files_from_folder(folder_path):
    """ .ply 파일들을 읽어서 (Shape리스트, Name리스트) 튜플로 반환 """
    files = sorted(glob.glob(os.path.join(folder_path, "*.ply")))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(folder_path, "*.obj"))) # obj 백업
    
    if len(files) == 0:
        # 파일이 없으면 빈 리스트 두 개 반환
        return [], []
    
    print(f"   Found {len(files)} PLY files (Shape) in {os.path.basename(folder_path)}. Reading...")
    shape_list = []
    name_list = [] # [변경점 1] 파일명 저장용 리스트 생성

    for f in tqdm(files, desc="Loading Shapes", unit="file"):
        try:
            # [변경점 1] 파일명 추출 (경로와 확장자 제거) 예: "data/Train/001.ply" -> "001"
            basename = os.path.splitext(os.path.basename(f))[0]
            name_list.append(basename)

            plydata = PlyData.read(f)
            vertex = plydata['vertex']
            points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
            shape_list.append(points)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return shape_list, name_list # [변경점 1] 튜플 (shapes, names) 반환

def read_asc_files_from_folder(folder_path):
    """ .asc 파일들을 읽어서 Landmark(L, 3) 리스트로 반환 """
    if not os.path.exists(folder_path):
        return []

    files = sorted(glob.glob(os.path.join(folder_path, "*.asc")))
    if len(files) == 0:
        return []
    
    print(f"   Found {len(files)} ASC files (Landmark) in {os.path.basename(folder_path)}. Reading...")
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
# 1. Shape Data Load (Modified: Train/test folders & Name return)
# -----------------------------------------------------------------------------
def load_shape_data(dataset, data_root):
    # [변경점 2] Train과 test 폴더 탐색 및 이름(Names) 리스트 통합
    train_dir = os.path.join(data_root, 'Train')
    test_dir = os.path.join(data_root, 'test')
    
    all_shapes = []
    all_names = [] # [변경점 2] 전체 이름 리스트
    
    # Train 폴더
    if os.path.exists(train_dir):
        # read_ply_files_from_folder가 이제 (shapes, names)를 뱉으므로 받아서 처리
        s, n = read_ply_files_from_folder(train_dir)
        all_shapes.extend(s)
        all_names.extend(n)
        
    # test 폴더
    if os.path.exists(test_dir):
        s, n = read_ply_files_from_folder(test_dir)
        all_shapes.extend(s)
        all_names.extend(n)
        
    if len(all_shapes) > 0:
        return all_shapes, all_names

    # [우선순위 2] 기존 데이터셋 로직 (Fallback) - 여기도 Names 반환하도록 수정됨
    dataset_dir = os.path.join(data_root, dataset)
    
    if dataset == 'Ear296_Korean':
        ply_path = os.path.join(dataset_dir, 'template-registered_data')
        if os.path.exists(ply_path): 
            return read_ply_files_from_folder(ply_path) 
        
        # .mat 파일일 경우 이름이 없으므로 임의로 생성 (mat_000, mat_001...)
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
# 2. Landmark Position Load (Modified: Train/test folders)
# -----------------------------------------------------------------------------
def load_landmark_position(dataset, data_root, shape_all=None):
    # [우선순위 1] Train/test 폴더 탐색
    train_dir = os.path.join(data_root, 'Train')
    test_dir = os.path.join(data_root, 'test')
    
    all_landmarks = []
    
    if os.path.exists(train_dir):
        all_landmarks.extend(read_asc_files_from_folder(train_dir))
        
    if os.path.exists(test_dir):
        all_landmarks.extend(read_asc_files_from_folder(test_dir))
        
    if len(all_landmarks) > 0:
        print(f">> Total Landmarks Loaded: {len(all_landmarks)}")
        return all_landmarks

    # [우선순위 2] 기존 로직 (Fallback)
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

        # Index fallback
        print(">> .asc files missing. Falling back to Index-based derivation.")
        index_path = os.path.join(data_root, 'Ear296_Korean', 'template_registered_data_mat', 'Ear296_Korean.mat')
        if not os.path.exists(index_path):
             index_path = os.path.join(data_root, dataset, 'template_registered_data_mat', 'Ear296_Korean.mat')
             
        if os.path.exists(index_path):
            data = sio.loadmat(index_path)
            raw = data['landmark_index_select_all'][0]
            common_indices = raw[0].flatten() - 1 
            
            landmark_positions = []
            
            # shape_all이 리스트인지 튜플(shapes, names)인지 확인하여 처리
            if shape_all is None: 
                s, _ = load_shape_data(dataset, data_root)
                shape_all = s
            elif isinstance(shape_all, tuple): # 만약 (shapes, names) 형태로 넘어왔다면
                 shape_all = shape_all[0] # shape만 꺼냄
                
            for shape in shape_all:
                landmark_positions.append(shape[common_indices, :])
            return landmark_positions

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
# Main Sampling Function (수정됨: 이름 저장)
# -----------------------------------------------------------------------------
def main_sample(num_points, seed, sigma, sample_way, dataset, data_root='../Data'):
    print(f'\n--- Processing: {dataset} (Looking for Train/test folders) ---')
    
    # [변경점 3] shapes 뿐만 아니라 name_all 리스트도 받아옴
    shape_all, name_all = load_shape_data(dataset, data_root)
    print(f'   Loaded {len(shape_all)} shapes.')

    landmark_position_sample = load_landmark_position(dataset, data_root, shape_all)
    print(f'   Loaded {len(landmark_position_sample)} landmarks.')

    print('   Calculating Heatmaps...')
    Heat_data_all = calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma)

    Heat_data_sample, shape_sample = random_sample(shape_all, Heat_data_all,
                                                   num_points, seed, sample_way, dataset, data_root)
    
    save_base_dir = os.path.join(data_root, f"{dataset}-npy")
    os.makedirs(save_base_dir, exist_ok=True)
    print(f"   Saving to: {save_base_dir}")
    np.save(os.path.join(save_base_dir, 'Heat_data_sample.npy'), Heat_data_sample)
    np.save(os.path.join(save_base_dir, 'shape_sample.npy'),      shape_sample)
    np.save(os.path.join(save_base_dir, 'landmark_sample.npy'),   landmark_position_sample)
    
    # [변경점 3] 파일명 리스트도 name_sample.npy로 저장
    np.save(os.path.join(save_base_dir, 'name_sample.npy'), np.array(name_all))
    print("   [INFO] Sample names saved to name_sample.npy")
    
    print("--- Done ---\n")