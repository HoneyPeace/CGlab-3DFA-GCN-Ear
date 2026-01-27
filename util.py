'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: util.py
@Description: Includes Partition Logic, Name Saving & FPS Progress Bar.
'''

import os 
import glob
import numpy as np
import scipy.io as sio
import torch
import h5py
import open3d as o3d

from plyfile import PlyData
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from functools import reduce

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: util.py
@Description: Includes Partition Logic, Name Saving & FPS Progress Bar + [NEW] MST Normal Estimation
'''

import os 
import glob
import numpy as np
import scipy.io as sio
import torch
import h5py
import open3d as o3d  # [필수] Open3D 추가

from plyfile import PlyData
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from functools import reduce

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
# -----------------------------------------------------------------------------
# [NEW] Normal Vector Calculation (MST + Centroid Check)
# -----------------------------------------------------------------------------
def compute_normals_consistent(points, k_neighbors=15):
    
    
    # 1. Open3D PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 2. 로컬 노말 추정
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30)
    )
    
    # 3. MST(Minimum Spanning Tree)를 이용한 방향성 전파 (Consistency 확보)
    pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)
    
    # 4. 전체 방향성 교정 (Centroid 기준)
    points_np = np.asarray(pcd.points)
    normals_np = np.asarray(pcd.normals)
    center = np.mean(points_np, axis=0)
    
    # (P - C) 벡터와 Normal의 내적 계산
    vec_from_center = points_np - center
    dot_products = np.sum(vec_from_center * normals_np, axis=1)
    
    # 내적이 음수인 비율이 절반 이상이면 전체를 뒤집음 (일관성 유지한 채 방향만 반전)
    if np.mean(dot_products < 0) > 0.5:
        normals_np *= -1
        
    return np.concatenate([points_np, normals_np], axis=-1)
"""
# -----------------------------------------------------------------------------
# [수정됨] Normal Vector Calculation (Scale-Invariant KNN)
# -----------------------------------------------------------------------------
def compute_normals_consistent(points, k_neighbors=15):

    # 1. Open3D PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 2. [핵심 수정] 로컬 노말 추정 방식을 'Hybrid' -> 'KNN'으로 변경
    # radius=0.1 제한을 없애고, 무조건 주변 30개 점을 찾아서 평면을 계산함
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15) 
    )
    
    # 3. MST(Minimum Spanning Tree)를 이용한 방향성 전파
    # (이웃끼리 방향을 비슷하게 맞춤)
    pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)
    
    # 4. 전체 방향성 교정 (Centroid 기준)
    points_np = np.asarray(pcd.points)
    normals_np = np.asarray(pcd.normals)
    center = np.mean(points_np, axis=0)
    
    # (P - C) 벡터와 Normal의 내적 계산
    vec_from_center = points_np - center
    dot_products = np.sum(vec_from_center * normals_np, axis=1)
    
    # 내적이 음수인(안쪽을 보는) 비율이 절반 이상이면 전체를 뒤집음 (바깥쪽을 보게)
    if np.mean(dot_products < 0) > 0.5:
        normals_np *= -1
        
    return np.concatenate([points_np, normals_np], axis=-1)

# -----------------------------------------------------------------------------
# [Helper] 파일 읽기 함수들 (Name 반환 유지)
# -----------------------------------------------------------------------------
def read_ply_files_from_folder(folder_path):
    """ .ply 파일들을 읽어서 (Shape리스트, Name리스트) 튜플로 반환 """
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
            # 파일명 추출 (확장자 제거)
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
    """ .asc 파일들을 읽어서 Landmark(L, 3) 리스트로 반환 """
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
    # 1. Partition에 따른 폴더 우선 탐색
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

    # 2. 기존 데이터셋 이름 하드코딩 처리 (Legacy Support)
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
    # 1. Partition에 따른 폴더 우선 탐색
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

    # 2. 기존 로직
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

        # [Fallback] Index 기반
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
# Core Processing (Safety Check 없음 - 유지)
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
        # [수정] tqdm 게이지바 추가
        # 리스트 컴프리헨션을 tqdm으로 감싸서 진행상황 표시
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
# Main Sampling Function (수정: Normal 계산 적용)
# -----------------------------------------------------------------------------
def main_sample(num_points, seed, sigma, sample_way, dataset, data_root='../Data', partition=None):
    suffix = "sample" 
    if partition == 'train': suffix = "train"
    elif partition == 'test': suffix = "test"

    print(f'\n--- Processing: {dataset} [Partition: {partition if partition else "ALL"}] ---')
    
    # 1. Load (이름 포함)
    shape_all, name_all = load_shape_data(dataset, data_root, partition)
    if len(shape_all) == 0:
        print(f"   [Warning] No data found for partition '{partition}'. Skipping.")
        return

    print(f'   Loaded {len(shape_all)} shapes.')

    # 2. Landmarks Load
    landmark_position_sample = load_landmark_position(dataset, data_root, (shape_all, name_all), partition)
    print(f'   Loaded {len(landmark_position_sample)} landmarks.')

    # 3. Heatmap
    print('   Calculating Heatmaps...')
    Heat_data_all = calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma)

    # 4. Sampling (XYZ only)
    Heat_data_sample, shape_sample = random_sample(shape_all, Heat_data_all,
                                                   num_points, seed, sample_way, dataset, data_root)
    if len(Heat_data_sample) == 0:
        print("Sampling failed or empty.")
        return

    # -------------------------------------------------------------------------
    # [NEW] Normal Vector Computing Loop
    # -------------------------------------------------------------------------
    print('   Computing Normals with MST & Centroid Alignment...')
    shape_with_normals = []
    
    # tqdm으로 진행상황 표시
    for s in tqdm(shape_sample, desc="   Normals", unit="shape"):
        # 여기서 (N, 3) -> (N, 6) 변환
        sn = compute_normals_consistent(s, k_neighbors=15)
        shape_with_normals.append(sn)
    
    # 저장할 변수 교체
    final_shape_data = shape_with_normals 

    # 5. Save
    save_base_dir = os.path.join(data_root, f"{dataset}-npy")
    os.makedirs(save_base_dir, exist_ok=True)
    
    print(f"   Saving to: {save_base_dir} (Suffix: _{suffix})")
    
    np.save(os.path.join(save_base_dir, f'Heat_data_{suffix}.npy'), Heat_data_sample)
    
    # [변경] Normal이 포함된 (N, 6) 데이터 저장
    np.save(os.path.join(save_base_dir, f'shape_{suffix}.npy'),      final_shape_data)
    
    np.save(os.path.join(save_base_dir, f'landmark_{suffix}.npy'),   landmark_position_sample)
    np.save(os.path.join(save_base_dir, f'name_{suffix}.npy'),       np.array(name_all))

    print("--- Done ---\n")