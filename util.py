'''
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: util.py
@Time: 2021/12/02 09:59 AM
'''

import os 
import numpy as np
import scipy.io as sio
import torch
import h5py
import cv2
import sys
import matplotlib.pyplot as plt
from functools import reduce
import torch.nn.functional as F
from sklearn.metrics import auc
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D
from My_args import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파라미터 설정
args = parser.parse_args()

def load_shape_data(dataset):
    if dataset == 'BU-3DFE':
        vertics_landmark_name = './BU-3DFE-dataset-mat/vertics_landmark_refine_Read.mat'
        vertics_landmark_all = h5py.File(vertics_landmark_name, 'r')
        shape_all = vertics_landmark_all['shape_all'][0]
        shape_all = [np.array(vertics_landmark_all[vertics_landmark_all['shape_all'][i][0]].value.transpose()) for i in range(len(vertics_landmark_all['shape_all']))]
    elif dataset == 'FRGC':
        vertics_landmark_name = './FRGC-dataset-mat/FRGC_vertics_landmark_Read.mat'
        vertics_landmark_all = h5py.File(vertics_landmark_name, 'r')
        shape_all = vertics_landmark_all['FRGC_shape_all'][0]
        shape_all = [np.array(vertics_landmark_all[vertics_landmark_all['FRGC_shape_all'][i][0]].value.transpose()) for i in range(len(vertics_landmark_all['FRGC_shape_all']))]
    #귀 이어 데이터 추가 (임시, 구조다름)
    elif dataset == 'Ear296_Korean':
        vertics_landmark_name = './EarData/Ear296_Korean/template_registered_data_mat/Ear296_Korean.mat'
        vertics_landmark_all = sio.loadmat(vertics_landmark_name)
        shape_all = vertics_landmark_all['shape_all'][0]
        shape_all = [s for s in shape_all]
    return shape_all

'''
    elif dataset == 'HeadEar200_Korean':
        vertics_landmark_name = './EarData/HeadEar200_Korean/template_registered_data_mat/HeadEar200_Korean.mat'
        vertics_landmark_all = h5py.File(vertics_landmark_name, 'r')
        shape_all = vertics_landmark_all['shape_all'][0]
        shape_all = [np.array(vertics_landmark_all[vertics_landmark_all['HeadEar200_Korean_shape_all'][i][0]].value.transpose()) for i in range(len(vertics_landmark_all['HeadEar200_Korean_shape_all']))]
'''
        

def load_landmark_index(dataset):
    if dataset == 'BU-3DFE':
        landmark_index_name = './BU-3DFE-dataset-mat/landmark_select_refine.mat'
        landmark_index_all = sio.loadmat(landmark_index_name)
        landmark_index_select_all = landmark_index_all['landmark_index_select_all'][0]
        landmark_index_select_all = [np.array(landmark_index_select_all[k]) for k in range(len(landmark_index_select_all))]
    elif dataset == 'FRGC':
        landmark_index_name = './FRGC-dataset-mat/FRGC_landmark_select.mat'
        landmark_index_all = sio.loadmat(landmark_index_name)
        landmark_index_select_all = landmark_index_all['FRGC_landmark_index_select_all'][0]
        landmark_index_select_all = [np.array(landmark_index_select_all[k]) for k in range(len(landmark_index_select_all))]
    #귀 이어 데이터 추가 (임시, 구조다름)
    elif dataset == 'Ear296_Korean':
        landmark_index_name = './EarData/Ear296_Korean/template_registered_data_mat/Ear296_Korean.mat'
        landmark_index_all = sio.loadmat(landmark_index_name)  # ✅ scipy 방식으로 불러오기
        landmark_index_select_all = landmark_index_all['landmark_index_select_all'][0]
        landmark_index_select_all = [np.array(landmark_index_select_all[k]) for k in range(len(landmark_index_select_all))]
    else:
        landmark_index_name = './FaceScape-publish-dataset-mat/FaceScape_landmark_select.mat'
        landmark_index_all = sio.loadmat(landmark_index_name)
        landmark_index_select_all = landmark_index_all['FaceScape_landmark_index_all'][0]
        landmark_index_select_all = [np.array(landmark_index_select_all[k]) for k in range(len(landmark_index_select_all))]
    
    return landmark_index_select_all


def load_landmark_position(dataset):
    if dataset == 'BU-3DFE':
        landmark_position_name = './BU-3DFE-dataset-mat/landmark_select_refine.mat'
        landmark_position_all = sio.loadmat(landmark_position_name)
        landmark_position_sample = landmark_position_all['landmark_position_select_all'][0]
        landmark_position_sample = [np.array(landmark_position_sample[k]) for k in range(len(landmark_position_sample))] 
    elif dataset == 'FRGC':
        landmark_position_name = './FRGC-dataset-mat/FRGC_landmark_select.mat'
        landmark_position_all = sio.loadmat(landmark_position_name)
        landmark_position_sample = landmark_position_all['FRGC_landmark_position_select_all'][0]
        landmark_position_sample = [np.array(landmark_position_sample[k]) for k in range(len(landmark_position_sample))] 
    #귀 이어 데이터 추가 (임시, 구조다름)
    elif dataset == 'Ear296_Korean':
        landmark_position_name = './EarData/Ear296_Korean/template_registered_data_mat/Ear296_Korean.mat'
        landmark_position_all = sio.loadmat(landmark_position_name)
        landmark_position_sample = landmark_position_all['landmark_position_select_all'][0]
        landmark_position_sample = [np.array(landmark_position_sample[k]) for k in range(len(landmark_position_sample))]
    else:
        landmark_position_name = './FaceScape-publish-dataset-mat/FaceScape_landmark_select.mat'
        landmark_position_all = sio.loadmat(landmark_position_name)
        landmark_position_sample = landmark_position_all['FaceScape_landmark_position_all'][0]
        landmark_position_sample = [np.array(landmark_position_sample[k]) for k in range(len(landmark_position_sample))]  
    return landmark_position_sample


def load_Heatmap_data():
    Heat_data_all = np.load('Heat_data_all.npy', allow_pickle=True)
    return Heat_data_all


def calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma):
    Heat_data_all = []
    for i in range(len(shape_all)):
        shape_i = shape_all[i].reshape(shape_all[i].shape[0], 1, shape_all[i].shape[1]).repeat(landmark_position_sample[i].shape[0], axis=1)
        Euclidean_distance_i = np.linalg.norm((shape_i - landmark_position_sample[i]), axis=2)
        Heat_data_i = Gaussian_Heatmap(Euclidean_distance_i, sigma)
        Heat_data_all.append(Heat_data_i)
    return Heat_data_all


def Gaussian_Heatmap(Distance, sigma):
    D2 = Distance * Distance
    S2 = 2.0 * sigma * sigma
    Exponent = D2 / S2
    heatmap = np.exp(-Exponent)
    return heatmap


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
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists)  # Very Important for dist = 0.
    return torch.sqrt(dists).float()


'''
Farthest Point Sampling from 
Qi C R, Yi L, Su H, et al. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. NIPS 2017.
The following module is based on https://github.com/erikwijmans/Pointnet2_PyTorch
'''
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


def random_sample(shape_all, Heat_data_all, num_points, rand_seed, sample_way, dataset):    
    print('Start load landmark index data! ')
    landmark_index_select_all = load_landmark_index(dataset)
    print('Load landmark index data successfully')
    if sample_way == 'Random':
        np.random.seed(rand_seed)
        random_matrix = [compute_sample_index(Heat_data_all[i], num_points, landmark_index_select_all[i]-1, rand_seed) for i in range(len(Heat_data_all))]
        print('Finish the caculation of random matrix')
        Heat_data_sample = [np.array(Heat_data_all[j])[random_matrix[j], :] for j in range(len(Heat_data_all))]
        shape_sample = [np.array(shape_all[j])[random_matrix[j], :] for j in range(len(shape_all))]
    elif sample_way == 'FPS':
        FPS_matrix = [fps(torch.from_numpy(shape_all[i]).unsqueeze(0).to(device), num_points) for i in range(len(Heat_data_all))]
        print('Finish the caculation of random matrix')
        Heat_data_sample = [np.array(Heat_data_all[j])[FPS_matrix[j].squeeze(0).cpu(), :] for j in range(len(Heat_data_all))]
        shape_sample = [np.array(shape_all[j])[FPS_matrix[j].squeeze(0).cpu(), :] for j in range(len(shape_all))]
    else:
        raise AssertionError('Invalid Sample Way')
    return Heat_data_sample, shape_sample


def soft_argmax(Heatmap, point, alpha):
    Heatmap = Heatmap * alpha
    soft_max = F.softmax(Heatmap, dim=2)
    indices_kernel = torch.arange(start=0, end=point.size(2), device=device).float()
    conv = soft_max * indices_kernel
    landmark_index_pred = conv.sum(2).floor().type_as(indices_kernel)
    landmark_coords_pred = [point[i, :, landmark_index_pred[i].long()].unsqueeze(0) for i in range(point.size(0))]
    landmark_coords_pred = torch.cat(landmark_coords_pred, dim=0)
    return landmark_coords_pred.permute(0, 2, 1)


def My_MDS(D, d=2):
    DSquare = D
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis = 0)
    rowMean = np.mean(DSquare, axis = 1)
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    eigVal, eigVec = np.linalg.eig(B)
    eigValSorted_indices = np.argsort(eigVal)
    topd_eigVec = eigVec[:,eigValSorted_indices[:-d-1:-1]] 
    X = np.dot(topd_eigVec, np.sqrt(np.diag(eigVal[:-d-1:-1])))
    return X

def landmark_regression(shape, Heatmap, regression_point_num, idx=None):
    """
    :params: shape [num_point, dims]
    :params: Heatmap [num_point, landmarks]
    :return: landmark3D [num_point, landmarks, 3]
    """
    shape = shape.cpu().numpy()
    Heatmap = Heatmap.cpu().numpy()
    Heatmap_sort = np.sort(Heatmap, 0)
    sortIdx = np.argsort(Heatmap, 0)
    ### Select r points with maximum values on each heatmap ###
    shape_sort_select = np.array([shape[sortIdx[-regression_point_num:, ld]] for ld in range(Heatmap.shape[1])]) 
    Heatmap_sort_select = np.array([Heatmap[sortIdx[-regression_point_num:, ld], ld] for ld in range(Heatmap.shape[1])]).reshape(-1, regression_point_num, 1) 

    shape_sort_select_rep = np.expand_dims(shape_sort_select, axis=-1).repeat(regression_point_num, axis=-1) 
    shape2_exp_eer = shape_sort_select_rep.transpose(0, 1, 3, 2) - shape_sort_select_rep.transpose(0, 3, 1, 2)
    ### Compute the distance matrix ###
    D_Matrix = np.linalg.norm(shape2_exp_eer, axis=3)
    Heatmap_weight = Heatmap_sort_select.repeat(regression_point_num, axis=-1)
    Distance_matrix = D_Matrix #* Heatmap_weight
    
    #mds = MDS(n_components=2, dissimilarity='precomputed')
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)  #06/10 ---> random_state=0 추가하여 디버깅용 mds 랜덤성 고정, 추후 삭제 요망
    
    shape_MDS = np.array([mds.fit_transform(Distance_matrix[i]) for i in range(Heatmap.shape[1])])
    shape_MDS = np.concatenate((shape_MDS, np.zeros((Heatmap.shape[1], regression_point_num, 1))), axis=2)
    '''
    ### Apply MDS to D_Matrix to obtain a dimension-degraded version of local shape ###
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)  # 06/10 ---> 디버깅용 mds 랜덤성 고정

    shape_MDS = []
    for i in range(Heatmap.shape[1]):
        D = Distance_matrix[i]
        D_sym = (D + D.T) / 2  # 대칭화
        shape_2d = mds.fit_transform(D_sym)  # (regression_point_num, 2)
        shape_3d = np.concatenate((shape_2d, np.zeros((regression_point_num, 1))), axis=1)  # z=0 추가 → (regression_point_num, 3)
        shape_MDS.append(shape_3d)

    shape_MDS = np.array(shape_MDS)  # 최종 shape: (num_landmarks, regression_point_num, 3)
    '''

    # 06/10 --->  디버깅 체크: Heatmap 합, MDS 편향, 반복된 이웃
    heatmap_sum = Heatmap_sort_select.sum(1)
    if (heatmap_sum < 1e-6).any():
        #print(f"[Sample {idx}]  히트맵크기=1e-6")
        np.save(f"debug/sample{idx}_heatmap_sum.npy", heatmap_sum)

    for i in range(shape_MDS.shape[0]):
        if np.allclose(shape_MDS[i], shape_MDS[i][0]):
            print(f"[Sample {idx}]  shape_MDS[{i}]가 모두 동일")
            np.save(f"debug/sample{idx}_shapeMDS_flat.npy", shape_MDS[i])

    landmark2D = np.sum(Heatmap_sort_select.repeat(3, axis=2) * shape_MDS, axis=1) / Heatmap_sort_select.sum(1)
    N = 6
    neigh = NearestNeighbors(n_neighbors=N)
    IDX = []
    for i in range(Heatmap.shape[1]):
        neigh.fit(shape_MDS[i])
        IDX_ = neigh.kneighbors(landmark2D[i].reshape(1,-1))[1]
        IDX.append(IDX_)

    #06/10 ---> 디버깅용 IDX
    IDX = np.array(IDX)
    for i in range(IDX.shape[0]):
        if len(np.unique(IDX[i])) == 1:
            print(f"[Sample {idx}]  IDX[{i}]가 모두 같은 포인트 가리킴")
            np.save(f"debug/sample{idx}_IDX_repeat.npy", IDX[i])

    shape_ext = np.array([shape_MDS[i, IDX[i], :].reshape(-1,3) - landmark2D[i].reshape(1,-1).repeat(N, axis=0) for i in range(Heatmap.shape[1])])
    shape_ext_T = np.array([shape_sort_select[i, IDX[i], :] for i in range(Heatmap.shape[1])]).reshape(-1,N,3)
    ### shape Centralization and Scale uniformization ###
    w1 = shape_ext - np.repeat(shape_ext.mean(1, keepdims=True), N, axis=1)    
    w2 = shape_ext_T - np.repeat(shape_ext_T.mean(1, keepdims=True), N, axis=1)   
    w1 = np.linalg.norm(w1.reshape(Heatmap.shape[1], -1), axis=1).reshape(-1, 1, 1)  
    w2 = np.linalg.norm(w2.reshape(Heatmap.shape[1], -1), axis=1).reshape(-1, 1, 1) 
    
    #06.11 코드 추가 (0 나눗셈 방지)
    w1[w1 < 1e-6] = 1e-6
    w2[w2 < 1e-6] = 1e-6
    
    #06.10 코드 추가 ---> 계산 오류 및 분모 0 확인
    if (w1 == 0).any():
        os.makedirs("debug", exist_ok=True)  # 💾 debug 폴더 자동 생성
        print("Zero w1 found!")
        print(w1)
        print(shape_ext)
        if idx is not None:
            print(f"[Sample {idx}] 0 detected in w1.")
            np.save(f"debug/sample{idx}_w1_zero.npy", w1)
        else:
            print(" idx is None. Skipping sample-specific debug save.")
            np.save("debug/sample_unknown_w1_zero.npy", w1)
        #input()
    if np.isnan(w1).any():
        os.makedirs("debug", exist_ok=True)
        print(" NaN detected in w1.")
        print(w1)
        print(shape_ext)
        if idx is not None:
            print(f"[Sample {idx}] NaN detected in w1.")
            np.save(f"debug/sample{idx}_w1_nan.npy", w1)
        else:
            print(" idx is None. Skipping sample-specific debug save.")
            np.save("debug/sample_unknown_w1_nan.npy", w1)
        #input()
    
    #w1 = np.where(w1 == 0, 1e-6, w1)   # 06/09 추가 <-- w1 0으로 분모가 되지 않게 하기위함
    shape_ext = shape_ext * w2 / w1  
    ### Get the 3D landmark coordinates after registration ###
    landmark3D = np.array([get_rigid(shape_ext[i], shape_ext_T[i])[:, 3] for i in range(Heatmap.shape[1])])
    return torch.from_numpy(landmark3D).unsqueeze(0).to(device)


def get_rigid(src, dst):
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    H = reduce(lambda s, p: s + np.outer(p[0], p[1]), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    
    try:
        u, s, v = np.linalg.svd(H)
    except Exception as e:
        print(e)
        #input()
        print("src_mean")
        print(src_mean)
        #input()
        print("dst_mean")
        print(dst_mean)
        #input()
        print(f"H.shape: {H.shape}")
        #input()
        print(H)
        print(f"np.any(np.isnan(H)): {np.any(np.isnan(H))}")
        #input()
        print("src")
        print(src.shape)
        print(src)
        #input()
        print("dst")
        print(dst.shape)
        print(dst)
        
    R = v.T.dot(u.T)
    T = - R.dot(src_mean) + dst_mean
    return np.hstack((R, T[:, np.newaxis]))


def get_3D_FAN_NME(pred_landmark, gt_landmark):
    NME_single = torch.sum(torch.norm(pred_landmark - gt_landmark, dim=2), 0)
    NME = torch.mean(NME_single)
    return NME, NME_single
"""
#임시 
def get_3D_FAN_NME(pred_landmark, gt_landmark):
    # 차원 자동 확장: [L, 3] → [1, L, 3]
    if pred_landmark.dim() == 2:
        pred_landmark = pred_landmark.unsqueeze(0)
    if gt_landmark.dim() == 2:
        gt_landmark = gt_landmark.unsqueeze(0)

    dists = torch.norm(pred_landmark - gt_landmark, dim=2)  # shape: [B, L]
    nme = torch.mean(dists)  # 평균 거리
    return nme, dists
"""

def calc_error_rate_i(pred_landmark_coords, gt_landmark_coords):
    error_single = torch.norm(pred_landmark_coords - gt_landmark_coords, dim=1)
    error = torch.mean(error_single)
    return error, error_single

def main_sample(num_points, seed, sigma, sample_way, dataset):
    # load point clouds of faces
    print('Start load shape data ! ')
    shape_all = load_shape_data(dataset)
    print('Load shape data successfully')
    # load landmark position of faces
    print('Start load landmark position data ! ')
    landmark_position_sample = load_landmark_position(dataset)
    print('Load landmark position data successfully')
    # compute the distance(Geodesic or Euclidean distance) from landmarks to all points
    print('Start calculate and save all Heatmaps !')
    Heat_data_all = calculateHeatMap_Euclidean(shape_all, landmark_position_sample, sigma)
    print('Calculate and save all Heatmaps successfully')
    Heat_data_sample, shape_sample, = random_sample(shape_all, Heat_data_all, num_points, seed, sample_way, dataset)
    np.save('./%s-npy/Heat_data_sample.npy' % dataset, Heat_data_sample)
    np.save('./%s-npy/shape_sample.npy' % dataset, shape_sample)
    np.save('./%s-npy/landmark_sample.npy' % dataset, landmark_position_sample)

    # 추가 저장: 조건별 폴더에 복사 저장 ---> 07/04
    save_folder = os.path.join(f"{dataset}-npy", f"FPS{num_points}_sigma{sigma}")
    os.makedirs(save_folder, exist_ok=True)

    # 파일 저장
    np.save(os.path.join(save_folder, "Heat_data_sample.npy"), Heat_data_sample)
    np.save(os.path.join(save_folder, "shape_sample.npy"), shape_sample)
    np.save(os.path.join(save_folder, "landmark_sample.npy"), landmark_position_sample)

    print(f" 추가 저장 완료: {save_folder}")

    ###  GT 히트맵 시각화 (100개마다) 추가 ----> 07/04
    vis_folder = os.path.join(save_folder, "GT_heatmap")
    os.makedirs(vis_folder, exist_ok=True)

    for idx in range(len(shape_sample)):
        if idx % 100 != 0:
            continue  # 100개마다 저장

        points_np = shape_sample[idx]                    # (num_points, 3)
        heatmap_np = Heat_data_sample[idx]               # (num_points, num_landmarks)
        heatmap_np = heatmap_np.T                        # → (num_landmarks, num_points)

        landmark_num = heatmap_np.shape[0]

        for landmark_id in range(min(40, landmark_num)):
            colors = heatmap_np[landmark_id]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, cmap='jet', s=1)

            # ⛰️ 위에서 내려다보는 시점으로 설정
            ax.view_init(elev=90, azim=-90)

            plt.colorbar(sc, label=f"GT Heatmap value for landmark {landmark_id}")
            plt.title(f"Sample {idx:03d} - GT Heatmap (Landmark {landmark_id})")
            plt.savefig(os.path.join(vis_folder, f"sample{idx:03d}_landmark{landmark_id}.png"))
            plt.close()

    print(f" GT 히트맵 시각화 완료: {vis_folder}")


"""
#06/11 ---> 디버깅 예측이랑 GT 히트맵 비교 (작동안함)
def save_heatmap_3d(point, gt_heatmap, pred_heatmap, epoch, sample_idx, save_dir='heatmap_debug'):

    os.makedirs(save_dir, exist_ok=True)

    point = point.squeeze(0).cpu().numpy()               # (N, 3)
    gt_heatmap = gt_heatmap.squeeze(0).cpu().numpy()     # (N, L)
    pred_heatmap = pred_heatmap.squeeze(0).detach().cpu().numpy().T  # (L, N)

    N = point.shape[0]

    for lm_id in range(min(3, gt_heatmap.shape[1])):
        if gt_heatmap.shape[0] != N or pred_heatmap.shape[1] != N:
            print(f"[ Skip] shape mismatch at epoch {epoch}, sample {sample_idx}, LM {lm_id}")
            continue

        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(point[:, 0], point[:, 1], point[:, 2], c=gt_heatmap[:, lm_id], cmap='Blues', s=2)
        ax1.set_title(f'GT Heatmap - LM {lm_id}')

        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(point[:, 0], point[:, 1], point[:, 2], c=pred_heatmap[lm_id], cmap='Reds', s=2)
        ax2.set_title(f'Pred Heatmap - LM {lm_id}')

        plt.suptitle(f'Epoch {epoch}, Sample {sample_idx}')
        plt.savefig(f'{save_dir}/heatmap_e{epoch}_s{sample_idx}_lm{lm_id}.png')
        plt.close()
"""