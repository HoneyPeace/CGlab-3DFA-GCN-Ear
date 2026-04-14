import torch
import torch.nn as nn
import torch.nn.functional as F
from deeppa_semseg import DeepPA_semseg

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "utils" / "pointnet2_ops_lib"))

# 🔥 초고속 C++ 커널 로드
try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cpp
    USE_CPP_FPS = True
    print(">>> [INFO] DeepPA 모드: 초고속 C++ FPS 커널을 사용합니다.")
except ImportError:
    USE_CPP_FPS = False

def farthest_point_sample(xyz, npoint):
    """
    [Farthest Point Sampling (FPS) 래퍼 함수]
    """
    if USE_CPP_FPS:
        xyz = xyz.contiguous() 
        idx = fps_cpp(xyz, npoint)
        return idx.long()
    else:
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids


# ==============================================================================
# 🌟 [신규 추가] Direct Regression Head (직접 좌표 회귀 헤드)
# ==============================================================================
class DirectRegressionHead(nn.Module):
    """
    [NotebookLM을 위한 모듈 요약]
    이 모듈은 기존의 '히트맵 기반 Soft-argmax' 방식을 완전히 대체하기 위해 탄생한 "직접 좌표 회귀 헤드"입니다.
    
    - 입력: 백본(DeepPA_semseg)이 뱉어낸 조밀한 피처 (B, Landmark_Num, N) + 원본 3D 좌표 (B, 3, N)
    - 과정: 점마다 부여된 특징과 공간 좌표를 결합한 뒤, PointNet 스타일의 1D-Conv와 Global Max Pooling을 
           거쳐 공간의 모든 정보를 하나의 글로벌 벡터(1024차원)로 압축합니다.
    - 출력: 거대한 MLP 네트워크를 통과하여 최종적으로 깔끔한 [Batch, Landmark_Num, 3] 형태의 (X, Y, Z) 좌표를 뱉어냅니다.
    """
    def __init__(self, in_channels, landmark_num):
        super(DirectRegressionHead, self).__init__()
        # 특징 압축 (Feature Extraction)
        self.conv1 = nn.Conv1d(in_channels, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 512, 1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(512, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # 글로벌 특징으로부터 3D 좌표 예측 (MLP Regressor)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # 최종 출력: 랜드마크 개수(36) * 3(XYZ)
            nn.Linear(256, landmark_num * 3) 
        )

    def forward(self, features, xyz):
        # features: (B, 36, N) | xyz: (B, 3, N)
        # 1. 랜드마크별 특징과 실제 3D 물리 공간의 좌표를 결합하여 위치 감각 부여
        x = torch.cat([features, xyz], dim=1) # (B, 39, N)
        
        # 2. 1D Convolution 수행
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 3. Global Max Pooling: N개의 점들 중 가장 강한 특징만 뽑아서 전체 3D 형태를 1차원 벡터로 요약
        x = torch.max(x, 2, keepdim=False)[0] # (B, 1024)
        
        # 4. 다층 퍼셉트론(MLP)을 거쳐 3D 좌표 맵핑
        coords = self.mlp(x) # (B, 108)
        
        # 5. [B, 36, 3] 형태로 예쁘게 잘라서 반환
        return coords.view(-1, coords.size(1) // 3, 3)


# ==============================================================================
# DeepPA 래퍼 클래스
# ==============================================================================
class DeepPA_Wrapper(nn.Module):
    def __init__(self, args, landmark_num=None):
        super(DeepPA_Wrapper, self).__init__()
        
        class Config: pass
        self.dl_args = dl_args = Config()
        dl_args.num_classes = landmark_num if landmark_num is not None else args.landmark_num   
        num_points = args.num_points 
        
        dl_args.bn_momentum = 0.1
        dl_args.act = nn.GELU     
        dl_args.head_dim = 256    
        dl_args.mlp_ratio = 1.0               
        dl_args.depths = [20, 20, 60, 20] 
        
        total_depth = sum(dl_args.depths)
        dl_args.use_cp = True if total_depth >= 120 else False
            
        dl_args.dims = [64, 128, 256, 512]   
        dl_args.ks = [24, 24, 24, 24]         
        dl_args.nbr_dims = [64, 128, 256, 512] 
        dl_args.cor_std = [1.0, 1.0, 1.0, 1.0] 
        dl_args.head_drops = [0.1, 0.1, 0.1, 0.1] 
        
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dl_args.drop_paths = []
        cur = 0
        for d in dl_args.depths:
            dl_args.drop_paths.append(dpr[cur:cur + d])
            cur += d
            
        dl_args.ns = [num_points, num_points // 4, num_points // 16, num_points // 64]
        dl_args.in_channels = 7 
        
        self.k = dl_args.ks[0]
        self.stage_count = len(dl_args.depths)
        
        # 1. 뼈대(Backbone) 네트워크 생성
        self.model = DeepPA_semseg(dl_args)
        
        # ---------------------------------------------------------------------
        # 🌟 [신규 추가] Direct Regression 활성화 확인 및 헤드 부착
        # ---------------------------------------------------------------------
        # My_args.py에서 스위치가 켜져 있으면, 직접 회귀 헤드를 모델 끝에 붙입니다.
        self.use_direct_regression = getattr(args, 'use_direct_regression', False)
        if self.use_direct_regression:
            print(">>> [INFO] 🚀 Direct Regression Head 활성화: 모델이 3D 좌표를 직접 출력합니다!")
            # 입력 차원: 백본 출력(클래스 수) + 원본 3D 좌표(3차원)
            in_channels_head = dl_args.num_classes + 3
            self.regression_head = DirectRegressionHead(in_channels=in_channels_head, landmark_num=dl_args.num_classes)

    def forward(self, x, prior_heatmap=None):
        B, C, N = x.shape
        
        xyz = x[:, :3, :].permute(0, 2, 1).contiguous().detach()
        
        feature = x.permute(0, 2, 1).contiguous() 
        if self.training:
            feature.requires_grad_(True)  
            
        if prior_heatmap is not None:
            prior_heatmap = prior_heatmap.permute(0, 2, 1).contiguous()
            if self.training:
                prior_heatmap.requires_grad_(True) 
            
        device = x.device
        up_idx_list = []
        down_knn_list = []
        cur_xyz = xyz
        
        # 계층적 인덱스 사전 계산
        for d in range(self.stage_count):
            num_points = cur_xyz.shape[1]
            safe_k = min(self.dl_args.ks[d], num_points)
            
            if safe_k < 1:
                idx = torch.zeros((B, num_points, self.dl_args.ks[d]), device=device, dtype=torch.long)
            else:
                dist = torch.cdist(cur_xyz, cur_xyz)
                idx = torch.topk(dist, safe_k, dim=-1, largest=False)[1]
                if safe_k < self.dl_args.ks[d]: 
                    idx = torch.cat([idx, idx[:, :, -1:].expand(-1, -1, self.dl_args.ks[d] - safe_k)], dim=-1)
                    
            down_knn_list.append(idx)
            
            if d < self.stage_count - 1:
                next_points = self.dl_args.ns[d+1]
                down_idx = farthest_point_sample(cur_xyz, next_points)
                down_knn_list.append(down_idx)
                
                batch_indices = torch.arange(B, dtype=torch.long, device=device).view(-1, 1).repeat(1, next_points)
                next_xyz = cur_xyz[batch_indices, down_idx, :]
                
                dist_up = torch.cdist(cur_xyz, next_xyz)
                up_idx = torch.argmin(dist_up, dim=2)
                up_idx_list.append(up_idx)
                
                cur_xyz = next_xyz
                
        down_knn_list = down_knn_list[::-1]
        indices = up_idx_list + down_knn_list
        
        # ---------------------------------------------------------------------
        # 🌟 모델 포워딩 및 분기 처리 (Branching)
        # ---------------------------------------------------------------------
        # 백본 통과 (출력: B, N, 36)
        out = self.model(xyz, feature, indices, prior_heatmap=prior_heatmap)
        if isinstance(out, tuple):
            out = out[0]
            
        # (B, 36, N) 형태로 변환 (채널을 중간으로)
        dense_features = out.permute(0, 2, 1).contiguous()
        
        # [분기 1] Direct Regression 스위치가 켜진 경우 -> 헤드 통과 후 (B, 36, 3) 좌표 반환
        if self.use_direct_regression:
            xyz_input = xyz.permute(0, 2, 1).contiguous() # (B, 3, N)
            pred_coords = self.regression_head(dense_features, xyz_input)
            return pred_coords
            
        # [분기 2] 옛날 방식(Ablation)인 경우 -> 기존처럼 (B, 36, N) 히트맵 반환
        else:
            return dense_features