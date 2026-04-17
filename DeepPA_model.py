import torch
import torch.nn as nn
import torch.nn.functional as F
from deeppa_semseg import DeepPA_semseg

import sys
from pathlib import Path

# ==============================================================================
# 🌟 [NotebookLM 경로 설정 및 최적화]
# ==============================================================================
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
    - 모듈명: Direct Regression Head (256ch 원본 직결형)
    - 논문 내 역할: 히트맵 기반의 격자 해상도 한계를 극복하기 위한 '해상도 독립적(Resolution-free)' 회귀 엔진입니다.
    - 핵심 구조: 
      백본(DeepPA_semseg)이 120층을 거쳐 복구한 256차원의 기하학적 라텐트 피처를 압축 없이 그대로 수용합니다.
      여기에 실제 3D 물리 좌표(XYZ)를 연결(Concat)하여 공간 감각을 극대화한 뒤, 
      PointNet 스타일의 1D-Conv와 Global Max Pooling을 거쳐 공간의 모든 정보를 
      하나의 글로벌 벡터(1024차원)로 압축합니다.
    - 출력: 거대한 MLP를 통과하여 [Batch, Landmark_Num, 3] 형태의 3D 물리 좌표를 다이렉트로 방출합니다.
    """
    def __init__(self, in_channels, landmark_num):
        super(DirectRegressionHead, self).__init__()
        # 특징 압축 및 확장 (259 -> 1024차원)
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
        # features: (B, 256, N) | xyz: (B, 3, N)
        # 1. 256차원 특징과 실제 3D 물리 공간의 좌표를 결합하여 위치 감각 부여
        x = torch.cat([features, xyz], dim=1) # (B, 259, N)
        
        # 2. 고차원 사영 수행
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 3. Global Max Pooling: 3D 형태를 1차원 글로벌 맥락으로 요약
        x = torch.max(x, 2, keepdim=False)[0] # (B, 1024)
        
        # 4. 3D 좌표 회귀
        coords = self.mlp(x) # (B, 108)
        
        # 5. [B, 36, 3] 형태로 반환
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
        # 백본의 출력 차원을 256으로 명시
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
        
        # 1. 뼈대(Backbone) 네트워크 생성 (DeepPA_semseg)
        self.model = DeepPA_semseg(dl_args)
        
        # ---------------------------------------------------------------------
        # 🌟 [핵심] 다이렉트 좌표 회귀 모델 강제 세팅 및 보조 로스 헤드 부착
        # ---------------------------------------------------------------------
        print(">>> [INFO] 🚀 모델 아키텍처: 256ch Raw Latent Direct Regression (히트맵 우회 모드)")
        
        # 입력 차원: 백본 출력(256차원) + 원본 3D 좌표(3차원)
        in_channels_head = dl_args.head_dim + 3
        
        # 메인 회귀 경로
        self.regression_head = DirectRegressionHead(in_channels=in_channels_head, landmark_num=dl_args.num_classes)
        
        # 학습용 보조 닻(Auxiliary Anchor) 경로
        # 256채널 라텐트가 공간적 위치 감각을 잃지 않도록 감독하는 역할입니다.
        #self.aux_heatmap_head = nn.Conv1d(dl_args.head_dim, dl_args.num_classes, 1)

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
        # 🌟 모델 포워딩 및 Latent Direct Regression
        # ---------------------------------------------------------------------
        # 백본 통과 (출력은 더 이상 36이 아닌, 256차원 고차원 텐서입니다)
        out = self.model(xyz, feature, indices, prior_heatmap=prior_heatmap)
        if isinstance(out, tuple):
            out = out[0]
            
        # 형태 변환: (B, 256, N)
        dense_features = out.permute(0, 2, 1).contiguous()
        
        # 물리 좌표 정렬: (B, 3, N)
        xyz_input = xyz.permute(0, 2, 1).contiguous() 
        
        # [메인 출력] 다이렉트 헤드를 통과하여 서브 밀리미터 단위 좌표 산출
        pred_coords = self.regression_head(dense_features, xyz_input)
        
        # [보조 출력] 학습 시, 라텐트 피처의 위치 정렬을 돕기 위해 보조 히트맵 산출
        #aux_heatmap = self.aux_heatmap_head(dense_features)
        
        # 항상 (좌표, 보조 히트맵)의 튜플 형태로 일관성 있게 반환합니다.
        #return pred_coords, aux_heatmap
        return pred_coords, None