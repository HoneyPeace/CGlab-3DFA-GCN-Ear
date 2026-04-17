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
# 🌟 [디펜스 포인트 8, 9] Direct Regression Head (히트맵 앵커 적용판)
# ==============================================================================
class DirectRegressionHead(nn.Module):
    """
    - 입력: [256ch (백본) + 3ch (XYZ) + 36ch (PAConv 히트맵)] = 총 295ch
    - 구조: 295ch의 융합된 공간 정보를 1024ch로 넓게 확장(Expansion)하여 
            전역 맥락(Global Context)을 파악한 뒤, 
            MLP를 거쳐 (1024 -> 512 -> 256 -> 좌표) 점진적으로 압축합니다.
    """
    def __init__(self, in_channels, landmark_num):
        super(DirectRegressionHead, self).__init__()
        
        # 1. 공간 특징 점진적 확장 (Point-wise Convolution)
        self.conv1 = nn.Conv1d(in_channels, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 1024, 1)
        self.bn2 = nn.BatchNorm1d(1024)

        # 2. 글로벌 맥락 압축 회귀 (MLP Regressor)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            # 최종 출력: 랜드마크 개수 * 3(XYZ)
            nn.Linear(256, landmark_num * 3) 
        )

    def forward(self, features, xyz, prior_heatmap):
        # features: (B, 256, N) | xyz: (B, 3, N) | prior_heatmap: (B, 36, N)
        
        # 🌟 히트맵 힌트 결합 (그래디언트 역전파 차단)
        prior_heatmap = prior_heatmap.detach()
        
        # 295차원(256+3+36)으로 융합
        x = torch.cat([features, xyz, prior_heatmap], dim=1) 
        
        # 고차원 1024ch 확장
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        
        # Global Max Pooling: 3D 형태를 1차원 글로벌 맥락으로 요약 -> (B, 1024)
        x = torch.max(x, 2, keepdim=False)[0] 
        
        # 3D 좌표 회귀
        coords = self.mlp(x) 
        
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
        # 🌟 헤드 결합부
        # ---------------------------------------------------------------------
        print(">>> [INFO] 🚀 모델 아키텍처: 256ch 백본 + 36ch HM 가이드 다이렉트 회귀")
        
        # 입력 차원: 256(백본) + 3(XYZ) + 랜드마크 수(PAConv 히트맵 채널)
        in_channels_head = dl_args.head_dim + 3 + dl_args.num_classes
        
        self.regression_head = DirectRegressionHead(in_channels=in_channels_head, landmark_num=dl_args.num_classes)

    def forward(self, x, prior_latent=None, prior_heatmap=None):
        B, C, N = x.shape
        device = x.device
        
        xyz = x[:, :3, :].permute(0, 2, 1).contiguous().detach()
        feature = x.permute(0, 2, 1).contiguous() 
        if self.training:
            feature.requires_grad_(True)  
            
        # 1. 라텐트 피처(128ch) 정렬
        if prior_latent is not None:
            prior_latent = prior_latent.permute(0, 2, 1).contiguous()
            if self.training:
                prior_latent.requires_grad_(True) 
                
        # 2. 헤드용 히트맵(36ch) 정렬
        if prior_heatmap is None:
            # 단독 학습 시 오류 방지용 Dummy 히트맵
            prior_heatmap = torch.zeros((B, self.dl_args.num_classes, N), device=device)
        else:
            # 🌟 [수정 포인트]: permute 제거하고 연속성만 보장!
            prior_heatmap = prior_heatmap.contiguous()
            
        up_idx_list = []
        down_knn_list = []
        cur_xyz = xyz
        
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
        # 🌟 모델 포워딩
        # ---------------------------------------------------------------------
        # 1. 백본 통과 (HDS 모의고사 반환 포함)
        out = self.model(xyz, feature, indices, prior_heatmap=prior_latent)
        
        if isinstance(out, tuple):
            dense_features, spa_loss, sem_list = out[0], out[1], out[2]
        else:
            dense_features, spa_loss, sem_list = out, torch.tensor(0.0).to(device), []
            
        dense_features = dense_features.permute(0, 2, 1).contiguous() # (B, 256, N)
        xyz_input = xyz.permute(0, 2, 1).contiguous() # (B, 3, N)
        
        # 2. [최종단 결합] 다이렉트 헤드를 통과하여 좌표 산출
        pred_coords = self.regression_head(dense_features, xyz_input, prior_heatmap)
        
        return pred_coords, spa_loss, sem_list, indices