import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

# ==============================================================================
# 🌟 [NotebookLM 경로 설정 및 모듈 임포트]
# ==============================================================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "utils" / "pointnet2_ops_lib"))

# deeppa_semseg.py 내부에 정의된 함수를 직접 가져오도록 수정합니다.
from deeppa_semseg import DeepPA_semseg, index_points

# [주의]: PAConv 원본 모델의 임포트 경로는 연구자님의 환경에 맞게 수정해주세요.
try:
    from paconv import PAConv
except ImportError:
    pass # 실제 환경에서 import 확인 필요

# ==============================================================================
# 🔥 초고속 C++ 커널 로드 (원본 유지)
# ==============================================================================
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
# 👑 최상위 마스터 융합 모델 (Latent Gating Only 버전)
# ==============================================================================
class DeepPA_Wrapper(nn.Module):
    def __init__(self, args, landmark_num): 
        super().__init__()
        self.args = args
        self.landmark_num = landmark_num
        
        # 🌟 [에러 방어 1] args에 use_cp 없으면 강제 할당
        if not hasattr(args, 'use_cp'):
            args.use_cp = False
            
        # 🌟 [에러 방어 2] DeepLA 구버전에서 훔쳐온 완벽한 Stochastic Depth 로직
        if hasattr(args, 'depths'):
            total_depth = sum(args.depths)
            drop_path_rate = getattr(args, 'drop_path_rate', 0.1)
            
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
            
            args.drop_paths = []
            cur = 0
            for d in args.depths:
                args.drop_paths.append(dpr[cur:cur + d])
                cur += d
                
            # 🔥 [에러 방어 3: 시한폭탄 영구 제거!]
            if not hasattr(args, 'head_drops'):
                args.head_drops = [0.1, 0.1, 0.1, 0.1]
            if not hasattr(args, 'cor_std'):
                args.cor_std = [1.0, 1.0, 1.0, 1.0]
                
        # 🟡 [Phase 2] DeepPA (120층 백본 - 기하학적 스나이퍼)
        self.model = DeepPA_semseg(args)
        
        # 🔴 [Phase 3] 점진적 압축 회귀 헤드 (Direct Regression Head)
        bn_mom = getattr(args, 'bn_momentum', 0.1)
        
        # 🌟 [핵심 수정 1] 295 -> 259 채널로 다이어트 (가이드 히트맵 36ch 제거)
        self.head_conv = nn.Sequential(
            nn.Conv1d(259, 1024, 1),
            nn.BatchNorm1d(1024, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        self.head_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.landmark_num * 3) 
        )

    # 🌟 [핵심 수정 2] prior_heatmap 인자 제거, 오직 prior_latent(128ch 힌트)만 받음
    def forward(self, x, prior_latent=None):
        B, C_in, N_in = x.shape
        device = x.device
        
        # 1. 공간 좌표 분리 및 인덱스 세트 구성
        xyz_coords = x[:, :3, :].permute(0, 2, 1).contiguous() 
        cur_xyz = xyz_coords
        down_knn_list, up_idx_list = [], []
        
        for i in range(len(self.args.depths)):
            # KNN 계산 (이웃 정보)
            dist = torch.cdist(cur_xyz, cur_xyz)
            knn_idx = torch.topk(dist, self.args.ks[i], dim=-1, largest=False)[1]
            down_knn_list.append(knn_idx)
            
            if i < len(self.args.depths) - 1:
                next_points = self.args.npoints[i]
                down_idx = farthest_point_sample(cur_xyz, next_points)
                down_knn_list.append(down_idx) # FPS 저장
                
                batch_indices = torch.arange(B, dtype=torch.long, device=device).view(-1, 1).repeat(1, next_points)
                next_xyz = cur_xyz[batch_indices, down_idx, :]
                
                # 🌟 업샘플링(보간) 인덱스: 상위 스테이지로 정보를 다시 올릴 때 사용
                dist_up = torch.cdist(cur_xyz, next_xyz)
                up_idx = torch.argmin(dist_up, dim=2)
                up_idx_list.append(up_idx)
                cur_xyz = next_xyz
            
        down_knn_list = down_knn_list[::-1]
        indices = up_idx_list + down_knn_list
        
        # 2. 백본 통과 (출력: 2,048점의 특징맵)
        in_features = x.permute(0, 2, 1).contiguous()
        # 🌟 라텐트 힌트 주입 (백본 내부에서 Gate 융합 처리됨)
        out = self.model(xyz=xyz_coords, x=in_features, indices=indices, prior_heatmap=prior_latent)
        
        dense_features = out[0] if isinstance(out, tuple) else out
        if dense_features.shape[-1] == 256: dense_features = dense_features.permute(0, 2, 1).contiguous() 

        # =====================================================================
        # 🌟 2,048점을 8,192점으로 복원 (Upsampling)
        # =====================================================================
        if dense_features.shape[2] != N_in:
            dense_features = index_points(dense_features.permute(0, 2, 1), up_idx_list[0]).permute(0, 2, 1)

        # =====================================================================
        # 3. 최종 결합 (Full 8,192 Resolution)
        # =====================================================================
        xyz_full = xyz_coords.permute(0, 2, 1).contiguous() 
        
        # 🌟 [핵심 수정 3] 불필요한 히트맵/어텐션 로직 싹 제거
        # 대통합 콘캣: [백본(256) + 좌표(3)] = 총 259채널 (B, 259, 8192)
        fused_features = torch.cat([dense_features, xyz_full], dim=1) 
        
        # 4. 회귀 헤드 통과
        x_fused = self.head_conv(fused_features) 
        x_pool = torch.max(x_fused, dim=2)[0]    
        coords = self.head_linear(x_pool).view(B, self.landmark_num, 3) 
        
        if self.training:
            # 백본에서 나온 spa_loss, sem_list가 있다면 함께 리턴
            spa_loss = out[1] if isinstance(out, tuple) else torch.tensor(0.0).to(device)
            sem_list = out[2] if isinstance(out, tuple) else []
            return coords, spa_loss, sem_list
        return coords

if __name__ == '__main__':
    # 간단한 작동 테스트
    class Args:
        bn_momentum = 0.1
        depths = [2, 2, 2] # 예시
        npoints = [2048, 512, 128] # 예시
        # use_spatial_attention 인자는 더 이상 필요 없음
    
    # args = Args()
    # model = DeepPA_Wrapper(args, 36).cuda()
    # print("DeepPA_Wrapper 모델 초기화 완료. 259ch -> 108 좌표 압축 준비됨 (가이드 히트맵 완전 제거).")