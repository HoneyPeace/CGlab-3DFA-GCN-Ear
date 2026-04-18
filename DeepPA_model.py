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
# 예시: from paconv import PAConv
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
# 👑 최상위 마스터 융합 모델
# ==============================================================================
class DeepPA_Wrapper(nn.Module):
    """
    [0.47mm 달성을 위한 최상위 마스터 융합 모델]
    Phase 1(PAConv)와 Phase 2(DeepPA)를 결합하고, 점진적 압축 회귀 헤드를 통해 3D 좌표를 출력합니다.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # ---------------------------------------------------------------------
        # 🟢 [Phase 1] PAConv (내비게이션 - 거시적 랜드마크 영역 탐지)
        # ---------------------------------------------------------------------
        self.paconv = PAConv(args) 
        
        # 🌟 [디펜스 포인트 1] 전면 프리징 (Freezing) 철벽 방어
        # 120층 백본의 수렴 진동을 막기 위해 1단계 가이드라인의 가중치를 영구 고정합니다.
        for param in self.paconv.parameters():
            param.requires_grad = False
        self.paconv.eval() 

        # ---------------------------------------------------------------------
        # 🟡 [Phase 2] DeepPA (120층 백본 - 기하학적 스나이퍼)
        # ---------------------------------------------------------------------
        self.model = DeepPA_semseg(args)
        
        # ---------------------------------------------------------------------
        # 🔴 [Phase 3] 점진적 압축 회귀 헤드 (Direct Regression Head)
        # ---------------------------------------------------------------------
        # 🌟 [디펜스 포인트 9] 295ch -> 1024ch -> 512ch -> (Global Pool) -> 256ch -> 108ch
        
        # 1. 고밀도 기하 피처 추출용 1D Conv (공간적 특징 학습)
        self.head_conv = nn.Sequential(
            nn.Conv1d(295, 1024, 1),
            nn.BatchNorm1d(1024, momentum=args.bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512, momentum=args.bn_momentum),
            nn.ReLU(inplace=True),
        )
        
        # 2. 전역적 맥락(Global Context) 확보 후 최종 좌표 출력 (MLP)
        self.head_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=args.bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 108) # 36개 랜드마크 * 3(XYZ) = 108
        )

    def forward(self, xyz, feature):
        B, N_in, C_in = xyz.shape
        device = xyz.device
        
        # =====================================================================
        # 1. PAConv 통과 (역전파 완벽 차단)
        # =====================================================================
        with torch.no_grad():
            self.paconv.eval()
            prior_latent, out_paconv = self.paconv(xyz, feature)
            # prior_latent: (B, N, 128) - DeepPA 잔차 융합용 힌트
            # out_paconv: (B, N, 36) - 최종단 앵커용 히트맵
            
        # =====================================================================
        # 2. 계층적 다운샘플링 인덱스(FPS & KNN) 추출 로직 (원본 완벽 유지)
        # =====================================================================
        cur_xyz = xyz
        down_knn_list = []
        up_idx_list = []
        
        for i in range(len(self.args.depths)):
            # 깊이에 따라 점 개수를 줄임 (기존 로직 또는 args 설정 기준)
            if hasattr(self.args, 'npoints'):
                next_points = self.args.npoints[i]
            else:
                next_points = N_in // (4 ** (i + 1))
                
            down_idx = farthest_point_sample(cur_xyz, next_points)
            
            # KNN 검색 (현재 원본 코드 로직 복원)
            batch_indices = torch.arange(B, dtype=torch.long, device=device).view(-1, 1).repeat(1, next_points)
            next_xyz = cur_xyz[batch_indices, down_idx, :]
            down_knn_list.append(down_idx)
            
            # Up-sampling 보간을 위한 거리(cdist) 기반 인덱스 추출
            dist_up = torch.cdist(cur_xyz, next_xyz)
            up_idx = torch.argmin(dist_up, dim=2)
            up_idx_list.append(up_idx)
            
            cur_xyz = next_xyz
            
        down_knn_list = down_knn_list[::-1]
        indices = up_idx_list + down_knn_list
        
        # =====================================================================
        # 3. DeepPA 백본 통과 (게이트 융합 및 HDS 채점)
        # =====================================================================
        out = self.model(xyz, feature, indices, prior_heatmap=prior_latent)
        
        if isinstance(out, tuple):
            dense_features, spa_loss, sem_list = out[0], out[1], out[2]
        else:
            dense_features, spa_loss, sem_list = out, torch.tensor(0.0).to(device), []
            
        # 백본 출력 형태 맞추기: (B, N_down, 256) -> (B, 256, N_down)
        dense_features = dense_features.permute(0, 2, 1).contiguous() 
        
        # =====================================================================
        # 4. 🌟 최종단 앵커 연결 및 차원 동기화 (Ground Truth Alignment)
        # =====================================================================
        # DeepPA를 통과하며 가장 작게 줄어든 해상도(N_down) 인덱스
        last_idx = indices[-1] 
        
        # 원본 좌표(XYZ)를 마지막 해상도에 맞게 솎아냄
        xyz_down = index_points(xyz, last_idx).permute(0, 2, 1) # (B, 3, N_down)
        
        # 🌟 [디펜스 포인트 8] 히트맵 앵커 차원 맞추기 및 역전파 차단 (.detach())
        # (B, N_in, 36) -> (B, 36, N_down)
        paconv_heatmap = index_points(out_paconv, last_idx).permute(0, 2, 1).detach() 
        
        # [백본(256) + 원본좌표(3) + 앵커히트맵(36)] = 총 295채널 결합
        fused_features = torch.cat([dense_features, xyz_down, paconv_heatmap], dim=1) # (B, 295, N_down)
        
        # =====================================================================
        # 5. 점진적 압축 회귀 헤드 통과
        # =====================================================================
        x = self.head_conv(fused_features) # (B, 295, N_down) -> (B, 512, N_down)
        
        # 전역적 맥락(Global Context) 병합 (Point-wise Max Pooling)
        x = torch.max(x, dim=2)[0]         # (B, 512, N_down) -> (B, 512)
        
        # 0.1mm 초정밀 좌표로 쥐어짜기
        coords = self.head_linear(x)       # (B, 512) -> (B, 108)
        coords = coords.view(B, 36, 3)     # (B, 36, 3) 텐서로 정리
        
        if self.training:
            return coords, spa_loss, sem_list
        return coords

if __name__ == '__main__':
    # 간단한 작동 테스트
    class Args:
        bn_momentum = 0.1
        depths = [2, 2, 2] # 예시
        npoints = [2048, 512, 128] # 예시
    
    # args = Args()
    # model = DeepPA_Auto(args).cuda()
    # print("DeepPA_Auto 모델 초기화 완료. 295ch -> 108 좌표 압축 준비됨.")