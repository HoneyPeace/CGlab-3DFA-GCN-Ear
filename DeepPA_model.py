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
# 👑 최상위 마스터 융합 모델 (HybridPipeline 호환 버전)
# ==============================================================================
class DeepPA_Wrapper(nn.Module):
    """
    [0.47mm 달성을 위한 2단계 정밀 타격 모델]
    train.py의 HybridPipeline에서 건네주는 라텐트 힌트와 날것의 히트맵을 받아 
    최종 3D 좌표를 Direct Regression으로 출력합니다.
    """
    def __init__(self, args, landmark_num): # 🌟 landmark_num 파라미터 연동
        super().__init__()
        self.args = args
        self.landmark_num = landmark_num
        
        # 🌟 VRAM 최적화: PAConv는 train.py에서 관리하므로 내장하지 않음
        
        # ---------------------------------------------------------------------
        # 🟡 [Phase 2] DeepPA (120층 백본 - 기하학적 스나이퍼)
        # ---------------------------------------------------------------------
        self.model = DeepPA_semseg(args)
        
        # ---------------------------------------------------------------------
        # 🔴 [Phase 3] 점진적 압축 회귀 헤드 (Direct Regression Head)
        # ---------------------------------------------------------------------
        bn_mom = getattr(args, 'bn_momentum', 0.1)
        
        # 1. 고밀도 기하 피처 추출용 1D Conv (공간적 특징 학습)
        self.head_conv = nn.Sequential(
            nn.Conv1d(295, 1024, 1),
            nn.BatchNorm1d(1024, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        # 2. 전역적 맥락(Global Context) 확보 후 최종 좌표 출력 (MLP)
        self.head_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.landmark_num * 3) # 🌟 하드코딩(108) 방지, 동적 확장
        )

    # 🌟 통제실(train.py)에서 던져주는 힌트 파라미터를 정확히 수신
    def forward(self, x, prior_latent=None, prior_heatmap=None):
        # x: (B, C_in, N) 규격
        B, C_in, N_in = x.shape
        device = x.device
        
        # =====================================================================
        # 1. 계층적 다운샘플링 인덱스 추출 (FPS용 차원 변경)
        # =====================================================================
        # FPS 커널은 공간 좌표만을 필요로 하며 (B, N, 3) 형태를 요구함
        xyz_coords = x[:, :3, :].permute(0, 2, 1).contiguous() 
        
        cur_xyz = xyz_coords
        down_knn_list = []
        up_idx_list = []
        
        for i in range(len(self.args.depths)):
            if hasattr(self.args, 'npoints'):
                next_points = self.args.npoints[i]
            else:
                next_points = N_in // (4 ** (i + 1))
                
            down_idx = farthest_point_sample(cur_xyz, next_points)
            
            batch_indices = torch.arange(B, dtype=torch.long, device=device).view(-1, 1).repeat(1, next_points)
            next_xyz = cur_xyz[batch_indices, down_idx, :]
            down_knn_list.append(down_idx)
            
            dist_up = torch.cdist(cur_xyz, next_xyz)
            up_idx = torch.argmin(dist_up, dim=2)
            up_idx_list.append(up_idx)
            
            cur_xyz = next_xyz
            
        down_knn_list = down_knn_list[::-1]
        indices = up_idx_list + down_knn_list
        
        # =====================================================================
        # 2. DeepPA 백본 통과 (1단계 라텐트 힌트 병합)
        # =====================================================================
        out = self.model(x, feature=None, indices=indices, prior_heatmap=prior_latent)
        
        if isinstance(out, tuple):
            dense_features, spa_loss, sem_list = out[0], out[1], out[2]
        else:
            dense_features, spa_loss, sem_list = out, torch.tensor(0.0).to(device), []
            
        if dense_features.shape[-1] == 256:
            dense_features = dense_features.permute(0, 2, 1).contiguous() 
        
        # =====================================================================
        # 3. 최종 앵커 결합 (Ground Truth Alignment) - Concatenation
        # =====================================================================
        last_idx = indices[-1] 
        xyz_down = index_points(xyz_coords, last_idx).permute(0, 2, 1).contiguous() 
        
        if prior_heatmap is not None:
            # 1단계의 날것(Raw) 히트맵을 현재 해상도에 맞게 필터링
            prior_heatmap_trans = prior_heatmap.permute(0, 2, 1).contiguous() 
            paconv_heatmap = index_points(prior_heatmap_trans, last_idx).permute(0, 2, 1).detach() 
        else:
            paconv_heatmap = torch.zeros(B, self.landmark_num, xyz_down.shape[2], device=device)
        
        # [백본(256) + 원본좌표(3) + 앵커히트맵(36)] = 총 295채널 결합!
        fused_features = torch.cat([dense_features, xyz_down, paconv_heatmap], dim=1) 
        
        # =====================================================================
        # 4. 점진적 압축 회귀 헤드 통과
        # =====================================================================
        x_fused = self.head_conv(fused_features) 
        x_pool = torch.max(x_fused, dim=2)[0]    
        
        coords = self.head_linear(x_pool)        
        coords = coords.view(B, self.landmark_num, 3) 
        
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