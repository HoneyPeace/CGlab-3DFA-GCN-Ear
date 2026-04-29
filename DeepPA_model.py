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

from deeppa_semseg import DeepPA_semseg, index_points

try:
    from paconv import PAConv
except ImportError:
    pass 

# ==============================================================================
# 🔥 초고속 C++ 커널 로드 
# ==============================================================================
try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cpp
    USE_CPP_FPS = True
    print(">>> [INFO] DeepPA 모드: 초고속 C++ FPS 커널을 사용합니다.")
except ImportError:
    USE_CPP_FPS = False

def farthest_point_sample(xyz, npoint):
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
# 👑 최상위 마스터 융합 모델 (Raw Injection 대응 완결본)
# ==============================================================================
class DeepPA_Wrapper(nn.Module):
    def __init__(self, args, landmark_num): 
        super().__init__()
        self.args = args
        self.landmark_num = landmark_num
        
        if not hasattr(args, 'use_cp'): args.use_cp = False
            
        if hasattr(args, 'depths'):
            total_depth = sum(args.depths)
            drop_path_rate = getattr(args, 'drop_path_rate', 0.1)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
            args.drop_paths = []
            cur = 0
            for d in args.depths:
                args.drop_paths.append(dpr[cur:cur + d])
                cur += d
                
            if not hasattr(args, 'head_drops'): args.head_drops = [0.1, 0.1, 0.1, 0.1]
            if not hasattr(args, 'cor_std'): args.cor_std = [1.0, 1.0, 1.0, 1.0]
        
        # 🌟 [뿌리 수정 1] Raw Injection을 수신할 번역기(Translator) 레이어 준비
        # 교사가 1344 날것을 던지면, 학생이 알아서 4단계(64,128,256,512)로 깎아냅니다.
        self.use_raw_injection = getattr(args, 'use_raw_injection', False)
        if self.use_raw_injection:
            self.raw_proj_1 = nn.Conv1d(1344, 64, kernel_size=1, bias=False)
            self.raw_proj_2 = nn.Conv1d(1344, 128, kernel_size=1, bias=False)
            self.raw_proj_3 = nn.Conv1d(1344, 256, kernel_size=1, bias=False)
            self.raw_proj_4 = nn.Conv1d(1344, 512, kernel_size=1, bias=False)
                
        # 🟡 DeepPA 백본
        self.model = DeepPA_semseg(args)
        
        # 🔴 점진적 압축 회귀 헤드
        bn_mom = getattr(args, 'bn_momentum', 0.1)
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

    def forward(self, x, prior_hints=None):
        B, C_in, N_in = x.shape
        device = x.device
        
        # 🌟 [뿌리 수정 2] 단일 텐서(1344)가 들어오면 학생이 자체적으로 리스트로 쪼갭니다.
        # 이렇게 하면 기존 백본(DeepPA_semseg) 코드를 한 줄도 고치지 않아도 완벽 호환됩니다.
        if self.use_raw_injection and prior_hints is not None and isinstance(prior_hints, torch.Tensor):
            # prior_hints는 (B, 1344, N) 상태입니다.
            h1 = self.raw_proj_1(prior_hints)
            h2 = self.raw_proj_2(prior_hints)
            h3 = self.raw_proj_3(prior_hints)
            h4 = self.raw_proj_4(prior_hints)
            prior_hints = [h1, h2, h3, h4]

        # 1. 공간 좌표 분리 및 인덱스 세트 구성
        xyz_coords = x[:, :3, :].permute(0, 2, 1).contiguous() 
        cur_xyz = xyz_coords
        down_knn_list, up_idx_list = [], []
        
        for i in range(len(self.args.depths)):
            dist = torch.cdist(cur_xyz, cur_xyz)
            knn_idx = torch.topk(dist, self.args.ks[i], dim=-1, largest=False)[1]
            down_knn_list.append(knn_idx)
            
            if i < len(self.args.depths) - 1:
                next_points = self.args.npoints[i]
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
        
        # 2. 백본 통과
        in_features = x.permute(0, 2, 1).contiguous()
        
        # 🌟 prior_hints는 이제 완벽히 [64, 128, 256, 512] 리스트 형태이므로 무사 통과합니다.
        out = self.model(xyz=xyz_coords, x=in_features, indices=indices, prior_hints=prior_hints)
        
        dense_features = out[0] if isinstance(out, tuple) else out
        if dense_features.shape[-1] == 256: dense_features = dense_features.permute(0, 2, 1).contiguous() 

        # 2,048점을 8,192점으로 복원
        if dense_features.shape[2] != N_in:
            dense_features = index_points(dense_features.permute(0, 2, 1), up_idx_list[0]).permute(0, 2, 1)

        # 3. 최종 결합
        xyz_full = xyz_coords.permute(0, 2, 1).contiguous() 
        fused_features = torch.cat([dense_features, xyz_full], dim=1) 
        
        # 4. 회귀 헤드 통과
        x_fused = self.head_conv(fused_features) 
        x_pool = torch.max(x_fused, dim=2)[0]    
        coords = self.head_linear(x_pool).view(B, self.landmark_num, 3) 
        
        if self.training:
            spa_loss = out[1] if isinstance(out, tuple) else torch.tensor(0.0).to(device)
            sem_list = out[2] if isinstance(out, tuple) else []
            return coords, spa_loss, sem_list
            
        return coords