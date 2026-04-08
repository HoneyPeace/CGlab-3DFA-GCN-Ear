import torch
import torch.nn as nn
from deeppa_semseg import DeepPA_semseg

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "utils" / "pointnet2_ops_lib"))

# 🔥 다시 초고속 C++ 커널을 켭니다!
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
        if total_depth >= 120:
            dl_args.use_cp = True
        else:
            dl_args.use_cp = False
            
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
        
        # 🔥 14채널 엣지 구성을 위한 7채널(XYZ 3 + 주방향 3 + 곡률 1) 입력 고정
        dl_args.in_channels = 7 
        
        self.k = dl_args.ks[0]
        self.stage_count = len(dl_args.depths)
        
        self.model = DeepPA_semseg(dl_args)

    def forward(self, x, prior_heatmap=None):
        B, C, N = x.shape
        
        # 🌟 [해결책 1. VRAM 및 연산 폭발 방지] 
        # 거리 계산용 xyz는 미분 계산을 아예 안 하도록 강제로 끊어버립니다. (속도/메모리 최적화)
        # KNN 및 FPS에는 순수물리좌표인 앞의 3채널만 사용됩니다.
        xyz = x[:, :3, :].permute(0, 2, 1).contiguous().detach()
        
        # 🌟 [해결책 2. 경고 제거 및 120층 진짜 학습 활성화!]
        # 모델 내부로 들어가는 피처에는 7채널 전체를 넘겨줍니다.
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
        
        out = self.model(xyz, feature, indices, prior_heatmap=prior_heatmap)
        
        if isinstance(out, tuple):
            out = out[0]
            
        return out.permute(0, 2, 1).contiguous()