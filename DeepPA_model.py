import torch
import torch.nn as nn
from deeppa_semseg import DeepPA_semseg

import sys
from pathlib import Path
from deeppa_semseg import DeepPA_Offset_semseg
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
        
        dl_args.depths = [4, 4, 12, 4] 
        
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
        
        self.k = dl_args.ks[0]
        self.stage_count = len(dl_args.depths)
        
        # 🔥 [수정됨] 껍데기 믹서기 삭제! 오직 DeepPA_semseg 심장부만 장착!
        self.model = DeepPA_Offset_semseg(dl_args)


    def forward(self, x, prior_heatmap=None):
        B, C, N = x.shape
        
        # 3D 좌표 기본 세팅
        xyz = x.permute(0, 2, 1).contiguous()
        feature = xyz  # 원본 3D 좌표 그대로 출발
        
        # 🔥 [핵심 추가] PAConv 힌트도 DeepLA 내부 연산(B, N, C)에 맞게 형태를 돌려줌
        if prior_heatmap is not None:
            prior_heatmap = prior_heatmap.permute(0, 2, 1).contiguous()
            
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
        
        # 🔥 [수정됨] 힌트를 매번 다운샘플링하기 위해 메인 두뇌(self.model)에 직접 쏴줍니다!
        out = self.model(xyz, feature, indices, prior_heatmap=prior_heatmap)
        
        if isinstance(out, tuple):
            out = out[0]
            
        return out # (B, K, 3)