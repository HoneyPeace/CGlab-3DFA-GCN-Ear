import torch
import torch.nn as nn
from deepla_semseg import DeepLA_semseg

class DeepLA_Wrapper(nn.Module):
    def __init__(self, args, landmark_num=44):
        super(DeepLA_Wrapper, self).__init__()
        
        class Config: pass
        self.dl_args = dl_args = Config()
        
        dl_args.bn_momentum = 0.1
        dl_args.act = nn.GELU     
        dl_args.head_dim = 256    
        dl_args.num_classes = landmark_num 
        dl_args.mlp_ratio = 1.0               
        dl_args.use_cp = False                
        
        # DeepLA-24 논문 공식 계층 구조 (U-Net)
        dl_args.depths = [4, 4, 12, 4]         
        dl_args.dims = [64, 128, 256, 512]   
        dl_args.ks = [24, 24, 24, 24]         
        dl_args.nbr_dims = [64, 128, 256, 512] 
        dl_args.cor_std = [1.0, 1.0, 1.0, 1.0] 
        dl_args.head_drops = [0.1, 0.1, 0.1, 0.1] 
        
        drop_path_rate = 0.1
        total_depth = sum(dl_args.depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dl_args.drop_paths = []
        cur = 0
        for d in dl_args.depths:
            dl_args.drop_paths.append(dpr[cur:cur + d])
            cur += d
            
        num_points = getattr(args, 'num_points', 4096)
        dl_args.ns = [num_points, num_points // 4, num_points // 16, num_points // 64]
        
        self.k = dl_args.ks[0]
        self.stage_count = len(dl_args.depths)
        self.model = DeepLA_semseg(dl_args)

    def forward(self, x):
        B, C, N = x.shape
        xyz = x.permute(0, 2, 1).contiguous()
        feature = xyz  
        device = x.device
        
        up_idx_list = []
        down_knn_list = []
        cur_xyz = xyz
        
        # [FIX] 외부 KNN 함수 오류를 원천 차단하기 위해 PyTorch 내장 함수 사용
        for d in range(self.stage_count):
            num_points = cur_xyz.shape[1]
            safe_k = min(self.dl_args.ks[d], num_points)
            
            if safe_k < 1:
                idx = torch.zeros((B, num_points, self.dl_args.ks[d]), device=device, dtype=torch.long)
            else:
                dist = torch.cdist(cur_xyz, cur_xyz)
                # 가장 가까운 이웃의 인덱스만 정확히 추출 (16억 개 텐서 버그 해결)
                idx = torch.topk(dist, safe_k, dim=-1, largest=False)[1]
                if safe_k < self.dl_args.ks[d]:
                    idx = torch.cat([idx, idx[:, :, -1:].expand(-1, -1, self.dl_args.ks[d] - safe_k)], dim=-1)
                    
            down_knn_list.append(idx)
            
            if d < self.stage_count - 1:
                next_points = self.dl_args.ns[d+1]
                down_idx = torch.arange(next_points, device=device).unsqueeze(0).expand(B, -1)
                down_knn_list.append(down_idx)
                
                batch_indices = torch.arange(B, dtype=torch.long, device=device).view(-1, 1).repeat(1, next_points)
                next_xyz = cur_xyz[batch_indices, down_idx, :]
                
                dist_up = torch.cdist(cur_xyz, next_xyz)
                up_idx = torch.argmin(dist_up, dim=2)
                up_idx_list.append(up_idx)
                
                cur_xyz = next_xyz
                
        down_knn_list = down_knn_list[::-1]
        indices = up_idx_list + down_knn_list
        
        out = self.model(xyz, feature, indices)
        if isinstance(out, tuple):
            out = out[0]
            
        return out.permute(0, 2, 1).contiguous()