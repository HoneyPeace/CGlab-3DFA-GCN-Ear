import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path

# 경로 설정 (연구원님 환경 유지)
sys.path.append(str(Path(__file__).absolute().parent.parent))
try:
    from utils.timm.models.layers import DropPath
    from utils.cutils import knn_edge_maxpooling
except ImportError:
    pass

# =====================================================================
# 유틸리티 함수
# =====================================================================
def index_points(points, idx):
    """
    텐서(B, N, C)에서 idx(B, S)에 해당하는 점만 추출하여 (B, S, C) 반환
    """
    device = points.device
    B = points.shape[0]
    idx = idx.long() 
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def checkpoint(function, *args, **kwargs):
    try:
        return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)
    except ValueError:
        return torch_checkpoint(function, *args, **kwargs)

try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cpp
    USE_CPP_FPS = True
    print(">>> [INFO] DeepPA: 초고속 C++ FPS 커널 사용")
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

# =====================================================================
# 기본 빌딩 블록 (VFR, FFN, ResLFE)
# =====================================================================
class VFR(nn.Module):
    def __init__(self, in_dim, out_dim, bn_momentum, init=0.):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=bn_momentum)
        nn.init.constant_(self.bn.weight, init)

    def forward(self, x, knn):
        B, N, C = x.shape
        x = self.linear(x)
        x = knn_edge_maxpooling(x, knn, self.training)
        x = self.bn(x.view(-1, x.shape[-1])).view(B, N, -1)
        return x

class FFN(nn.Module):
    def __init__(self, in_dim, mlp_ratio, bn_momentum, act, init=0.):
        super().__init__()
        hid_dim = round(in_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            act(),
            nn.Linear(hid_dim, in_dim, bias=False),
            nn.BatchNorm1d(in_dim, momentum=bn_momentum),
        )
        nn.init.constant_(self.ffn[-1].weight, init)

    def forward(self, x):
        B, N, C = x.shape
        x = self.ffn(x.view(-1, C)).view(B, N, -1)
        return x

class ResLFE_Block(nn.Module):
    def __init__(self, dim, depth, drop_path, mlp_ratio, bn_momentum, act):
        super().__init__()
        self.depth = depth
        self.VFRs = nn.ModuleList([VFR(dim, dim, bn_momentum) for _ in range(depth)])
        self.mlp = FFN(dim, mlp_ratio, bn_momentum, act, 0.2)
        self.FFNs = nn.ModuleList([FFN(dim, mlp_ratio, bn_momentum, act) for _ in range(depth)])
        self.drop_paths = nn.ModuleList([DropPath(dp) for dp in drop_path])
        self.dp = [dp > 0. for dp in drop_path]

    def drop_path(self, x, i):
        if not self.dp[i] or not self.training:
            return x
        return self.drop_paths[i](x)

    def forward(self, x, pe, knn, pts=None):
        x = x + self.drop_path(self.mlp(x), 0)
        for i in range(self.depth):
            x = x + pe 
            x = x + self.drop_path(self.VFRs[i](x, knn), i)
            x = x + self.drop_path(self.FFNs[i](x), i)
        return x

# =====================================================================
# 🌟 계층적(Hierarchical) 스테이지 설계 (문제의 원인 완벽 수정)
# =====================================================================
class Stage_PA(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()
        self.depth = depth
        self.up_depth = len(args.depths) - 1
        self.first = depth == 0
        self.last = depth == self.up_depth
        self.k = args.ks[depth]
        self.cp = getattr(args, 'use_cp', False)
        cp_bn_momentum = args.cp_bn_momentum if self.cp else args.bn_momentum

        dim = args.dims[depth]
        self.use_raw_injection = getattr(args, 'use_raw_injection', False)
        
        # 🌟 [해결책 1] 들어오는 정보의 형태(1344 vs List)에 맞게 유동적 입구 설계
        in_hint_dim = 1344 if self.use_raw_injection else dim
        self.prior_proj = nn.Sequential(
            nn.Linear(in_hint_dim, dim, bias=False), 
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            args.act()
        )
        
        self.use_gate = getattr(args, 'use_gate', True)
        if self.use_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(dim * 2, dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum),
                nn.Sigmoid() 
            )

        if self.first:
            nbr_hid_dim = args.nbr_dims[0]
            in_channels = getattr(args, 'in_channels', 3)
            in_feat_dim = 14 if in_channels == 7 else (13 if in_channels == 6 else 10)
            
            self.nbr_embed = nn.Sequential(
                nn.Linear(in_feat_dim, nbr_hid_dim // 2, bias=False),  
                nn.BatchNorm1d(nbr_hid_dim // 2, momentum=cp_bn_momentum),
                args.act(),
                nn.Linear(nbr_hid_dim // 2, nbr_hid_dim, bias=False),
                nn.BatchNorm1d(nbr_hid_dim, momentum=cp_bn_momentum),
                args.act(),
                nn.Linear(nbr_hid_dim, dim, bias=False),
            )
            self.nbr_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            nn.init.constant_(self.nbr_bn.weight, 0.8)
            self.nbr_proj = nn.Identity()

        pe_hid_dim = args.nbr_dims[1] // 2
        self.pe_embed = nn.Sequential(
            nn.Linear(3, pe_hid_dim//2, bias=False),
            nn.BatchNorm1d(pe_hid_dim//2, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(pe_hid_dim//2, pe_hid_dim, bias=False),
            nn.BatchNorm1d(pe_hid_dim, momentum=cp_bn_momentum),
            args.act(),
            nn.Linear(pe_hid_dim, args.nbr_dims[1], bias=False),
        )
        self.pe_bn = nn.BatchNorm1d(dim, momentum=args.bn_momentum)
        nn.init.constant_(self.pe_bn.weight, 0.2)
        self.pe_proj = nn.Linear(args.nbr_dims[1], dim, bias=False)

        if not self.first:
            self.vfr = VFR(args.dims[depth - 1], dim, args.bn_momentum, 0.3)
            self.skip_proj = nn.Sequential(
                nn.Linear(args.dims[depth - 1], dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.reslfe = ResLFE_Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, cp_bn_momentum, args.act)
        self.drop = DropPath(args.head_drops[depth])

        self.sem_sup = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            nn.Linear(dim, getattr(args, 'num_classes', 36), bias=False), 
        )

        self.postproj = nn.Sequential(
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            nn.Linear(dim, args.head_dim, bias=False),
        )
        nn.init.constant_(self.postproj[0].weight, (args.dims[0] / dim) ** 0.5)

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        if not self.last:
            self.sub_stage = Stage_PA(args, depth + 1)

    def forward(self, x, xyz, prev_knn, indices, pts_list, prior_heatmap=None, sub_spa=None, sub_sem=None):
        B, N_in, C_in = x.shape
        
        # =====================================================================
        # 🌟 [해결책 2] 재귀적 다운샘플링의 정석 (List와 Tensor 모두 완벽 지원)
        # =====================================================================
        if not self.first:
            ids = indices.pop()
            xyz = index_points(xyz, ids)
            x_skip = index_points(self.skip_proj(x.view(-1, C_in)).view(B, N_in, -1), ids)
            x_vfr = index_points(self.vfr(x, prev_knn), ids)
            x = x_skip + x_vfr
            
            # 해상도 동기화: 부모로부터 넘어온 힌트도 현재 점 개수에 맞게 축소
            if prior_heatmap is not None:
                if isinstance(prior_heatmap, list):
                    # 리스트면 안의 텐서를 몽땅 다운샘플링
                    prior_heatmap = [index_points(h, ids) for h in prior_heatmap]
                else:
                    # 단일 텐서면 통째로 다운샘플링
                    prior_heatmap = index_points(prior_heatmap, ids)
            
        knn = indices.pop()
        B, N, C = x.shape

        xyz_knn = index_points(xyz, knn)
        pe = xyz_knn - xyz.unsqueeze(2) 

        if self.first:
            nbr_rel = pe.clone() 
            x_knn = index_points(x, knn) 
            dist = torch.norm(nbr_rel, dim=-1, keepdim=True) 
            vector = nbr_rel / (dist + 1e-8)
            
            if C_in == 7: nbr = torch.cat([nbr_rel, x_knn, dist, vector], dim=-1).view(-1, 14)
            elif C_in == 6: nbr = torch.cat([nbr_rel, x_knn, dist, vector], dim=-1).view(-1, 13) 
            else: nbr = torch.cat([nbr_rel, x_knn, dist, vector], dim=-1).view(-1, 10) 
            
            nbr_embed_func = lambda t: self.nbr_embed(t).view(B, N, self.k, -1).max(dim=2)[0]
            nbr = checkpoint(nbr_embed_func, nbr) if self.training and self.cp else nbr_embed_func(nbr)
            nbr = self.nbr_proj(nbr)
            x = self.nbr_bn(nbr.view(-1, nbr.shape[-1])).view(B, N, -1)

        # =====================================================================
        # 🌟 [해결책 3] 힌트 추출 및 투영 (List vs Raw Tensor 대응)
        # =====================================================================
        if prior_heatmap is not None:
            if isinstance(prior_heatmap, list):
                # 기존 방식: 리스트에서 자신의 깊이(Depth)에 맞는 힌트를 꺼냄
                p_feat_input = prior_heatmap[self.depth]
            else:
                # 교수님 방식(Raw): 통째로 받은 1344차원 텐서를 그대로 입력으로 씀
                p_feat_input = prior_heatmap
                
            # 자신의 채널(dim)에 맞게 1x1 Conv로 깎기
            p_feat = self.prior_proj(p_feat_input.view(-1, p_feat_input.shape[-1])).view(B, N, -1)
            
            if self.use_gate:
                fused_for_gate = torch.cat([x, p_feat], dim=-1) 
                gate_matrix = self.gate_mlp(fused_for_gate.view(-1, fused_for_gate.shape[-1])).view(B, N, -1)
                x = x + (gate_matrix * p_feat)
            else:
                x = x + p_feat

        pe = pe.view(-1, 3)
        pe_embed_func = lambda t: self.pe_embed(t).view(B, N, self.k, -1).max(dim=2)[0]
        pe = checkpoint(pe_embed_func, pe) if self.training and self.cp else pe_embed_func(pe)
        pe = self.pe_proj(pe)
        pe = self.pe_bn(pe.view(-1, pe.shape[-1])).view(B, N, -1)

        pts = pts_list.pop() if pts_list is not None else None
        x = checkpoint(self.reslfe, x, pe, knn, pts) if self.training and self.cp else self.reslfe(x, pe, knn, pts)

        if self.training:
            spa_info = xyz_knn - xyz.unsqueeze(2)
            spa_info.mul_(self.cor_std)
            feat_info = self.cor_head(x.view(-1, x.shape[-1])).view(B, N, -1)
            feat_info_knn = index_points(feat_info, knn)
            feat_info = feat_info_knn - feat_info.unsqueeze(2)
            
            closs = F.mse_loss(feat_info, spa_info) 
            sub_spa = sub_spa + closs if sub_spa is not None else closs

            sem = self.sem_sup(x.view(-1, x.shape[-1])).view(B, N, -1) 
            
            if sub_sem is not None:
                sub_sem.append(sem)
            else:
                sub_sem = [sem]

        # 🌟 통과 시, prior_heatmap도 하위 스테이지로 그대로 내려보냅니다.
        if not self.last:
            sub_x, sub_spa, sub_sem = self.sub_stage(x, xyz, knn, indices, pts_list, prior_heatmap, sub_spa, sub_sem)
        else:
            sub_x = None
            self.spa, self.sem = sub_spa, sub_sem

        x = self.postproj(x.view(-1, x.shape[-1])).view(B, N, -1)
        sub_x = sub_x + x if sub_x is not None else x 
        sub_x = self.drop(sub_x)
        
        if not self.first:
            back_nn = indices[self.depth-1]
            sub_x = index_points(sub_x, back_nn)

        return sub_x, sub_spa, sub_sem

# =====================================================================
# 백본 메인 모듈
# =====================================================================
class DeepPA_semseg(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum)**0.5
        self.stage = Stage_PA(args) 
        
        self.latent_head = nn.Sequential(
            nn.BatchNorm1d(args.head_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(args.head_dim, 256), 
            nn.BatchNorm1d(256, momentum=args.bn_momentum),
            args.act(),
            nn.Dropout(0.3),
            nn.Linear(256, 256) 
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, prior_heatmap=None, pts_list=None):
        indices = indices[:]
        x, spa, sem = self.stage(x, xyz, None, indices, pts_list, prior_heatmap)
        B, N, C = x.shape
        x = self.latent_head(x.view(-1, C)).view(B, N, -1)
        if self.training:
            return x, spa, sem 
        return x

# =====================================================================
# 👑 최상위 래퍼 모델 (입구 역할만 깔끔하게 수행)
# =====================================================================
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
                
        self.model = DeepPA_semseg(args)
        
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
        
        # 🌟 [해결책 4] 들어오는 포맷(List vs Tensor)에 맞게 공간/채널 차원 순서(Permute)만 맞춰줍니다.
        if prior_hints is not None:
            if isinstance(prior_hints, list):
                prior_hints = [h.permute(0, 2, 1).contiguous() for h in prior_hints]
            else:
                prior_hints = prior_hints.permute(0, 2, 1).contiguous()

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
        
        in_features = x.permute(0, 2, 1).contiguous()
        
        # 백본 통과
        out = self.model(xyz=xyz_coords, x=in_features, indices=indices, prior_hints=prior_hints)
        
        dense_features = out[0] if isinstance(out, tuple) else out
        if dense_features.shape[-1] == 256: dense_features = dense_features.permute(0, 2, 1).contiguous() 

        if dense_features.shape[2] != N_in:
            dense_features = index_points(dense_features.permute(0, 2, 1), up_idx_list[0]).permute(0, 2, 1)

        xyz_full = xyz_coords.permute(0, 2, 1).contiguous() 
        fused_features = torch.cat([dense_features, xyz_full], dim=1) 
        
        x_fused = self.head_conv(fused_features) 
        x_pool = torch.max(x_fused, dim=2)[0]    
        coords = self.head_linear(x_pool).view(B, self.landmark_num, 3) 
        
        if self.training:
            spa_loss = out[1] if isinstance(out, tuple) else torch.tensor(0.0).to(device)
            sem_list = out[2] if isinstance(out, tuple) else []
            return coords, spa_loss, sem_list
            
        return coords