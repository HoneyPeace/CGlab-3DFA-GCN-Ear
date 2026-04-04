import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import knn_edge_maxpooling

def index_points(points, idx):
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

class Stage(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()
        self.depth = depth
        self.up_depth = len(args.depths) - 1
        self.first = depth == 0
        self.last = depth == self.up_depth
        self.k = args.ks[depth]
        self.cp = args.use_cp
        cp_bn_momentum = args.cp_bn_momentum if self.cp else args.bn_momentum

        dim = args.dims[depth]
        if self.first:
            nbr_hid_dim = args.nbr_dims[0]
            
            # 🌟 [신규 추가] 6채널이 들어오면 피처를 12채널(상대위치3+입력6+방향변화3)로 동적 확장
            in_channels = getattr(args, 'in_channels', 3)
            in_feat_dim = 12 if in_channels == 6 else 6
            
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
            nn.BatchNorm1d(3, momentum=args.bn_momentum),
            nn.Linear(3, args.num_classes, bias=False),
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
            self.sub_stage = Stage(args, depth + 1)

    def forward(self, x, xyz, prev_knn, indices, pts_list, sub_spa=None, sub_sem=None):
        B, N_in, C_in = x.shape
        
        if not self.first:
            ids = indices.pop()
            xyz = index_points(xyz, ids)
            x_skip = index_points(self.skip_proj(x.view(-1, C_in)).view(B, N_in, -1), ids)
            x_vfr = index_points(self.vfr(x, prev_knn), ids)
            x = x_skip + x_vfr
            
        knn = indices.pop()
        B, N, C = x.shape

        xyz_knn = index_points(xyz, knn)
        pe = xyz_knn - xyz.unsqueeze(2)

        if self.first:
            nbr_rel = pe.clone()
            x_knn = index_points(x, knn)
            
            # 🌟 [다이나믹 6/12채널 조립]
            if C_in == 6:
                center_v = x[:, :, 3:].unsqueeze(2) 
                neighbor_v = x_knn[:, :, :, 3:]    
                relative_v = neighbor_v - center_v  
                # 상대위치(3) + 6채널입력(6) + 방향변화(3) = 12채널!
                nbr = torch.cat([nbr_rel, x_knn, relative_v], dim=-1).view(-1, 12) 
            else:
                # 3채널(기본) 입력 시 6채널 유지
                nbr = torch.cat([nbr_rel, x_knn], dim=-1).view(-1, 6) 
                
            nbr_embed_func = lambda t: self.nbr_embed(t).view(B, N, self.k, -1).max(dim=2)[0]
            nbr = checkpoint(nbr_embed_func, nbr) if self.training and self.cp else nbr_embed_func(nbr)
            nbr = self.nbr_proj(nbr)
            x = self.nbr_bn(nbr.view(-1, nbr.shape[-1])).view(B, N, -1)

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

            sem = self.sem_sup(torch.max(spa_info, dim=2)[0].view(-1, 3)).view(B, N, -1)
            if sub_sem is not None:
                sub_sem.append(sem)
            else:
                sub_sem = [sem]

        if not self.last:
            sub_x, sub_spa, sub_sem = self.sub_stage(x, xyz, knn, indices, pts_list, sub_spa, sub_sem)
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

class DeepLA_semseg(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum)**0.5
        self.stage = Stage(args)
        self.seg_head = nn.Sequential(
            nn.BatchNorm1d(args.head_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(args.head_dim, args.head_dim//2),
            nn.BatchNorm1d(args.head_dim//2, momentum=args.bn_momentum),
            args.act(),
            nn.Dropout(0.5),
            nn.Linear(args.head_dim//2, args.num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, pts_list=None):
        indices = indices[:]
        x, spa, sem = self.stage(x, xyz, None, indices, pts_list)
        B, N, C = x.shape
        x = self.seg_head(x.view(-1, C)).view(B, N, -1)
        if self.training:
            return x, spa, sem
        return x