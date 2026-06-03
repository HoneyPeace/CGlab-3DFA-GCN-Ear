# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: deeppa_semseg.py
# @Description: 
# [S2G 백본 최적화 - 단일 파라미터 체계 완벽 적용]
# - 🌟 latent_injection_type ('none', 'raw', 'compressed') 옵션 전면 적용
# - 🌟 [버그 픽스] Layer 투영(Linear vs Conv1d) 간의 텐서 Shape(B, N, C) 충돌 완벽 차단
# - 🌟 [버그 픽스] 해상도 다운샘플링(index_points) 시 발생할 수 있는 차원 혼돈 방지
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn.init import trunc_normal_
import sys
from pathlib import Path

# 경로 설정
sys.path.append(str(Path(__file__).absolute().parent.parent))
from utils.timm.models.layers import DropPath
from utils.cutils import knn_edge_maxpooling

def index_points(points, idx):
    """
    입력 points는 반드시 (B, N, C) 형태여야 합니다.
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

def get_decoder_out_dim(args, depth):
    mode = getattr(args, 'decoder_fusion', 'add').lower()
    dims = list(args.dims)
    if mode == 'add':
        return getattr(args, 'head_dim', 256)
    if mode == 'raw_concat':
        return sum(dims[depth:])
    if mode == 'prog_half_final320':
        out_dim = dims[-1]
        for d in range(len(dims) - 2, depth - 1, -1):
            merged_dim = out_dim + dims[d]
            out_dim = merged_dim if d == 0 else merged_dim // 2
        return out_dim
    raise ValueError(f"Unknown decoder_fusion: {mode}")

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
            nn.Linear(in_dim, hid_dim), act(),
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
        if not self.dp[i] or not self.training: return x
        return self.drop_paths[i](x)

    def forward(self, x, pe, knn, pts=None):
        x = x + self.drop_path(self.mlp(x), 0)
        for i in range(self.depth):
            x = x + pe 
            x = x + self.drop_path(self.VFRs[i](x, knn), i)
            x = x + self.drop_path(self.FFNs[i](x), i)
        return x

class Stage_PA(nn.Module):
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
        
        # =====================================================================
        # 🌟 1. 옵션 파라미터 단일화 (3지 선다형 문자열 기반)
        # =====================================================================
        self.injection_type = getattr(args, 'latent_injection_type', 'raw').lower()
        self.latent_fusion_mode = getattr(args, 'latent_fusion_mode', 'auto').lower()
        self.use_feature_gating = getattr(args, 'use_feature_gating', False) 
        self.use_interaction_fusion = getattr(args, 'use_interaction_fusion', True)
        self.fusion_residual_base = getattr(args, 'fusion_residual_base', 'deeppa').lower()
        self.decoder_fusion = getattr(args, 'decoder_fusion', 'add').lower()
        self.decoder_out_dim = get_decoder_out_dim(args, depth)
        
        # 🌟 2. 주입 타입별 투영(Projection) 레이어 동적 생성
        if self.injection_type != 'none':
            if self.injection_type == 'raw':
                # [Raw 모드]: 1344 -> 현재 층 dim 압축. Conv1d 사용하므로 (B, C, N) 입력을 기대함.
                self.prior_proj = nn.Sequential(
                    nn.Conv1d(1344, dim, kernel_size=1, bias=False), 
                    nn.BatchNorm1d(dim, momentum=args.bn_momentum),
                    args.act()
                )
            elif self.injection_type == 'compressed':
                # [Compressed 모드]: 맞춤 채널 -> 현재 층 dim 압축. Linear 사용하므로 (B, N, C) 입력을 기대함.
                in_c_list = [64, 128, 256, 512]
                in_c = in_c_list[depth] if depth < len(in_c_list) else 512
                self.prior_proj = nn.Sequential(
                    nn.Linear(in_c, dim, bias=False), 
                    nn.BatchNorm1d(dim, momentum=args.bn_momentum),
                    args.act()
                )
            
            if self.use_feature_gating:
                self.gate_mlp = nn.Sequential(
                    nn.Linear(dim * 2, dim, bias=False),
                    nn.BatchNorm1d(dim, momentum=args.bn_momentum),
                    nn.Sigmoid() 
                )
            if self.use_interaction_fusion or self.latent_fusion_mode in ('residual', 'concat'):
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(dim * 2, dim, bias=False),
                    nn.BatchNorm1d(dim, momentum=args.bn_momentum),
                    args.act()
                )

        if self.first:
            nbr_hid_dim = args.nbr_dims[0]
            in_channels = getattr(args, 'in_channels', 3)
            self.deeppa_feature_mode = getattr(args, 'deeppa_feature_mode', 'center_geometry').lower()
            if self.deeppa_feature_mode not in ('center_geometry', 'full_extension', 'deepla_neighbor_attr', 'surface_pair_no_delta', 'center_attr', 'xyzlocal_neighbor_attr'):
                raise ValueError(f"Unknown deeppa_feature_mode: {self.deeppa_feature_mode}")
            if in_channels == 7 and self.deeppa_feature_mode == 'full_extension':
                in_feat_dim = 22
            elif in_channels == 7 and self.deeppa_feature_mode == 'surface_pair_no_delta':
                in_feat_dim = 18
            elif in_channels == 7 and self.deeppa_feature_mode in ('deepla_neighbor_attr', 'center_attr'):
                in_feat_dim = 7
            elif in_channels == 7 and self.deeppa_feature_mode == 'xyzlocal_neighbor_attr':
                in_feat_dim = 14
            else:
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

        if self.decoder_fusion == 'prog_half_final320' and not self.last:
            child_out_dim = get_decoder_out_dim(args, depth + 1)
            merge_in_dim = child_out_dim + dim
            merge_out_dim = self.decoder_out_dim
            if merge_in_dim == merge_out_dim:
                self.decoder_merge_proj = nn.Identity()
            else:
                self.decoder_merge_proj = nn.Sequential(
                    nn.Linear(merge_in_dim, merge_out_dim, bias=False),
                    nn.BatchNorm1d(merge_out_dim, momentum=args.bn_momentum),
                    args.act(),
                )

        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        if not self.last:
            self.sub_stage = Stage_PA(args, depth + 1)

    def forward(self, x, xyz, prev_knn, indices, pts_list, prior_hints=None, sub_spa=None, sub_sem=None):
        B, N_in, C_in = x.shape

        # =====================================================================
        # 🌟 3. 데이터 전처리 및 None 처리
        # =====================================================================
        current_stage_hint = None
        
        if self.injection_type == 'none':
            prior_hints = None 

        if prior_hints is not None:
            if isinstance(prior_hints, list):
                hint_idx = min(self.depth, len(prior_hints) - 1)
                current_stage_hint = prior_hints[hint_idx]
            else:
                current_stage_hint = prior_hints
                
            # 🌟 [버그 픽스] 무조건 (B, N, C) 포맷으로 통일하여 다운샘플링 혼란 방지
            if current_stage_hint.shape[2] > current_stage_hint.shape[1]: 
                current_stage_hint = current_stage_hint.permute(0, 2, 1).contiguous()

        # =====================================================================
        # 🌟 4. 물리적 좌표 기반 해상도 다운샘플링 매칭
        # =====================================================================
        if not self.first:
            ids = indices.pop() 
            xyz = index_points(xyz, ids)
            x_skip = index_points(self.skip_proj(x.view(-1, C_in)).view(B, N_in, -1), ids)
            x_vfr = index_points(self.vfr(x, prev_knn), ids)
            x = x_skip + x_vfr
            
            if current_stage_hint is not None:
                # ids.shape[1]은 현재 층의 점의 개수(N). 
                # 힌트의 형태가 (B, N, C)이므로 shape[1]이 점의 개수인지 확인.
                if current_stage_hint.shape[1] != ids.shape[1]:
                     current_stage_hint = index_points(current_stage_hint, ids) 

        knn = indices.pop()
        B, N, C = x.shape

        xyz_knn = index_points(xyz, knn)
        pe = xyz_knn - xyz.unsqueeze(2) 

        if self.first:
            nbr_rel = pe.clone() 
            x_knn = index_points(x, knn) 
            dist = torch.norm(nbr_rel, dim=-1, keepdim=True) 
            vector = nbr_rel / (dist + 1e-8)
            
            if C_in == 7:
                center = x.unsqueeze(2).expand(-1, -1, self.k, -1)
                if self.deeppa_feature_mode == 'full_extension':
                    nbr = torch.cat([x_knn - center, x_knn, center, dist], dim=-1).view(-1, 22)
                elif self.deeppa_feature_mode == 'surface_pair_no_delta':
                    # Use only XYZ for the edge delta; dir/curv are attributes, not vector differences.
                    nbr = torch.cat([nbr_rel, x_knn, center, dist], dim=-1).view(-1, 18)
                elif self.deeppa_feature_mode == 'deepla_neighbor_attr':
                    # Original DeepLA first-stage pattern: relative XYZ plus grouped neighbor attributes.
                    neighbor_geom = x_knn[..., 3:]
                    nbr = torch.cat([nbr_rel, neighbor_geom], dim=-1).view(-1, 7)
                elif self.deeppa_feature_mode == 'center_attr':
                    center_geom = x[:, :, 3:].unsqueeze(2).expand(-1, -1, self.k, -1)
                    nbr = torch.cat([x_knn[..., :3], center_geom], dim=-1).view(-1, 7)
                elif self.deeppa_feature_mode == 'xyzlocal_neighbor_attr':
                    # Original XYZ-local descriptor plus neighbor dir/curv attributes.
                    nbr = torch.cat([nbr_rel, x_knn, dist, vector], dim=-1).view(-1, 14)
                else:
                    center_xyz = xyz.unsqueeze(2).expand(-1, -1, self.k, -1)
                    center_geom = x[:, :, 3:].unsqueeze(2).expand(-1, -1, self.k, -1)
                    nbr = torch.cat([nbr_rel, x_knn[..., :3], center_xyz, dist, center_geom], dim=-1).view(-1, 14)
            elif C_in == 6: nbr = torch.cat([nbr_rel, x_knn, dist, vector], dim=-1).view(-1, 13) 
            else: nbr = torch.cat([nbr_rel, x_knn, dist, vector], dim=-1).view(-1, 10) 
            
            nbr_embed_func = lambda t: self.nbr_embed(t).view(B, N, self.k, -1).max(dim=2)[0]
            nbr = checkpoint(nbr_embed_func, nbr) if self.training and self.cp else nbr_embed_func(nbr)
            nbr = self.nbr_proj(nbr)
            x = self.nbr_bn(nbr.view(-1, nbr.shape[-1])).view(B, N, -1)

        # =====================================================================
        # 🌟 5. 타입별 피처 주입 & 게이팅 제어 (Shape Error 원천 차단)
        # =====================================================================
        if current_stage_hint is not None:
            if self.injection_type == 'raw':
                # Conv1d는 (B, C, N) 입력 필요. 들어온 힌트는 (B, N, C) 상태임.
                hint_bcn = current_stage_hint.permute(0, 2, 1).contiguous()
                p_feat = self.prior_proj(hint_bcn)
                p_feat = p_feat.permute(0, 2, 1).contiguous() # (B, N, C)로 원상복구
            elif self.injection_type == 'compressed':
                # Linear는 (B, N, C) 입력 필요. 힌트는 이미 (B, N, C) 상태임.
                p_feat = self.prior_proj(current_stage_hint)
            
            if self.latent_fusion_mode == 'auto':
                use_interaction_fusion = self.use_interaction_fusion
                use_feature_gating = self.use_feature_gating
            elif self.latent_fusion_mode == 'residual':
                use_interaction_fusion = True
                use_feature_gating = False
            elif self.latent_fusion_mode in ('concat', 'add'):
                use_interaction_fusion = False
                use_feature_gating = False
            else:
                raise ValueError(f"Unknown latent_fusion_mode: {self.latent_fusion_mode}")

            if self.latent_fusion_mode == 'concat':
                fused = torch.cat([x, p_feat], dim=-1)
                x = self.fusion_mlp(fused.view(-1, fused.shape[-1])).view(B, N, -1)
            elif use_interaction_fusion:
                fused = torch.cat([x, p_feat], dim=-1)
                mixed_residual = self.fusion_mlp(fused.view(-1, fused.shape[-1])).view(B, N, -1)
                if self.fusion_residual_base == 'prior':
                    x = p_feat + mixed_residual
                else:
                    x = x + mixed_residual
            elif use_feature_gating:
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

        if not self.last:
            sub_x, sub_spa, sub_sem = self.sub_stage(x, xyz, knn, indices, pts_list, prior_hints, sub_spa, sub_sem)
        else:
            sub_x = None
            self.spa, self.sem = sub_spa, sub_sem

        if self.decoder_fusion == 'add':
            x = self.postproj(x.view(-1, x.shape[-1])).view(B, N, -1)
            sub_x = sub_x + x if sub_x is not None else x
        elif self.decoder_fusion == 'raw_concat':
            sub_x = torch.cat([sub_x, x], dim=-1) if sub_x is not None else x
        elif self.decoder_fusion == 'prog_half_final320':
            if sub_x is not None:
                merged_x = torch.cat([sub_x, x], dim=-1)
                if isinstance(self.decoder_merge_proj, nn.Identity):
                    sub_x = merged_x
                else:
                    sub_x = self.decoder_merge_proj(merged_x.view(-1, merged_x.shape[-1])).view(B, N, -1)
            else:
                sub_x = x
        else:
            raise ValueError(f"Unknown decoder_fusion: {self.decoder_fusion}")
        sub_x = self.drop(sub_x)
        
        if not self.first:
            back_nn = indices[self.depth-1]
            sub_x = index_points(sub_x, back_nn)

        return sub_x, sub_spa, sub_sem

class DeepPA_semseg(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum)**0.5
        self.decoder_fusion = getattr(args, 'decoder_fusion', 'add').lower()
        self.stage = Stage_PA(args) 

        if self.decoder_fusion == 'add':
            self.out_channels = 256
            self.latent_head = nn.Sequential(
                nn.BatchNorm1d(args.head_dim, momentum=args.bn_momentum),
                args.act(),
                nn.Linear(args.head_dim, 256),
                nn.BatchNorm1d(256, momentum=args.bn_momentum),
                args.act(),
                nn.Dropout(0.3),
                nn.Linear(256, 256)
            )
        else:
            self.out_channels = self.stage.decoder_out_dim
            self.latent_head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, prior_hints=None, pts_list=None):
        indices = indices[:]
        x, spa, sem = self.stage(x, xyz, None, indices, pts_list, prior_hints)
        B, N, C = x.shape

        if self.decoder_fusion == 'add':
            x = self.latent_head(x.view(-1, C)).view(B, N, -1)
        else:
            x = self.latent_head(x)
        
        if self.training:
            return x, spa, sem 
        return x
