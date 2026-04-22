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

# =====================================================================
# 유틸리티 함수
# =====================================================================
def index_points(points, idx):
    """
    [특정 인덱스의 포인트 데이터 추출 (Gathering)]
    다운샘플링(FPS)이나 K-NN 검색으로 얻은 인덱스(idx) 배열을 이용해, 원본 좌표/특징 텐서를 추출합니다.
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
    """ VRAM 절약을 위한 PyTorch Gradient Checkpointing 래퍼 함수 """
    try:
        return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)
    except ValueError:
        return torch_checkpoint(function, *args, **kwargs)

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
# 계층적(Hierarchical) 스테이지 설계
# =====================================================================
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
        
        # 🌟 [디펜스 포인트] Latent Feature 투영기 (게이트 OFF 시에도 채널 일치를 위해 무조건 실행됨)
        self.prior_proj = nn.Sequential(
            nn.Linear(128, dim, bias=False), 
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            args.act()
        )
        
        # 🌟 [Ablation] 게이트 사용 스위치 로드 및 조건부 파라미터 생성
        self.use_gate = getattr(args, 'use_gate', True)
        
        if self.use_gate:
            # 수식의 \sigma 부분: 2배 차원(DeepPA+PAConv)을 받아 0~1 게이트 점수 생성
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

        # =====================================================================
        # 🚨 [버그 픽스 완료!] HDS 모의고사용 Semantic 로스 헤드
        # 기존: 3채널(단순 거리)만 받던 것을 -> dim 채널(딥러닝 특징 전체)을 받도록 수정
        # =====================================================================
        self.sem_sup = nn.Sequential(
            nn.Dropout(0.5),
            nn.BatchNorm1d(dim, momentum=args.bn_momentum), # 3 -> dim 으로 수정됨
            nn.Linear(dim, getattr(args, 'num_classes', 36), bias=False), # 3 -> dim 으로 수정됨
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
        
        if not self.first:
            ids = indices.pop()
            xyz = index_points(xyz, ids)
            x_skip = index_points(self.skip_proj(x.view(-1, C_in)).view(B, N_in, -1), ids)
            x_vfr = index_points(self.vfr(x, prev_knn), ids)
            x = x_skip + x_vfr
            
            # 해상도 동기화
            if prior_heatmap is not None:
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
        # 🌟 수식 완벽 구현: 게이트 기반 잔차 연결 vs 단순 덧셈 (Ablation)
        # =====================================================================
        if not self.first and prior_heatmap is not None:
            # 1. PAConv 힌트(128ch)를 백본 차원(dim)에 맞게 투영 (차원 에러 방어!)
            p_feat = self.prior_proj(prior_heatmap.view(-1, prior_heatmap.shape[-1])).view(B, N, -1)
            
            if getattr(self, 'use_gate', True):
                # 🌟 [게이트 ON]: 콘캣 -> 시그모이드 게이트 생성 -> 곱해서 더하기
                fused_for_gate = torch.cat([x, p_feat], dim=-1) 
                gate_matrix = self.gate_mlp(fused_for_gate.view(-1, fused_for_gate.shape[-1])).view(B, N, -1)
                x = x + (gate_matrix * p_feat)
            else:
                # 🌟 [게이트 OFF]: 게이트 행렬 곱셈 없이 투영된 힌트를 그대로 덧셈 (Simple Residual Add)
                x = x + p_feat

        pe = pe.view(-1, 3)
        pe_embed_func = lambda t: self.pe_embed(t).view(B, N, self.k, -1).max(dim=2)[0]
        pe = checkpoint(pe_embed_func, pe) if self.training and self.cp else pe_embed_func(pe)
        pe = self.pe_proj(pe)
        pe = self.pe_bn(pe.view(-1, pe.shape[-1])).view(B, N, -1)

        pts = pts_list.pop() if pts_list is not None else None
        x = checkpoint(self.reslfe, x, pe, knn, pts) if self.training and self.cp else self.reslfe(x, pe, knn, pts)

        if self.training:
            # HDS 모의고사 채점
            spa_info = xyz_knn - xyz.unsqueeze(2)
            spa_info.mul_(self.cor_std)
            feat_info = self.cor_head(x.view(-1, x.shape[-1])).view(B, N, -1)
            feat_info_knn = index_points(feat_info, knn)
            feat_info = feat_info_knn - feat_info.unsqueeze(2)
            
            closs = F.mse_loss(feat_info, spa_info) 
            sub_spa = sub_spa + closs if sub_spa is not None else closs

            # =====================================================================
            # 🚨 [버그 픽스 완료!] spa_info가 아닌 모델 특징 'x'를 투입!
            # =====================================================================
            sem = self.sem_sup(x.view(-1, x.shape[-1])).view(B, N, -1) 
            
            if sub_sem is not None:
                sub_sem.append(sem)
            else:
                sub_sem = [sem]

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
# 최종 통합 모델 (DeepPA_semseg 백본 진입점)
# =====================================================================
class DeepPA_semseg(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum)**0.5
        self.stage = Stage_PA(args) 
        
        # 🌟 [디펜스 포인트] 256채널 병목 제거 유지
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