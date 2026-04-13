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

"""
[NotebookLM을 위한 모듈 요약]
이 파일은 DeepPA(Deep Position Adaptive) 네트워크의 Semantic Segmentation(랜드마크 지역 정밀 분할) 버전입니다.
DeepPA_auto 모드에서 '2단계(Local Refinement)'를 담당하며, 
1단계(PAConv)에서 대략적으로 예측한 랜드마크 위치(prior_heatmap)를 힌트로 받아들여 
로컬 영역의 굴곡과 위상(Topology)을 깊게 파고들어 최종적으로 아주 정밀한 위치를 짚어내는 역할을 합니다.
"""

# =====================================================================
# 유틸리티 함수 및 기본 블록
# =====================================================================

def index_points(points, idx):
    """
    [특정 인덱스의 포인트 데이터 추출 (Gathering)]
    다운샘플링(FPS)이나 K-NN 검색으로 얻은 인덱스(idx) 배열을 이용해, 원본 좌표/특징 텐서에서 실제 값들을 뽑아냅니다.
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

class VFR(nn.Module):
    """
    [Vector Field Routing (VFR) Block]
    단순한 점 단위(Point-wise) MLP가 아니라, K-NN 이웃들 사이의 연결(Edge)을 바탕으로 
    가장 두드러지는 특징을 끌어올리는(Max-pooling) 모듈입니다.
    """
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
    """
    [Feed Forward Network (FFN)]
    Point-wise 채널 확장을 통해 비선형성을 부여하는 일반적인 모듈. (Transformer의 FFN과 유사한 역할)
    """
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
    """
    [Residual Local Feature Extraction (ResLFE) Block]
    - 위치 인코딩(PE: Position Encoding)을 매 깊이(depth)마다 지속적으로 더해주어(Residual),
      네트워크가 깊어져도 모델이 "현재 점의 물리적 위치(Local Geometry)"를 잊지 않도록 설계된 핵심 블록.
    """
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
            x = x + pe # 위치 정보(PE)를 매 반복마다 주입 (Residual)
            x = x + self.drop_path(self.VFRs[i](x, knn), i)
            x = x + self.drop_path(self.FFNs[i](x), i)
        return x

# =====================================================================
# 계층적(Hierarchical) 스테이지 설계
# =====================================================================

class Stage_PA(nn.Module):
    """
    [U-Net 형태의 계층적 특징 추출 스테이지 (Hierarchical Stage)]
    - 다운샘플링(Down-sampling)을 통해 해상도는 줄이되 수용 영역(Receptive Field)을 넓혀가며
      형태를 파악하는 서브 스테이지(sub_stage)들을 재귀적으로 호출하는 구조.
    """
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
        
        # 🌟 [Two-stage Fusion] 1단계 결과물(prior_heatmap)을 현재 깊이의 차원(dim)에 맞게 투영하는 모듈
        self.prior_proj = nn.Sequential(
            nn.Linear(args.num_classes, dim, bias=False),
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            args.act()
        )
        
        # 원본 기하학 피처와 prior_heatmap 피처를 하나로 융합(Fusion)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False),
            nn.BatchNorm1d(dim, momentum=args.bn_momentum),
            args.act()
        )

        # 스테이지의 맨 처음(First)일 경우 입력 데이터의 채널을 처리하기 위한 블록
        if self.first:
            nbr_hid_dim = args.nbr_dims[0]
            
            # 🌟 [7채널 기하학 피처 동기화 및 엣지 생성 (util.py, My_args.py 연동)]
            # 단순히 (X,Y,Z)만 있다면 중심점과 이웃의 차이를 구해 (3차원) 등을 만들겠지만,
            # in_channels에 따라 주방향 벡터와 곡률까지 포함되어 Edge Feature Dimension이 동적으로 바뀜.
            in_channels = getattr(args, 'in_channels', 3)
            if in_channels == 7:
                in_feat_dim = 14 # nbr_rel(3) + x_knn(7) + dist(1) + vector(3) = 14
            elif in_channels == 6:
                in_feat_dim = 13
            else:
                in_feat_dim = 10 # 기본 모드
            
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

        # 이웃 간 상대적 위치(Relative Position)를 인코딩 (Local Geometry)
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

        # U-Net 구조의 Skip Connection을 위한 프로젝션 레이어
        if not self.first:
            self.vfr = VFR(args.dims[depth - 1], dim, args.bn_momentum, 0.3)
            self.skip_proj = nn.Sequential(
                nn.Linear(args.dims[depth - 1], dim, bias=False),
                nn.BatchNorm1d(dim, momentum=args.bn_momentum)
            )
            nn.init.constant_(self.skip_proj[1].weight, 0.3)

        self.reslfe = ResLFE_Block(dim, args.depths[depth], args.drop_paths[depth], args.mlp_ratio, cp_bn_momentum, args.act)
        self.drop = DropPath(args.head_drops[depth])

        # 보조 손실(Auxiliary Loss)을 계산하기 위한 헤드(Head)
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

        # Feature 공간 상의 거리가 물리적(Spatial) 거리와 비례하도록 강제하는 Contrastive Head
        self.cor_std = 1 / args.cor_std[depth]
        self.cor_head = nn.Sequential(
            nn.Linear(dim, 32, bias=False),
            nn.BatchNorm1d(32, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(32, 3, bias=False),
        )

        # 재귀적으로 다음 깊이(depth + 1)의 스테이지를 생성
        if not self.last:
            self.sub_stage = Stage_PA(args, depth + 1)

    def forward(self, x, xyz, prev_knn, indices, pts_list, prior_heatmap=None, sub_spa=None, sub_sem=None):
        B, N_in, C_in = x.shape
        
        # 1. 다운샘플링 및 Skip Connection 처리
        if not self.first:
            ids = indices.pop()
            xyz = index_points(xyz, ids)
            x_skip = index_points(self.skip_proj(x.view(-1, C_in)).view(B, N_in, -1), ids)
            x_vfr = index_points(self.vfr(x, prev_knn), ids)
            x = x_skip + x_vfr
            
            # 🔥 1단계 힌트(prior) 역시 다운샘플링 포인트에 맞춰 해상도를 줄임
            if prior_heatmap is not None:
                prior_heatmap = index_points(prior_heatmap, ids)
            
        knn = indices.pop()
        B, N, C = x.shape

        xyz_knn = index_points(xyz, knn)
        pe = xyz_knn - xyz.unsqueeze(2) # (B, N, K, 3) 이웃 점들과의 상대 좌표

        # 2. 첫 번째 스테이지의 엣지 피처(Neighborhood Features) 인코딩
        if self.first:
            nbr_rel = pe.clone() 
            x_knn = index_points(x, knn) 
            dist = torch.norm(nbr_rel, dim=-1, keepdim=True) 
            vector = nbr_rel / (dist + 1e-8)
            
            # 🔥 [채널 분기] 모델 설정(in_channels)에 맞게 14채널, 13채널, 10채널로 조립
            if C_in == 7:
                nbr = torch.cat([nbr_rel, x_knn, dist, vector], dim=-1).view(-1, 14)
            elif C_in == 6:
                nbr = torch.cat([nbr_rel, x_knn, dist, vector], dim=-1).view(-1, 13) 
            else:
                nbr = torch.cat([nbr_rel, x_knn, dist, vector], dim=-1).view(-1, 10) 
            
            nbr_embed_func = lambda t: self.nbr_embed(t).view(B, N, self.k, -1).max(dim=2)[0]
            nbr = checkpoint(nbr_embed_func, nbr) if self.training and self.cp else nbr_embed_func(nbr)
            nbr = self.nbr_proj(nbr)
            x = self.nbr_bn(nbr.view(-1, nbr.shape[-1])).view(B, N, -1)

        # 3. 🌟 [핵심] Prior Heatmap Fusion (1단계 PAConv의 힌트 주입)
        if prior_heatmap is not None:
            # 1단계 확률 지도를 현재 피처 차원(dim)으로 맵핑
            p_feat = self.prior_proj(prior_heatmap.view(-1, prior_heatmap.shape[-1])).view(B, N, -1)
            fused = torch.cat([x, p_feat], dim=-1) # (기존 피처 + 힌트 피처) 연결
            # MLP를 통과시켜 의미 있는 정보만 추출(mixed_residual) 후 원본 피처에 주입
            mixed_residual = self.fusion_mlp(fused.view(-1, fused.shape[-1])).view(B, N, -1)
            x = x + mixed_residual

        # 4. 상대 위치 기반 포지션 인코딩 (Position Encoding)
        pe = pe.view(-1, 3)
        pe_embed_func = lambda t: self.pe_embed(t).view(B, N, self.k, -1).max(dim=2)[0]
        pe = checkpoint(pe_embed_func, pe) if self.training and self.cp else pe_embed_func(pe)
        pe = self.pe_proj(pe)
        pe = self.pe_bn(pe.view(-1, pe.shape[-1])).view(B, N, -1)

        pts = pts_list.pop() if pts_list is not None else None

        # 5. ResLFE 블록 통과 (Feature Extraction)
        x = checkpoint(self.reslfe, x, pe, knn, pts) if self.training and self.cp else self.reslfe(x, pe, knn, pts)

        # 6. 학습 중 보조 손실(Auxiliary Loss) 계산 (물리적 거리 보정 및 얕은 층의 예측력 강화)
        if self.training:
            spa_info = xyz_knn - xyz.unsqueeze(2)
            spa_info.mul_(self.cor_std)
            feat_info = self.cor_head(x.view(-1, x.shape[-1])).view(B, N, -1)
            feat_info_knn = index_points(feat_info, knn)
            feat_info = feat_info_knn - feat_info.unsqueeze(2)
            closs = F.mse_loss(feat_info, spa_info) # 피처 거리가 물리 거리를 따르도록 강제 (Spatial Loss)
            sub_spa = sub_spa + closs if sub_spa is not None else closs

            sem = self.sem_sup(torch.max(spa_info, dim=2)[0].view(-1, 3)).view(B, N, -1) # 보조 Semantic 분할
            if sub_sem is not None:
                sub_sem.append(sem)
            else:
                sub_sem = [sem]

        # 7. 재귀적 하위 스테이지 호출 (U-Net 형태)
        if not self.last:
            sub_x, sub_spa, sub_sem = self.sub_stage(x, xyz, knn, indices, pts_list, prior_heatmap, sub_spa, sub_sem)
        else:
            sub_x = None
            self.spa, self.sem = sub_spa, sub_sem

        x = self.postproj(x.view(-1, x.shape[-1])).view(B, N, -1)
        sub_x = sub_x + x if sub_x is not None else x # 업샘플링된 피처와 결합 (Skip Connection 합치기)
        sub_x = self.drop(sub_x)
        
        # 8. 업샘플링(Up-sampling) - 하위 깊이에서 줄여놨던 해상도를 다시 복원
        if not self.first:
            back_nn = indices[self.depth-1]
            sub_x = index_points(sub_x, back_nn)

        return sub_x, sub_spa, sub_sem

# =====================================================================
# 최종 통합 모델 (DeepPA_semseg)
# =====================================================================

class DeepPA_semseg(nn.Module):
    """
    [DeepPA 모델 진입점]
    - Stage_PA를 통해 계층적 피처 추출을 끝낸 후, 최종적으로 Seg_Head를 통과하여 
      원하는 랜드마크 개수(num_classes) 만큼의 확률 지도(Heatmap)를 내보냅니다.
    """
    def __init__(self, args):
        super().__init__()
        args.cp_bn_momentum = 1 - (1 - args.bn_momentum)**0.5
        self.stage = Stage_PA(args) # U-Net 스타일의 중추 신경망
        
        # 마지막 피처를 각 점이 N번째 랜드마크일 확률로 분류하는 헤드
        self.seg_head = nn.Sequential(
            nn.BatchNorm1d(args.head_dim, momentum=args.bn_momentum),
            args.act(),
            nn.Linear(args.head_dim, args.head_dim//2),
            nn.BatchNorm1d(args.head_dim//2, momentum=args.bn_momentum),
            args.act(),
            nn.Dropout(0.5),
            nn.Linear(args.head_dim//2, args.num_classes) # num_classes = 랜드마크 개수
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, xyz, x, indices, prior_heatmap=None, pts_list=None):
        indices = indices[:]
        # Stage_PA 통과 (피처 추출 + 1단계 힌트 결합)
        x, spa, sem = self.stage(x, xyz, None, indices, pts_list, prior_heatmap)
        B, N, C = x.shape
        
        # Seg Head 통과하여 최종 랜드마크 히트맵 픽셀 값 출력
        x = self.seg_head(x.view(-1, C)).view(B, N, -1)
        
        if self.training:
            return x, spa, sem # 학습 중에는 보조 손실(spa, sem)도 같이 반환
        return x