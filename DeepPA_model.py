# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: DeepPA_model.py
# @Description: 
# [S2G 3D 랜드마크 탐지 모델 - DeepPA Wrapper 완결본]
# - 🌟 latent_injection_type ('none', 'raw', 'compressed') 완벽 동기화
# - 🌟 [최적화 1] Eval 상태 시 텐서 언패킹 크래시 방지 (반환 타입 통일)
# - 🌟 [최적화 2] PyTorch3D 초고속 C++ KNN 하이브리드 탑재 (학습 속도 극대화)
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

# ==============================================================================
# 🌟 [경로 설정 및 모듈 임포트]
# ==============================================================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "utils" / "pointnet2_ops_lib"))

from deeppa_semseg import DeepPA_semseg, index_points
from loss import get_differentiable_coords
from utils.cutils import grid_subsampling

try:
    from paconv import PAConv
except ImportError:
    pass 

# ==============================================================================
# 🔥 초고속 C++ 커널 로드 (FPS & KNN)
# ==============================================================================
try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cpp
    USE_CPP_FPS = True
    print(">>> [INFO] DeepPA 모드: 초고속 C++ FPS 커널을 사용합니다.")
except ImportError:
    USE_CPP_FPS = False
    print(">>> [WARNING] ⚠️ DeepPA 모드: C++ FPS 커널이 없어 PyTorch Fallback을 사용합니다.")

# 🌟 PyTorch3D 초고속 KNN 로드 시도
try:
    from pytorch3d.ops import knn_points
    USE_PYTORCH3D_KNN = True
    print(">>> [SUCCESS] 🚀 DeepPA 모드: PyTorch3D C++ KNN 커널 장착! (학습 속도 대폭 향상)")
except ImportError:
    USE_PYTORCH3D_KNN = False
    print(">>> [WARNING] ⚠️ DeepPA 모드: PyTorch3D가 없어 기존 PyTorch KNN을 사용합니다.")

def farthest_point_sample(xyz, npoint):
    if USE_CPP_FPS and xyz.is_cuda:
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

def _parse_float_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    text = str(value).strip().strip("[]")
    if not text:
        return []
    return [float(v.strip()) for v in text.split(",") if v.strip()]

def _fit_indices_to_npoint(xyz, indices, npoint):
    device = xyz.device
    total_points = xyz.shape[0]
    indices = indices.to(device=device, dtype=torch.long).view(-1)
    valid = (indices >= 0) & (indices < total_points)
    indices = torch.unique(indices[valid], sorted=False)

    if indices.numel() >= npoint:
        if indices.numel() == npoint:
            return indices
        candidate_xyz = xyz[indices].unsqueeze(0).contiguous()
        keep = farthest_point_sample(candidate_xyz, npoint).squeeze(0)
        return indices[keep]

    fps_idx = farthest_point_sample(xyz.unsqueeze(0).contiguous(), npoint).squeeze(0)
    if indices.numel() == 0:
        return fps_idx

    combined = torch.unique(torch.cat([indices, fps_idx], dim=0), sorted=False)
    if combined.numel() >= npoint:
        return combined[:npoint]

    remaining_mask = torch.ones(total_points, dtype=torch.bool, device=device)
    remaining_mask[combined] = False
    fill = fps_idx[remaining_mask[fps_idx]]
    if fill.numel() >= npoint - combined.numel():
        pad = fill[:npoint - combined.numel()]
    else:
        pad = fps_idx[:npoint - combined.numel()]
    return torch.cat([combined, pad], dim=0)

def _grid_subsample_single_fixed_count(xyz, npoint, grid_size, search_iters):
    shifted = xyz.detach().float()
    shifted = (shifted - shifted.min(dim=0, keepdim=True)[0]).cpu().contiguous()

    if grid_size > 0:
        indices = grid_subsampling(shifted, float(grid_size)).long()
        return _fit_indices_to_npoint(xyz, indices, npoint)

    span = shifted.max(dim=0)[0].clamp_min(1e-6)
    base = float((span.prod().item() / max(npoint, 1)) ** (1.0 / 3.0))
    if not math.isfinite(base) or base <= 0:
        base = 1e-3

    low = base / 16.0
    high = base * 16.0
    best_indices = None
    last_indices = None

    for _ in range(max(int(search_iters), 1)):
        mid = (low + high) * 0.5
        last_indices = grid_subsampling(shifted, float(mid)).long()
        if last_indices.numel() >= npoint:
            best_indices = last_indices
            low = mid
        else:
            high = mid

    if best_indices is None:
        best_indices = last_indices if last_indices is not None else torch.empty(0, dtype=torch.long)
    return _fit_indices_to_npoint(xyz, best_indices, npoint)

def grid_subsample_fixed_count(xyz, npoint, grid_size=0.0, search_iters=8):
    indices = [
        _grid_subsample_single_fixed_count(xyz[b], npoint, grid_size, search_iters)
        for b in range(xyz.shape[0])
    ]
    return torch.stack(indices, dim=0)

# ==============================================================================
# 👑 최상위 마스터 융합 모델
# ==============================================================================
class DeepPA_Wrapper(nn.Module):
    def __init__(self, args, landmark_num): 
        super().__init__()
        self.args = args
        self.landmark_num = landmark_num
        
        # 🌟 1. 3지 선다형 옵션 연결
        self.injection_type = getattr(args, 'latent_injection_type', 'raw').lower()
        # DeepPA coordinates are forced to heatmap readout; keep this attribute
        # only for old scripts/checkpoint metadata that may still reference it.
        self.coord_from_heatmap = True
        self.latest_main_heatmap_logits = None
        self.latest_sem_heatmap_logits = None
        self.latest_hm_attn_pooled_xyz = None
        self.latest_hm_attn_residual = None
        self.latest_hm_attn_final_coords = None
        self.current_residual_limit_norm = None
        
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
                
        # 🟡 DeepPA 백본
        self.model = DeepPA_semseg(args)
        decoder_out_channels = getattr(self.model, 'out_channels', 256)
        
        # 🔴 점진적 압축 회귀 헤드 (259 -> 1024 -> 512 -> 256 -> 108)
        bn_mom = getattr(args, 'bn_momentum', 0.1)
        self.head_conv = nn.Sequential(
            nn.Conv1d(decoder_out_channels + 3, 1024, 1),
            nn.BatchNorm1d(1024, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, 1),
            nn.BatchNorm1d(512, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.main_heatmap_head = nn.Conv1d(512, self.landmark_num, 1)
        self.coord_residual_mlp = nn.Sequential(
            nn.Linear(512 + 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 3)
        )
        self.coord_feature_residual_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 3)
        )
        self.head_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.landmark_num * 3) 
        )

    def set_residual_limit_norm(self, value):
        self.current_residual_limit_norm = None if value is None else float(value)

    def _activate_heatmap(self, logits):
        return torch.sigmoid(logits)

    def _heatmap_sigmoid_xyz_pool_coords(self, xyz_coords, heatmap_logits):
        attention = torch.sigmoid(heatmap_logits)
        attention = attention / (attention.sum(dim=2, keepdim=True) + 1e-6)
        return torch.bmm(attention, xyz_coords)

    def _heatmap_attention_residual_coords(self, xyz_coords, point_features, heatmap_logits):
        attention = torch.sigmoid(heatmap_logits)
        attention = attention / (attention.sum(dim=2, keepdim=True) + 1e-6)

        pooled_xyz = torch.bmm(attention, xyz_coords)
        pooled_feature = torch.bmm(attention, point_features.transpose(1, 2).contiguous())
        residual_input = torch.cat([pooled_feature, pooled_xyz], dim=2)

        B, L, _ = residual_input.shape
        residual = self.coord_residual_mlp(residual_input.view(B * L, -1)).view(B, L, 3)

        limit = self.current_residual_limit_norm
        if limit is None:
            limit = float(getattr(self.args, 'hm_attn_residual_max_norm', 0.0))
        if limit and limit > 0.0:
            residual = float(limit) * torch.tanh(residual)

        final_coords = pooled_xyz + residual
        self.latest_hm_attn_pooled_xyz = pooled_xyz.detach()
        self.latest_hm_attn_residual = residual.detach()
        self.latest_hm_attn_final_coords = final_coords.detach()
        return final_coords

    def _heatmap_topk_residual_coords(self, xyz_coords, point_features, heatmap_logits):
        heatmap_scores = torch.sigmoid(heatmap_logits)
        k_val = min(int(getattr(self.args, 'regression_point_num', 10)), heatmap_scores.shape[2])
        topk_scores, topk_idx = torch.topk(heatmap_scores, k_val, dim=2)
        attention = F.softmax(topk_scores, dim=2)

        topk_xyz = index_points(xyz_coords, topk_idx)
        point_features_bnc = point_features.transpose(1, 2).contiguous()
        topk_features = index_points(point_features_bnc, topk_idx)
        pooled_xyz = torch.sum(topk_xyz * attention.unsqueeze(-1), dim=2)
        pooled_feature = torch.sum(topk_features * attention.unsqueeze(-1), dim=2)
        residual_input = torch.cat([pooled_feature, pooled_xyz], dim=2)

        B, L, _ = residual_input.shape
        residual = self.coord_residual_mlp(residual_input.view(B * L, -1)).view(B, L, 3)

        limit = self.current_residual_limit_norm
        if limit is None:
            limit = float(getattr(self.args, 'hm_attn_residual_max_norm', 0.0))
        if limit and limit > 0.0:
            residual = float(limit) * torch.tanh(residual)

        final_coords = pooled_xyz + residual
        self.latest_hm_attn_pooled_xyz = pooled_xyz.detach()
        self.latest_hm_attn_residual = residual.detach()
        self.latest_hm_attn_final_coords = final_coords.detach()
        return final_coords

    def _heatmap_attention_feature_residual_coords(self, xyz_coords, point_features, heatmap_logits):
        attention = torch.sigmoid(heatmap_logits)
        attention = attention / (attention.sum(dim=2, keepdim=True) + 1e-6)

        pooled_xyz = torch.bmm(attention, xyz_coords)
        pooled_feature = torch.bmm(attention, point_features.transpose(1, 2).contiguous())

        B, L, _ = pooled_feature.shape
        residual = self.coord_feature_residual_mlp(pooled_feature.view(B * L, -1)).view(B, L, 3)

        limit = self.current_residual_limit_norm
        if limit is None:
            limit = float(getattr(self.args, 'hm_attn_residual_max_norm', 0.0))
        if limit and limit > 0.0:
            residual = float(limit) * torch.tanh(residual)

        final_coords = pooled_xyz + residual
        self.latest_hm_attn_pooled_xyz = pooled_xyz.detach()
        self.latest_hm_attn_residual = residual.detach()
        self.latest_hm_attn_final_coords = final_coords.detach()
        return final_coords

    def _direct_regression_coords(self, point_features):
        global_feature = torch.max(point_features, dim=2)[0]
        coords = self.head_linear(global_feature)
        return coords.view(point_features.size(0), self.landmark_num, 3)

    def forward(self, x, prior_hints=None):
        B, C_in, N_in = x.shape
        device = x.device
        self.latest_stage_hm_indices = None
        self.latest_main_heatmap_logits = None
        self.latest_sem_heatmap_logits = None
        self.latest_hm_attn_pooled_xyz = None
        self.latest_hm_attn_residual = None
        self.latest_hm_attn_final_coords = None

        # ==============================================================================
        # 🌟 2. [Ablation] 3지 선다 옵션에 따른 피처 주입 전처리
        # ==============================================================================
        if self.injection_type == 'none':
            prior_hints = None
            
        if prior_hints is not None and isinstance(prior_hints, list):
            if self.injection_type == 'raw':
                upsampled_hints = []
                target_N = prior_hints[0].shape[2] 
                
                for hint in prior_hints:
                    if hint.shape[2] != target_N:
                        hint_up = F.interpolate(hint, size=target_N, mode='nearest')
                        upsampled_hints.append(hint_up)
                    else:
                        upsampled_hints.append(hint)
                        
                prior_hints = torch.cat(upsampled_hints, dim=1) 
                
            elif self.injection_type == 'compressed':
                pass 
        # ==============================================================================

        # 🌟 3. [VRAM 최적화 & PyTorch3D 초고속화] k-NN 인덱스 추출
        xyz_coords = x[:, :3, :].permute(0, 2, 1).contiguous() 
        cur_xyz = xyz_coords
        down_knn_list, up_idx_list = [], []
        stage_hm_indices = [torch.arange(N_in, dtype=torch.long, device=device).view(1, N_in).repeat(B, 1)]
        cur_hm_indices = stage_hm_indices[0]
        downsample_method = getattr(self.args, 'stage_downsample_method', 'fps').lower()
        stage_grid_sizes = _parse_float_list(getattr(self.args, 'stage_grid_sizes', ''))
        stage_grid_search_iters = getattr(self.args, 'stage_grid_search_iters', 8)
        
        with torch.no_grad(): # VRAM 누수 원천 차단
            for i in range(len(self.args.depths)):
                
                # 🚀 최적화 포인트 1: 현재 해상도에서의 주변 이웃 찾기 (topk)
                if USE_PYTORCH3D_KNN:
                    _, knn_idx, _ = knn_points(cur_xyz, cur_xyz, K=self.args.ks[i])
                else:
                    dist = torch.cdist(cur_xyz, cur_xyz)
                    knn_idx = torch.topk(dist, self.args.ks[i], dim=-1, largest=False)[1]
                    
                down_knn_list.append(knn_idx)
                
                # 다음 층(Downsampling)으로 넘어갈 준비
                if i < len(self.args.depths) - 1:
                    next_points = self.args.npoints[i]
                    if downsample_method == 'grid':
                        grid_size = stage_grid_sizes[i] if i < len(stage_grid_sizes) else 0.0
                        down_idx = grid_subsample_fixed_count(
                            cur_xyz,
                            next_points,
                            grid_size=grid_size,
                            search_iters=stage_grid_search_iters,
                        )
                    elif downsample_method == 'fps':
                        down_idx = farthest_point_sample(cur_xyz, next_points)
                    else:
                        raise ValueError(f"Unknown stage_downsample_method: {downsample_method}")
                    down_knn_list.append(down_idx) 
                    cur_hm_indices = torch.gather(cur_hm_indices, 1, down_idx)
                    stage_hm_indices.append(cur_hm_indices)
                    
                    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(-1, 1).repeat(1, next_points)
                    next_xyz = cur_xyz[batch_indices, down_idx, :]
                    
                    # 🚀 최적화 포인트 2: 다운샘플링된 점들과 가장 가까운 원본 점 찾기 (argmin)
                    if USE_PYTORCH3D_KNN:
                        # K=1로 설정하여 가장 가까운 점 1개만 초고속 추출
                        _, up_idx_knn, _ = knn_points(cur_xyz, next_xyz, K=1)
                        up_idx = up_idx_knn.squeeze(-1) # (B, N, 1) -> (B, N)
                    else:
                        dist_up = torch.cdist(cur_xyz, next_xyz)
                        up_idx = torch.argmin(dist_up, dim=2)
                        
                    up_idx_list.append(up_idx)
                    cur_xyz = next_xyz
                
        down_knn_list = down_knn_list[::-1]
        indices = up_idx_list + down_knn_list
        self.latest_stage_hm_indices = stage_hm_indices
        
        # 4. 백본 통과
        in_features = x.permute(0, 2, 1).contiguous()
        out = self.model(xyz=xyz_coords, x=in_features, indices=indices, prior_hints=prior_hints)
        
        dense_features = out[0] if isinstance(out, tuple) else out
        if dense_features.shape[1] == N_in:
            dense_features = dense_features.permute(0, 2, 1).contiguous() 

        # 2,048점 -> 8,192점 복원
        if dense_features.shape[2] != N_in:
            dense_features = index_points(dense_features.permute(0, 2, 1), up_idx_list[0]).permute(0, 2, 1)

        # 5. 최종 결합
        xyz_full = xyz_coords.permute(0, 2, 1).contiguous() 
        fused_features = torch.cat([dense_features, xyz_full], dim=1) 
        
        # 6. 회귀 헤드 (Regression Head)
        x_fused = self.head_conv(fused_features) 
        readout_mode = getattr(self.args, 'train_coord_readout', 'topk').lower()
        if readout_mode == 'direct_regression':
            coords = self._direct_regression_coords(x_fused)
            spa_loss = out[1] if isinstance(out, tuple) else torch.tensor(0.0).to(device)
            self.latest_sem_heatmap_logits = []
            return coords, spa_loss, []

        main_heatmap_logits = self.main_heatmap_head(x_fused)
        self.latest_main_heatmap_logits = main_heatmap_logits
        main_heatmap = torch.sigmoid(main_heatmap_logits)
        # Coordinates are always derived from the main heatmap. The linear head is
        # kept in the module for checkpoint compatibility, but it is not used for
        # DeepPA coordinate supervision/evaluation.
        k_val = getattr(self.args, 'regression_point_num', 10)
        if readout_mode == 'heatmap_attn_residual':
            coords = self._heatmap_attention_residual_coords(xyz_coords, x_fused, main_heatmap_logits)
        elif readout_mode == 'topk_heatmap_residual':
            coords = self._heatmap_topk_residual_coords(xyz_coords, x_fused, main_heatmap_logits)
        elif readout_mode == 'heatmap_attn_residual_feature_only':
            coords = self._heatmap_attention_feature_residual_coords(xyz_coords, x_fused, main_heatmap_logits)
        elif readout_mode == 'sigmoid_xyz_pool':
            coords = self._heatmap_sigmoid_xyz_pool_coords(xyz_coords, main_heatmap_logits)
        else:
            coords = get_differentiable_coords(xyz_coords, main_heatmap, k=k_val)
        
        # 🌟 7. Train/Eval 상관없이 무조건 튜플 통일 반환
        spa_loss = out[1] if isinstance(out, tuple) else torch.tensor(0.0).to(device)
        raw_sem_list = out[2] if isinstance(out, tuple) and len(out) > 2 else []
        if torch.is_tensor(raw_sem_list):
            raw_sem_list = [raw_sem_list]
        else:
            raw_sem_list = list(raw_sem_list)
        sem_list = [torch.sigmoid(s) for s in raw_sem_list] if len(raw_sem_list) > 0 else []
        sem_list.append(main_heatmap)
        self.latest_sem_heatmap_logits = raw_sem_list + [main_heatmap_logits]
        
        return coords, spa_loss, sem_list
