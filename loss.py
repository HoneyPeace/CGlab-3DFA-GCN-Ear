# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: loss.py
import torch
from torch import nn
import torch.nn.functional as F

# ==============================================================================
# [0.47mm 3D 랜드마크 탐지를 위한 초정밀 하이브리드 로스 시스템]
# 1. 확률 공간 학습: AdaptiveWingLoss (L_sem / L_spa 융합)
# 2. 좌표 보간 추출: Soft-Argmax (get_differentiable_coords)
# 3. 물리 공간 학습: Focal L1 + CurvatureSurface
# 4. HDS 히트맵 분리: DeepPA_HierarchicalHeatmapLoss
# 5. [신규] 최종 통합 매니저: UnifiedLandmarkLossManager (가중치 자동 스케줄링)
# ==============================================================================

# ---------------------------------------------------------
# 1. 기본 손실 함수 (Base Criterions)
# ---------------------------------------------------------
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega, self.theta, self.epsilon, self.alpha = omega, theta, epsilon, alpha

    def forward(self, pred, target):
        delta_y = (target - pred).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = target[delta_y < self.theta]
        y2 = target[delta_y >= self.theta]
        
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1))
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.omega, self.alpha - y2))
        loss2 = A * delta_y2 - C
        
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2) + 1e-6)

def focal_l1_loss(pred_coords, gt_coords, gamma=2.0):
    l1_errors = torch.norm(pred_coords - gt_coords, p=1, dim=-1)
    focal_weights = torch.pow(1.0 + l1_errors.detach(), gamma)
    weighted_loss = (l1_errors * focal_weights).mean()
    return weighted_loss

def compute_structural_loss(pred_coords, gt_coords):
    pred_dist_matrix = torch.cdist(pred_coords, pred_coords)
    gt_dist_matrix = torch.cdist(gt_coords, gt_coords)
    return F.l1_loss(pred_dist_matrix, gt_dist_matrix)

# ---------------------------------------------------------
# 2. 3D 좌표 및 기하학적 연산 유틸리티
# ---------------------------------------------------------
def get_differentiable_coords(points, heatmaps, k=10):
    B, K_lm, N = heatmaps.shape
    topk_vals, topk_idx = torch.topk(heatmaps, k, dim=2) 
    topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3) 
    points_expanded = points.unsqueeze(1).expand(-1, K_lm, -1, -1)     
    topk_coords = torch.gather(points_expanded, 2, topk_idx_expanded)  
    weights = F.softmax(topk_vals, dim=2) 
    pred_coords = torch.sum(topk_coords * weights.unsqueeze(-1), dim=2) 
    return pred_coords

def find_knn_points(pred_coords, points, k=10):
    dist_matrix = torch.cdist(pred_coords, points) 
    _, knn_indices = torch.topk(dist_matrix, k, dim=2, largest=False) 
    idx_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    points_expanded = points.unsqueeze(1).expand(-1, pred_coords.size(1), -1, -1)
    return torch.gather(points_expanded, 2, idx_expanded) 

class CurvatureSurfaceLoss(nn.Module):
    def __init__(self, k_p2p=5, k_curv=30, alpha=10.0, beta=1.0):
        super().__init__()
        self.k_p2p, self.k_curv = k_p2p, k_curv
        self.alpha, self.beta = alpha, beta

    def forward(self, pred_coords, gt_coords, points, disable_norm=False):
        knn_p2p_gt = find_knn_points(gt_coords, points, self.k_p2p)
        center_p2p_gt = knn_p2p_gt.mean(dim=2, keepdim=True)
        cov_p2p_gt = torch.matmul((knn_p2p_gt - center_p2p_gt).transpose(2, 3), (knn_p2p_gt - center_p2p_gt))
        _, eigvec_p2p_gt = torch.linalg.eigh(cov_p2p_gt)
        normal_gt = eigvec_p2p_gt[..., 0] 

        knn_curv_pred = find_knn_points(pred_coords, points, self.k_curv)
        knn_curv_gt = find_knn_points(gt_coords, points, self.k_curv)
        cov_curv_pred = torch.matmul((knn_curv_pred - knn_curv_pred.mean(dim=2, keepdim=True)).transpose(2, 3), (knn_curv_pred - knn_curv_pred.mean(dim=2, keepdim=True)))
        cov_curv_gt = torch.matmul((knn_curv_gt - knn_curv_gt.mean(dim=2, keepdim=True)).transpose(2, 3), (knn_curv_gt - knn_curv_gt.mean(dim=2, keepdim=True)))
        eigval_pred, eigvec_pred = torch.linalg.eigh(cov_curv_pred)
        eigval_gt, eigvec_gt = torch.linalg.eigh(cov_curv_gt)

        c_pred = eigval_pred[..., 0] / (torch.sum(eigval_pred, dim=-1) + 1e-6)
        c_gt = (eigval_gt[..., 0] / (torch.sum(eigval_gt, dim=-1) + 1e-6)).detach()
        primary_dir_pred = eigvec_pred[..., 2] 
        primary_dir_gt = eigvec_gt[..., 2].detach() 
        cos_sim = torch.abs(torch.sum(primary_dir_pred * primary_dir_gt, dim=-1))

        diff_curvature = torch.abs(c_pred - c_gt)
        diff_direction = 1.0 - cos_sim
        p2p_distance = torch.abs(torch.sum((pred_coords - center_p2p_gt.squeeze(2).detach()) * normal_gt.detach(), dim=-1))

        if disable_norm:
            weight_multiplier = 1.0 
        else:
            weight_multiplier = 1.0 + (self.alpha * diff_curvature) + (self.beta * diff_direction)
        
        loss_unified = (p2p_distance * weight_multiplier).mean()
        return loss_unified, p2p_distance.mean(), diff_curvature.mean(), diff_direction.mean()

# ---------------------------------------------------------
# 3. 로스 제어 및 매니저 (Gating & HDS)
# ---------------------------------------------------------
class S2GGatingManager:
    def __init__(self, tau=0.65, beta=15.0):
        self.tau = tau   
        self.beta = beta 

    def get_geometric_weight(self, pred_heatmap, gt_heatmap):
        pred_flat = pred_heatmap.view(-1, pred_heatmap.shape[-1])
        gt_flat = gt_heatmap.view(-1, gt_heatmap.shape[-1])
        sim = F.cosine_similarity(pred_flat, gt_flat, dim=-1).mean()
        w_geom = 1.0 / (1.0 + torch.exp(-self.beta * (sim - self.tau)))
        return torch.clamp(w_geom, min=0.0, max=1.0)

class DeepPA_HierarchicalHeatmapLoss(nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        self.criterion = criterion if criterion is not None else AdaptiveWingLoss()

    def forward(self, sem_list, target_hm):
        device = target_hm.device
        L_main_hm = torch.tensor(0.0).to(device)
        L_aux_hm = torch.tensor(0.0).to(device)

        if not sem_list:
            return L_main_hm, L_aux_hm

        safe_sem_list = []
        for sp in sem_list:
            if sp.shape[1] != target_hm.shape[1]: 
                sp = sp.permute(0, 2, 1).contiguous()
            if sp.shape[2] == target_hm.shape[2]: 
                safe_sem_list.append(sp)

        if len(safe_sem_list) > 0:
            L_main_hm = self.criterion(safe_sem_list[-1], target_hm)
            if len(safe_sem_list) > 1:
                aux_preds = safe_sem_list[:-1]
                L_aux_hm = sum([self.criterion(p, target_hm) for p in aux_preds]) / len(aux_preds)

        return L_main_hm, L_aux_hm

# =====================================================================
# 🌟 [신규] 학습 루프용 마스터 로스 매니저 (가중치 자동화)
# =====================================================================
class UnifiedLandmarkLossManager(nn.Module):
    """
    모든 로스를 통합 관리하고, 에폭에 따른 HDS(Aux) 가중치 감쇠를 자동 처리합니다.
    """
    def __init__(self, use_aux=True, aux_min_weight=0.05, aux_decay_epochs=30):
        super().__init__()
        self.heatmap_loss = DeepPA_HierarchicalHeatmapLoss(criterion=AdaptiveWingLoss())
        self.curv_loss = CurvatureSurfaceLoss()
        self.gating = S2GGatingManager(tau=0.65)
        
        self.use_aux = use_aux
        self.aux_min_weight = aux_min_weight
        self.aux_decay_epochs = aux_decay_epochs # 이 에폭이 지나면 최소 가중치로 고정됨

    def get_dynamic_aux_weight(self, current_epoch):
        """논문의 HDS 전략처럼 초기에는 강하게, 갈수록 약하게(최소 가중치로) 조절합니다."""
        if not self.use_aux:
            return 0.0
        
        # 선형 감쇠 (Linear Decay): 0에폭일때 1.0 -> aux_decay_epochs일때 aux_min_weight
        decay_rate = max(0.0, 1.0 - (current_epoch / self.aux_decay_epochs))
        weight = self.aux_min_weight + (1.0 - self.aux_min_weight) * decay_rate
        return weight

    def forward(self, sem_list, target_hm, pred_coords, target_coords, points, current_epoch):
        # 1. 히트맵 로스 (Main & Aux)
        L_main_hm, L_aux_hm = self.heatmap_loss(sem_list, target_hm)
        
        # Aux 가중치 스케줄링 적용
        w_aux = self.get_dynamic_aux_weight(current_epoch)
        L_heatmap_total = L_main_hm + (w_aux * L_aux_hm)
        
        # 2. 물리적 좌표 로스 (Focal L1)
        L_focal = focal_l1_loss(pred_coords, target_coords)
        
        # 3. 구조적/기하학적 로스 (Gating 적용)
        # Frozen 모델 실험 시 메인 히트맵이 target_hm을 사용합니다.
        main_heatmap_pred = sem_list[-1] if sem_list else target_hm
        w_geom = self.gating.get_geometric_weight(main_heatmap_pred, target_hm)
        L_curv_unified, _, _, _ = self.curv_loss(pred_coords, target_coords, points)
        
        L_geom_total = L_focal + (w_geom * L_curv_unified)
        
        # 4. 최종 Total Loss (히트맵 덩어리 + 물리적 좌표 덩어리)
        total_loss = L_heatmap_total + L_geom_total
        
        # 로깅(Logging)을 위해 딕셔너리로 세부 로스 반환
        loss_dict = {
            'total_loss': total_loss,
            'L_main_hm': L_main_hm,
            'L_aux_hm': L_aux_hm,
            'w_aux': torch.tensor(w_aux), # 현재 에폭의 Aux 가중치 확인용
            'L_focal': L_focal,
            'L_curv': L_curv_unified,
            'w_geom': w_geom
        }
        
        return total_loss, loss_dict