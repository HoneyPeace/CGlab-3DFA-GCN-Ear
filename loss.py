# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: loss.py
# @Description:
# [S2G 3D 랜드마크 탐지를 위한 순수 오차 계산 모듈]
# 1. 확률 공간: AdaptiveWingLoss (L_sem)
# 2. 물리 공간: Focal L1 (L_crd) + CurvatureSurface (L_srf) + Structural (L_str)
# 3. 계층적 힌트 분리: DeepPA_HierarchicalHeatmapLoss (L_main, L_aux)
# ※ 주의: 가중치 스케줄링(Gating)은 loss_controller.py에서 전담하므로 여기선 배제됨.
# ==============================================================================

import torch
from torch import nn
import torch.nn.functional as F

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

class SoftmaxHeatmapDistributionLoss(nn.Module):
    """
    Heatmap distribution loss over points.

    pred_logits: [B, L, N], raw heatmap logits before sigmoid
    target: [B, L, N], GT heatmap values normalized to a point-wise distribution
    """
    def __init__(self, temperature=1.0, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, pred_logits, target):
        temperature = max(float(self.temperature), 1e-6)
        log_probs = F.log_softmax(pred_logits / temperature, dim=2)
        target_probs = target / target.sum(dim=2, keepdim=True).clamp_min(self.eps)
        return -(target_probs.detach() * log_probs).sum(dim=2).mean()

def focal_l1_loss(pred_coords, gt_coords, gamma=2.0):
    l1_errors = torch.norm(pred_coords - gt_coords, p=1, dim=-1)
    focal_weights = torch.pow(1.0 + l1_errors.detach(), gamma)
    weighted_loss = (l1_errors * focal_weights).mean()
    return weighted_loss

def compute_structural_loss(pred_coords, gt_coords):
    """
    [관계 로스] 랜드마크들 사이의 상호 거리를 비교하여 물리적 뼈대가 무너지는 것을 방지
    """
    pred_dist_matrix = torch.cdist(pred_coords, pred_coords)
    gt_dist_matrix = torch.cdist(gt_coords, gt_coords)
    return F.l1_loss(pred_dist_matrix, gt_dist_matrix)

def heatmap_distribution_structural_loss(points, heatmap_scores, gt_coords, temperature=1.0):
    """
    Compare landmark pairwise structure directly from heatmap distributions.

    points: [B, N, 3]
    heatmap_scores: [B, L, N], preferably logits before sigmoid
    gt_coords: [B, L, 3]

    This avoids an [N, N] point-pair matrix. For each landmark distribution p_l:
        mu_l = E[X_l]
        s_l = E[||X_l||^2]
    Then:
        E[||X_a - X_b||^2] = s_a + s_b - 2 * mu_a dot mu_b
    """
    temperature = max(float(temperature), 1e-6)
    weights = F.softmax(heatmap_scores / temperature, dim=2)

    pred_mean = torch.sum(points.unsqueeze(1) * weights.unsqueeze(-1), dim=2)
    point_sq_norm = torch.sum(points * points, dim=2)
    pred_sq_mean = torch.sum(weights * point_sq_norm.unsqueeze(1), dim=2)

    pred_pair_sq = (
        pred_sq_mean.unsqueeze(2)
        + pred_sq_mean.unsqueeze(1)
        - 2.0 * torch.bmm(pred_mean, pred_mean.transpose(1, 2))
    ).clamp_min(0.0)
    gt_pair_sq = torch.cdist(gt_coords, gt_coords).pow(2)

    num_landmarks = gt_coords.size(1)
    pair_mask = ~torch.eye(num_landmarks, device=gt_coords.device, dtype=torch.bool)
    return F.smooth_l1_loss(pred_pair_sq[:, pair_mask], gt_pair_sq[:, pair_mask])

# ---------------------------------------------------------
# 2. 3D 좌표 및 기하학적 연산 모듈
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

def get_softargmax_coords(points, heatmap_scores, temperature=1.0):
    # points: [B, N, 3], heatmap_scores: [B, L, N]
    temperature = max(float(temperature), 1e-6)
    weights = F.softmax(heatmap_scores / temperature, dim=2)
    return torch.sum(points.unsqueeze(1) * weights.unsqueeze(-1), dim=2)

def soft_ot_topk_mask(heatmap_scores, k=10, epsilon=0.1, iters=50):
    # Entropic OT relaxation of a top-k indicator.
    # Row masses are 1 per point, target masses are [k selected, N-k unselected].
    B, L, N = heatmap_scores.shape
    k = min(max(int(k), 1), N)
    if k >= N:
        return torch.ones_like(heatmap_scores)

    epsilon = max(float(epsilon), 1e-6)
    iters = max(int(iters), 1)

    selected_log_kernel = heatmap_scores / epsilon
    unselected_log_kernel = torch.zeros_like(selected_log_kernel)
    log_kernel = torch.stack([selected_log_kernel, unselected_log_kernel], dim=-1)

    log_mu = torch.zeros(B, L, N, device=heatmap_scores.device, dtype=heatmap_scores.dtype)
    log_nu_values = torch.tensor([float(k), float(N - k)], device=heatmap_scores.device, dtype=heatmap_scores.dtype)
    log_nu = torch.log(log_nu_values).view(1, 1, 2)

    log_u = torch.zeros_like(log_mu)
    log_v = torch.zeros(B, L, 2, device=heatmap_scores.device, dtype=heatmap_scores.dtype)
    for _ in range(iters):
        log_u = log_mu - torch.logsumexp(log_kernel + log_v.unsqueeze(2), dim=3)
        log_v = log_nu - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=2)

    log_transport = log_kernel + log_u.unsqueeze(-1) + log_v.unsqueeze(2)
    return torch.exp(log_transport[..., 0])

def get_soft_ot_topk_coords(points, heatmap_scores, k=10, epsilon=0.1, iters=50):
    # points: [B, N, 3], heatmap_scores: [B, L, N]
    selected_mass = soft_ot_topk_mask(heatmap_scores, k=k, epsilon=epsilon, iters=iters)
    weights = selected_mass / (selected_mass.sum(dim=2, keepdim=True) + 1e-6)
    return torch.sum(points.unsqueeze(1) * weights.unsqueeze(-1), dim=2)

def expected_distance_heatmap_loss(points, heatmap_scores, gt_coords, temperature=1.0):
    # Penalize probability mass assigned to points far from each GT landmark.
    temperature = max(float(temperature), 1e-6)
    weights = F.softmax(heatmap_scores / temperature, dim=2)
    distances = torch.cdist(gt_coords, points)
    return torch.sum(weights * distances, dim=2).mean()

def ranking_heatmap_loss(points, heatmap_scores, gt_coords, margin_scale=1.0, temperature=1.0):
    # Structured logit ranking: GT-nearest point should outrank distant points.
    temperature = max(float(temperature), 1e-6)
    distances = torch.cdist(gt_coords, points).detach()
    pos_idx = torch.argmin(distances, dim=2, keepdim=True)
    pos_scores = torch.gather(heatmap_scores, 2, pos_idx)
    margin = distances / (distances.amax(dim=2, keepdim=True) + 1e-6)
    logits = (heatmap_scores + float(margin_scale) * margin - pos_scores) / temperature
    return torch.logsumexp(logits, dim=2).mean()

def find_knn_points(pred_coords, points, k=10):
    """
    36개의 예측 랜드마크 주변의 포인트 클라우드 표면 정보를 탐색 (메모리 부담 적음)
    """
    dist_matrix = torch.cdist(pred_coords, points) 
    _, knn_indices = torch.topk(dist_matrix, k, dim=2, largest=False) 
    idx_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    points_expanded = points.unsqueeze(1).expand(-1, pred_coords.size(1), -1, -1)
    return torch.gather(points_expanded, 2, idx_expanded) 

class CurvatureSurfaceLoss(nn.Module):
    """
    [표면 로스] 점들이 3D 모델의 실제 표면(곡률, 법선벡터) 위에 안착하도록 유도
    """
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
# 3. 계층적 히트맵 오차 추출기
# ---------------------------------------------------------
def _soft_local_covariance(points, centers, sigma=0.0, k_for_auto=30, min_sigma=1e-4):
    # points: [B, N, 3], centers: [B, L, 3]
    dist = torch.cdist(centers, points)
    if float(sigma) > 0.0:
        sigma_tensor = torch.ones_like(dist[..., 0]) * float(sigma)
    else:
        k_for_auto = min(max(int(k_for_auto), 1), points.shape[1])
        with torch.no_grad():
            sigma_tensor = torch.topk(dist.detach(), k_for_auto, dim=2, largest=False).values[..., -1]
    sigma_tensor = sigma_tensor.clamp_min(float(min_sigma))
    logits = -(dist * dist) / (2.0 * sigma_tensor.unsqueeze(-1).pow(2))
    weights = F.softmax(logits, dim=2)
    mean = torch.sum(weights.unsqueeze(-1) * points.unsqueeze(1), dim=2)
    centered = points.unsqueeze(1) - mean.unsqueeze(2)
    cov = torch.einsum('bln,blni,blnj->blij', weights, centered, centered)
    eye = torch.eye(3, device=points.device, dtype=points.dtype).view(1, 1, 3, 3)
    return mean, cov + (eye * 1e-6)

class SoftLocalCurvatureSurfaceLoss(nn.Module):
    """
    Differentiable local surface loss using soft distance weights instead of
    hard pred-coordinate top-k neighborhood selection.
    """
    def __init__(self, k_p2p=5, k_curv=30, alpha=10.0, beta=1.0, sigma=0.0, min_sigma=1e-4):
        super().__init__()
        self.k_p2p, self.k_curv = k_p2p, k_curv
        self.alpha, self.beta = alpha, beta
        self.sigma = sigma
        self.min_sigma = min_sigma

    def forward(self, pred_coords, gt_coords, points, disable_norm=False):
        center_p2p_gt, cov_p2p_gt = _soft_local_covariance(
            points, gt_coords, self.sigma, self.k_p2p, self.min_sigma
        )
        _, eigvec_p2p_gt = torch.linalg.eigh(cov_p2p_gt)
        normal_gt = eigvec_p2p_gt[..., 0]

        _, cov_curv_pred = _soft_local_covariance(
            points, pred_coords, self.sigma, self.k_curv, self.min_sigma
        )
        _, cov_curv_gt = _soft_local_covariance(
            points, gt_coords, self.sigma, self.k_curv, self.min_sigma
        )
        eigval_pred, eigvec_pred = torch.linalg.eigh(cov_curv_pred)
        eigval_gt, eigvec_gt = torch.linalg.eigh(cov_curv_gt)

        c_pred = eigval_pred[..., 0] / (torch.sum(eigval_pred, dim=-1) + 1e-6)
        c_gt = (eigval_gt[..., 0] / (torch.sum(eigval_gt, dim=-1) + 1e-6)).detach()
        primary_dir_pred = eigvec_pred[..., 2]
        primary_dir_gt = eigvec_gt[..., 2].detach()
        cos_sim = torch.abs(torch.sum(primary_dir_pred * primary_dir_gt, dim=-1))

        diff_curvature = torch.abs(c_pred - c_gt)
        diff_direction = 1.0 - cos_sim
        p2p_distance = torch.abs(
            torch.sum((pred_coords - center_p2p_gt.detach()) * normal_gt.detach(), dim=-1)
        )

        if disable_norm:
            weight_multiplier = 1.0
        else:
            weight_multiplier = 1.0 + (self.alpha * diff_curvature) + (self.beta * diff_direction)

        loss_unified = (p2p_distance * weight_multiplier).mean()
        return loss_unified, p2p_distance.mean(), diff_curvature.mean(), diff_direction.mean()

class DeepPA_HierarchicalHeatmapLoss(nn.Module):
    """
    백본에서 출력된 여러 층의 히트맵(sem_list)을 받아,
    가장 마지막 층(Main)과 그 이전 중간층들(Aux)의 로스를 분리하여 반환합니다.
    """
    def __init__(self, criterion=None):
        super().__init__()
        self.criterion = criterion if criterion is not None else AdaptiveWingLoss()

    def _as_bln(self, pred, target_hm):
        if pred.shape[1] != target_hm.shape[1]:
            pred = pred.permute(0, 2, 1).contiguous()
        return pred

    def _gather_target_hm(self, target_hm, point_indices):
        gather_idx = point_indices.long().unsqueeze(1).expand(-1, target_hm.shape[1], -1)
        return torch.gather(target_hm, 2, gather_idx)

    def forward(self, sem_list, target_hm, stage_indices=None):
        device = target_hm.device
        L_main_hm = torch.tensor(0.0).to(device)
        L_aux_hm = torch.tensor(0.0).to(device)

        if not sem_list:
            return L_main_hm, L_aux_hm

        if stage_indices is not None and len(sem_list) > 1:
            aligned_sem_list = [self._as_bln(sp, target_hm) for sp in sem_list]
            main_pred = aligned_sem_list[-1]
            if main_pred.shape[2] == target_hm.shape[2]:
                L_main_hm = self.criterion(main_pred, target_hm)

            aux_losses = []
            for sp, point_indices in zip(aligned_sem_list[:-1], stage_indices):
                stage_target_hm = self._gather_target_hm(target_hm, point_indices)
                if sp.shape[2] == stage_target_hm.shape[2]:
                    aux_losses.append(self.criterion(sp, stage_target_hm))

            if aux_losses:
                L_aux_hm = sum(aux_losses) / len(aux_losses)

            return L_main_hm, L_aux_hm

        safe_sem_list = []
        for sp in sem_list:
            # 텐서 형태 (B, N, C) vs (B, C, N) 정렬
            sp = self._as_bln(sp, target_hm)
            if sp.shape[2] == target_hm.shape[2]: 
                safe_sem_list.append(sp)

        if len(safe_sem_list) > 0:
            L_main_hm = self.criterion(safe_sem_list[-1], target_hm)
            if len(safe_sem_list) > 1:
                aux_preds = safe_sem_list[:-1]
                L_aux_hm = sum([self.criterion(p, target_hm) for p in aux_preds]) / len(aux_preds)

        return L_main_hm, L_aux_hm
