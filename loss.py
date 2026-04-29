import torch
from torch import nn
import torch.nn.functional as F

# ==============================================================================
# [0.47mm 3D 랜드마크 탐지를 위한 하이브리드 로스 시스템]
# 1. 확률 공간 학습: AdaptiveWingLoss (L_sem)
# 2. 좌표 보간: Soft-Argmax (get_differentiable_coords, k=10)
# 3. 물리 공간 학습: Focal L1 + Structural + CurvatureSurface (k_p2p=5, k_curv=30)
# 4. 자율 융합 제어: S2GGatingManager (Adaptive Sigmoid Gating, tau=0.65)
# ==============================================================================

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

def focal_l1_loss(pred_coords, gt_coords, gamma=2.0):
    l1_errors = torch.norm(pred_coords - gt_coords, p=1, dim=-1)
    focal_weights = torch.pow(1.0 + l1_errors.detach(), gamma)
    weighted_loss = (l1_errors * focal_weights).mean()
    return weighted_loss

def compute_structural_loss(pred_coords, gt_coords):
    pred_dist_matrix = torch.cdist(pred_coords, pred_coords)
    gt_dist_matrix = torch.cdist(gt_coords, gt_coords)
    return F.l1_loss(pred_dist_matrix, gt_dist_matrix)

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

# =====================================================================
# 🌟 자율 로스 게이팅 매니저 (선택적 사용을 위해 구조 보존)
# =====================================================================
class S2GGatingManager:
    """
    [자율 로스 게이팅 매니저]
    - 예측 히트맵과 정답 히트맵 간의 코사인 유사도(sim)를 평가하여, 
      모델이 공간적 위치를 어느 정도 파악했을 때만 물리적 구조 로스(Geometric Loss)가 
      활성화되도록 가중치를 부드럽게(Sigmoid) 열어줍니다.
    """
    def __init__(self, tau=0.65, beta=15.0):
        self.tau = tau   # 문이 열리는 기준 임계점
        self.beta = beta # 개방 속도의 날카로움

    def get_geometric_weight(self, pred_heatmap, gt_heatmap):
        pred_flat = pred_heatmap.view(-1, pred_heatmap.shape[-1])
        gt_flat = gt_heatmap.view(-1, gt_heatmap.shape[-1])
        
        sim = F.cosine_similarity(pred_flat, gt_flat, dim=-1).mean()
        w_geom = 1.0 / (1.0 + torch.exp(-self.beta * (sim - self.tau)))
        
        return torch.clamp(w_geom, min=0.0, max=1.0)