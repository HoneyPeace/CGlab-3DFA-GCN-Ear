import torch
from torch import nn
import torch.nn.functional as F

# ==============================================================================
# [0.47mm 3D 랜드마크 탐지를 위한 하이브리드 로스 모듈]
# 1. HDS 모의고사용: AdaptiveWingLoss (L_sem), ChamferSpatialLoss (L_spa)
# 2. 최종 정밀 타격용: Focal L1 (L_coord) + Structural (L_struct) + CurvatureSurface (L_surf)
# ==============================================================================

class AdaptiveWingLoss(nn.Module):
    """
    [L_sem: 의미론적 세그멘테이션 로스]
    - 120층 백본의 해부학적 인지 능력을 기르는 HDS 모의고사 로스 (CVPR Eq. 5 대응)
    """
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
        
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

# =====================================================================
# 공통 유틸리티 (KNN, PCA 기반 방향/곡률 추출)
# =====================================================================
def find_knn_points(pred_coords, points, k=10):
    dist_matrix = torch.cdist(pred_coords, points) 
    _, knn_indices = torch.topk(dist_matrix, k, dim=2, largest=False) 
    idx_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    points_expanded = points.unsqueeze(1).expand(-1, pred_coords.size(1), -1, -1)
    return torch.gather(points_expanded, 2, idx_expanded) 

# =====================================================================
# 🌟 최종 예측용 3대 기하학 로스 (L_Pred)
# =====================================================================

def focal_l1_loss(pred_coords, gt_coords, gamma=2.0):
    """
    [L_Focal: 표준 Focal L1 로스]
    - (변경점): 모호한 Dynamic 임계값을 제거하고, 수렴이 입증된 Standard Focal L1 채택
    - 오차가 클수록(어려운 랜드마크일수록) 가중치 (1+Error)^gamma 를 곱해 집중 타격
    """
    l1_errors = torch.norm(pred_coords - gt_coords, p=1, dim=-1)
    focal_weights = torch.pow(1.0 + l1_errors.detach(), gamma)
    weighted_loss = (l1_errors * focal_weights).mean()
    return weighted_loss

def compute_structural_loss(pred_coords, gt_coords):
    """
    [L_Struct: 해부학적 형태 보존 로스]
    - 36개 랜드마크 간의 '상대적 거리(뼈대 비율)'가 무너지지 않도록 그래프 구조 강제
    """
    pred_dist_matrix = torch.cdist(pred_coords, pred_coords)
    gt_dist_matrix = torch.cdist(gt_coords, gt_coords)
    return F.l1_loss(pred_dist_matrix, gt_dist_matrix)

class CurvatureSurfaceLoss(nn.Module):
    """
    [L_Surface: 하이브리드 곡률 표면 밀착 로스 - 논문의 SOTA 공헌점]
    - 수직 거리(P2P)를 기반으로 하되, 곡률과 방향 오차를 '독립적인 덧셈 패널티'로 부과
    """
    def __init__(self, k_p2p=5, k_curv=30, alpha=10.0, beta=1.0):
        super().__init__()
        self.k_p2p, self.k_curv = k_p2p, k_curv
        self.alpha, self.beta = alpha, beta

    def forward(self, pred_coords, gt_coords, points, disable_norm=False):
        # 1. P2P 투영용 법선 추출 (k=5, 좁은 반경)
        knn_p2p_gt = find_knn_points(gt_coords, points, self.k_p2p)
        center_p2p_gt = knn_p2p_gt.mean(dim=2, keepdim=True)
        cov_p2p_gt = torch.matmul((knn_p2p_gt - center_p2p_gt).transpose(2, 3), (knn_p2p_gt - center_p2p_gt))
        _, eigvec_p2p_gt = torch.linalg.eigh(cov_p2p_gt)
        normal_gt = eigvec_p2p_gt[..., 0] 

        # 2. 곡률/방향 분석용 PCA (k=30, 넓은 반경)
        knn_curv_pred = find_knn_points(pred_coords, points, self.k_curv)
        knn_curv_gt = find_knn_points(gt_coords, points, self.k_curv)
        cov_curv_pred = torch.matmul((knn_curv_pred - knn_curv_pred.mean(dim=2, keepdim=True)).transpose(2, 3), (knn_curv_pred - knn_curv_pred.mean(dim=2, keepdim=True)))
        cov_curv_gt = torch.matmul((knn_curv_gt - knn_curv_gt.mean(dim=2, keepdim=True)).transpose(2, 3), (knn_curv_gt - knn_curv_gt.mean(dim=2, keepdim=True)))
        eigval_pred, eigvec_pred = torch.linalg.eigh(cov_curv_pred)
        eigval_gt, eigvec_gt = torch.linalg.eigh(cov_curv_gt)

        # 3. 곡률 차이 및 방향 오차 계산
        c_pred = eigval_pred[..., 0] / (torch.sum(eigval_pred, dim=-1) + 1e-6)
        c_gt = (eigval_gt[..., 0] / (torch.sum(eigval_gt, dim=-1) + 1e-6)).detach()
        primary_dir_pred = eigvec_pred[..., 2] 
        primary_dir_gt = eigvec_gt[..., 2].detach() 
        cos_sim = torch.abs(torch.sum(primary_dir_pred * primary_dir_gt, dim=-1))

        # 4. 독립적 덧셈 패널티 기반 통합 로스 계산 (Crucial Update)
        diff_curvature = torch.abs(c_pred - c_gt)
        diff_direction = 1.0 - cos_sim
        p2p_distance = torch.abs(torch.sum((pred_coords - center_p2p_gt.squeeze(2).detach()) * normal_gt.detach(), dim=-1))

        if disable_norm:
            weight_multiplier = 1.0 
        else:
            # [디펜스 포인트]: 곱셈 교차 간섭 방지를 위한 덧셈(Linear) 패널티
            weight_multiplier = 1.0 + (self.alpha * diff_curvature) + (self.beta * diff_direction)
        
        loss_unified = (p2p_distance * weight_multiplier).mean()

        return loss_unified, p2p_distance.mean(), diff_curvature.mean(), diff_direction.mean()
    
# =====================================================================
# 🌟 [초정밀 보간법] Soft-Argmax 좌표 추출기 (1.70mm 돌파의 핵심)
# =====================================================================
def get_differentiable_coords(points, heatmaps, k=10):
    """
    [Soft-Argmax 기반 초정밀 3D 좌표 추출]
    가장 핫한 1개의 점만 고르는 것(2.5mm의 한계)이 아니라,
    주변 상위 k개의 점을 모두 찾아 에너지를 가중치로 삼아 허공의 '무게중심'을 찍습니다.
    
    - points: (B, N, 3) 포인트 클라우드 좌표
    - heatmaps: (B, K_lm, N) 랜드마크 히트맵 확률 또는 로짓
    - k: 보간에 사용할 주변 점의 개수 (보통 10~20)
    """
    B, K_lm, N = heatmaps.shape
    
    # 1. 각 랜드마크별로 가장 에너지가 높은 상위 k개의 값과 인덱스 추출
    topk_vals, topk_idx = torch.topk(heatmaps, k, dim=2) # (B, K_lm, k)
    
    # 2. 상위 k개 점의 3D XYZ 좌표를 추출
    topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3) # (B, K_lm, k, 3)
    points_expanded = points.unsqueeze(1).expand(-1, K_lm, -1, -1)     # (B, K_lm, N, 3)
    topk_coords = torch.gather(points_expanded, 2, topk_idx_expanded)  # (B, K_lm, k, 3)
    
    # 3. 추출된 k개의 에너지를 가중치(Weight)로 변환 (Softmax를 통해 총합 1로 맞춤)
    weights = F.softmax(topk_vals, dim=2) # (B, K_lm, k)
    
    # 4. 가중치를 반영하여 k개 점들의 '무게중심(Center of Mass)' 3D 좌표 계산
    pred_coords = torch.sum(topk_coords * weights.unsqueeze(-1), dim=2) # (B, K_lm, 3)
    
    return pred_coords