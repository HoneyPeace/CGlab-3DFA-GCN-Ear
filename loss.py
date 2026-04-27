import torch
from torch import nn
import torch.nn.functional as F

# ==============================================================================
# [0.47mm 3D 랜드마크 탐지를 위한 S2G 하이브리드 로스 시스템]
# 1. 확률 공간 학습: AdaptiveWingLoss (L_sem)
# 2. 좌표 보간: Soft-Argmax (get_differentiable_coords, k=10)
# 3. 물리 공간 학습: Focal L1 + Structural + CurvatureSurface (k_p2p=5, k_curv=30)
# 4. 자율 융합 제어: S2GGatingManager (Adaptive Sigmoid Gating, tau=0.65)
# ==============================================================================

class AdaptiveWingLoss(nn.Module):
    """
    [L_sem: 의미론적 세그멘테이션 로스]
    - 120층 백본의 해부학적 인지 능력을 기르는 HDS 모의고사 로스
    - 정답 근처에서 페널티를 높여 히트맵이 뾰족한 가우시안 분포를 갖도록 유도
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
        
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2) + 1e-6)

# =====================================================================
# 🌟 [초정밀 보간법] Soft-Argmax 좌표 추출기 (1.70mm 돌파의 핵심)
# =====================================================================
def get_differentiable_coords(points, heatmaps, k=10):
    """
    [Soft-Argmax 기반 초정밀 3D 좌표 추출]
    - 주변 상위 k=10개의 점을 찾아 에너지를 가중치로 삼아 허공의 '무게중심' 좌표 생성
    - points: (B, N, 3), heatmaps: (B, K_lm, N)
    """
    B, K_lm, N = heatmaps.shape
    
    # 1. 각 랜드마크별로 가장 에너지가 높은 상위 k개의 값과 인덱스 추출
    topk_vals, topk_idx = torch.topk(heatmaps, k, dim=2) 
    
    # 2. 상위 k개 점의 3D XYZ 좌표를 추출
    topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3) 
    points_expanded = points.unsqueeze(1).expand(-1, K_lm, -1, -1)     
    topk_coords = torch.gather(points_expanded, 2, topk_idx_expanded)  
    
    # 3. 추출된 k개의 에너지를 가중치(Weight)로 변환
    weights = F.softmax(topk_vals, dim=2) 
    
    # 4. 가중치를 반영하여 k개 점들의 '무게중심' 3D 좌표 계산
    pred_coords = torch.sum(topk_coords * weights.unsqueeze(-1), dim=2) 
    
    return pred_coords

# =====================================================================
# 공통 유틸리티 (KNN 기반 검색)
# =====================================================================
def find_knn_points(pred_coords, points, k=10):
    """ 예측된 좌표와 가장 가까운 표면의 점 k개를 찾음 """
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
    - 오차가 클수록(어려운 랜드마크일수록) 가중치를 높여 집중 타격
    """
    l1_errors = torch.norm(pred_coords - gt_coords, p=1, dim=-1)
    focal_weights = torch.pow(1.0 + l1_errors.detach(), gamma)
    weighted_loss = (l1_errors * focal_weights).mean()
    return weighted_loss

def compute_structural_loss(pred_coords, gt_coords):
    """
    [L_Struct: 해부학적 형태 보존 로스]
    - 랜드마크 간의 상대적 거리(뼈대 비율) 오차를 계산
    """
    pred_dist_matrix = torch.cdist(pred_coords, pred_coords)
    gt_dist_matrix = torch.cdist(gt_coords, gt_coords)
    return F.l1_loss(pred_dist_matrix, gt_dist_matrix)

class CurvatureSurfaceLoss(nn.Module):
    """
    [L_Surface: 하이브리드 곡률 표면 밀착 로스]
    - 수직 거리(P2P, k=5)를 기반으로 곡률과 방향 오차(k=30)를 독립적으로 제어
    """
    def __init__(self, k_p2p=5, k_curv=30, alpha=10.0, beta=1.0):
        super().__init__()
        self.k_p2p, self.k_curv = k_p2p, k_curv
        self.alpha, self.beta = alpha, beta

    def forward(self, pred_coords, gt_coords, points, disable_norm=False):
        # 1. P2P 투영용 법선 추출 (k=5)
        knn_p2p_gt = find_knn_points(gt_coords, points, self.k_p2p)
        center_p2p_gt = knn_p2p_gt.mean(dim=2, keepdim=True)
        cov_p2p_gt = torch.matmul((knn_p2p_gt - center_p2p_gt).transpose(2, 3), (knn_p2p_gt - center_p2p_gt))
        _, eigvec_p2p_gt = torch.linalg.eigh(cov_p2p_gt)
        normal_gt = eigvec_p2p_gt[..., 0] 

        # 2. 곡률/방향 분석용 PCA (k=30)
        knn_curv_pred = find_knn_points(pred_coords, points, self.k_curv)
        knn_curv_gt = find_knn_points(gt_coords, points, self.k_curv)
        cov_curv_pred = torch.matmul((knn_curv_pred - knn_curv_pred.mean(dim=2, keepdim=True)).transpose(2, 3), (knn_curv_pred - knn_curv_pred.mean(dim=2, keepdim=True)))
        cov_curv_gt = torch.matmul((knn_curv_gt - knn_curv_gt.mean(dim=2, keepdim=True)).transpose(2, 3), (knn_curv_gt - knn_curv_gt.mean(dim=2, keepdim=True)))
        eigval_pred, eigvec_pred = torch.linalg.eigh(cov_curv_pred)
        eigval_gt, eigvec_gt = torch.linalg.eigh(cov_curv_gt)

        # 3. 곡률 및 방향 오차 계산
        c_pred = eigval_pred[..., 0] / (torch.sum(eigval_pred, dim=-1) + 1e-6)
        c_gt = (eigval_gt[..., 0] / (torch.sum(eigval_gt, dim=-1) + 1e-6)).detach()
        primary_dir_pred = eigvec_pred[..., 2] 
        primary_dir_gt = eigvec_gt[..., 2].detach() 
        cos_sim = torch.abs(torch.sum(primary_dir_pred * primary_dir_gt, dim=-1))

        # 4. 통합 로스 산출
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
# 🌟 [통합 제어] S2G 자율 로스 게이팅 (Adaptive Sigmoid Gating)
# =====================================================================
class S2GGatingManager:
    """
    [자율 로스 게이팅 매니저]
    - 예측 히트맵의 유사도(sim)를 평가하여 기하 로스의 밸브를 개방
    """
    def __init__(self, tau=0.65, beta=15.0):
        self.tau = tau   # 문이 열리는 기준 임계점
        self.beta = beta # 개방 속도의 날카로움

    def get_geometric_weight(self, pred_heatmap, gt_heatmap):
        """ 코사인 유사도 기반 기하 로스 가중치(0~1) 산출 """
        pred_flat = pred_heatmap.view(-1, pred_heatmap.shape[-1])
        gt_flat = gt_heatmap.view(-1, gt_heatmap.shape[-1])
        
        # 코사인 유사도 측정
        sim = F.cosine_similarity(pred_flat, gt_flat, dim=-1).mean()
        
        # 시그모이드 게이팅
        w_geom = 1.0 / (1.0 + torch.exp(-self.beta * (sim - self.tau)))
        
        return torch.clamp(w_geom, min=0.0, max=1.0)