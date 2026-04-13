import torch
from torch import nn
import torch.nn.functional as F

# ==============================================================================
# [NotebookLM을 위한 파이프라인 아키텍처 요약]
# 이 파일은 3D 랜드마크 탐지를 위한 다중 작업 학습(Multi-task Learning)의 핵심 오차(Loss)를 계산합니다.
# 
# 💡 핵심 기하학 로스 (CurvatureSurfaceLoss)의 의도:
# 단순 유클리드 좌표 차이(Point-to-Point)만 계산하면 예측점이 3D 귀 표면을 벗어나 허공에 뜨는 현상이 발생합니다.
# 이를 막기 위해 1) 예측점이 정답 표면에 완벽히 밀착되도록 수직 투영 거리(Point-to-Plane)를 구하고,
# 2) 예측점과 정답점의 곡률(Curvature) 형태와 뼈대 방향(Direction)이 다를 경우 오차를 기하급수적으로 증폭(Multiplier)시켜
# 해부학적으로 완벽한 랜드마크 배치를 강제합니다.
# ==============================================================================

class AdaptiveWingLoss(nn.Module):
    """
    [Adaptive Wing Loss]
    - 논문: ICCV 2019
    - [의도]: 히트맵 회귀 시, 정답 부근(오차가 작은 곳)에서는 Log 함수를 써서 미세한 오차에도 민감하게 반응하게 하고,
             오차가 큰 곳에서는 선형(Linear) 함수를 써서 그래디언트 폭발을 막아 안정적으로 수렴하게 만듭니다.
    """
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)
        )
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.omega, self.alpha - y2))
        loss2 = A * delta_y2 - C
        
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

# =====================================================================
# 미분 가능한 3D 좌표 추출 및 공통 유틸리티
# =====================================================================

def get_differentiable_coords(points, heatmap, k=10):
    """ 
    [Soft-argmax 기반 3D 좌표 추출]
    - [의도]: 단순히 확률이 가장 높은 점(argmax)을 고르면 미분이 끊겨 역전파(Backprop)가 불가능해집니다.
             따라서 상위 K개의 점을 뽑아 확률값을 가중치로 삼는 평균(무게중심)을 구하여 연속적이고 미분 가능한 좌표를 생성합니다.
    """
    B, K_lm, N = heatmap.shape
    pred_coords = []
    
    for i in range(K_lm):
        heat = heatmap[:, i, :] 
        topk_weights, topk_indices = torch.topk(heat, k, dim=1) 
        topk_weights_norm = F.softmax(topk_weights, dim=1) 
        
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, 3)
        topk_points = torch.gather(points, 1, idx_expanded) 
        
        coord = torch.sum(topk_points * topk_weights_norm.unsqueeze(-1), dim=1) 
        pred_coords.append(coord)
        
    return torch.stack(pred_coords, dim=1) 

def find_knn_points(pred_coords, points, k=10):
    """ 
    [K-Nearest Neighbors 추출]
    - [의도]: 단일 점의 정보만으로는 3D 표면을 이해할 수 없으므로, 주변 점 K개를 모아 '로컬 패치(표면)' 단위의 기하학 분석을 준비합니다.
    """
    dist_matrix = torch.cdist(pred_coords, points) 
    _, knn_indices = torch.topk(dist_matrix, k, dim=2, largest=False) 
    
    idx_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    points_expanded = points.unsqueeze(1).expand(-1, pred_coords.size(1), -1, -1)
    knn_points = torch.gather(points_expanded, 2, idx_expanded) 
    
    return knn_points

def compute_input_geometric_features(points, k=15):
    B, N, _ = points.shape
    knn_points = find_knn_points(points, points, k=k)
    
    center = knn_points.mean(dim=2, keepdim=True)
    centered = knn_points - center
    cov = torch.matmul(centered.transpose(2, 3), centered)
    _, eigvec = torch.linalg.eigh(cov)
    
    principal_dir = eigvec[..., 2]
    dot_product = torch.sum(principal_dir * center.squeeze(2), dim=-1, keepdim=True)
    principal_dir = principal_dir * torch.sign(dot_product)
    
    return principal_dir 

# =====================================================================
# 단위 통일(Unit-Aligned) 하이브리드 표면 로스
# =====================================================================

class CurvatureSurfaceLoss(nn.Module):
    def __init__(self, k_p2p=5, k_curv=30, alpha=10.0, beta=1.0):
        super().__init__()
        self.k_p2p = k_p2p        # 평면 피팅용 좁은 반경 (밀착도 측정)
        self.k_curv = k_curv      # 곡률 측정용 넓은 반경 (형태 파악)
        self.alpha = alpha        # 곡률 오차 패널티 승수
        self.beta = beta # 주방향 오차 패널티 승수

    def forward(self, pred_coords, gt_coords, points, disable_norm=False):
        # -------------------------------------------------------------
        # 1. Point-to-Plane (P2P) 투영용 법선(Normal Vector) 추출
        # [의도]: 정답 좌표 주변 k_p2p(5)개의 점으로 평면을 피팅하고 직교 벡터(법선)를 찾습니다.
        # -------------------------------------------------------------
        knn_p2p_gt = find_knn_points(gt_coords, points, self.k_p2p)
        center_p2p_gt = knn_p2p_gt.mean(dim=2, keepdim=True)
        
        centered_p2p_gt = knn_p2p_gt - center_p2p_gt
        cov_p2p_gt = torch.matmul(centered_p2p_gt.transpose(2, 3), centered_p2p_gt)
        _, eigvec_p2p_gt = torch.linalg.eigh(cov_p2p_gt)
        normal_gt = eigvec_p2p_gt[..., 0] 

        # -------------------------------------------------------------
        # 2. 곡률/방향 분석용 공분산 행렬 생성 (넓은 반경)
        # -------------------------------------------------------------
        knn_curv_pred = find_knn_points(pred_coords, points, self.k_curv)
        knn_curv_gt = find_knn_points(gt_coords, points, self.k_curv)

        center_curv_pred = knn_curv_pred.mean(dim=2, keepdim=True)
        center_curv_gt = knn_curv_gt.mean(dim=2, keepdim=True)

        centered_curv_pred = knn_curv_pred - center_curv_pred
        centered_curv_gt = knn_curv_gt - center_curv_gt

        cov_curv_pred = torch.matmul(centered_curv_pred.transpose(2, 3), centered_curv_pred)
        cov_curv_gt = torch.matmul(centered_curv_gt.transpose(2, 3), centered_curv_gt)

        eigval_pred, eigvec_pred = torch.linalg.eigh(cov_curv_pred)
        eigval_gt, eigvec_gt = torch.linalg.eigh(cov_curv_gt)

        # -------------------------------------------------------------
        # 3. 곡률 크기 (Curvature Magnitude) 및 방향성(Direction) 추출
        # -------------------------------------------------------------
        sum_eig_pred = torch.sum(eigval_pred, dim=-1) + 1e-6
        sum_eig_gt = torch.sum(eigval_gt, dim=-1) + 1e-6

        c_pred = eigval_pred[..., 0] / sum_eig_pred
        c_gt = eigval_gt[..., 0] / sum_eig_gt

        primary_dir_pred = eigvec_pred[..., 2] 
        primary_dir_gt = eigvec_gt[..., 2].detach() 
        cos_sim = torch.abs(torch.sum(primary_dir_pred * primary_dir_gt, dim=-1))

        # -------------------------------------------------------------
        # 4. 🌟 최종 Loss 조합 및 세부 지표 분리 반환
        # -------------------------------------------------------------
        c_gt = c_gt.detach()
        normal_gt = normal_gt.detach()
        center_p2p_gt = center_p2p_gt.squeeze(2).detach()

        # [1] 무차원 오차 계산 (엑셀 기록용 세부 지표)
        diff_curvature = torch.abs(c_pred - c_gt)  # 곡률 형태 오차
        diff_direction = 1.0 - cos_sim             # 곡선 방향 오차

        # [2] 물리적 거리: 순수 수직 거리(Point-to-Plane) (mm 단위)
        vector_to_plane = pred_coords - center_p2p_gt
        p2p_distance = torch.abs(torch.sum(vector_to_plane * normal_gt, dim=-1))

        # [3] 곡률 및 방향성 패널티 적용
        if disable_norm:
            weight_multiplier = 1.0 
        else:
            weight_multiplier = 1.0 + (self.alpha * diff_curvature) + (self.beta * diff_direction)
        
        # 🌟 통합 로스 (역전파용 메인 로스)
        loss_unified = (p2p_distance * weight_multiplier).mean()

        # 🌟 [NotebookLM 참고]: 연구자의 딥러닝 모니터링을 위해 통합로스뿐만 아니라 
        # 수직거리(P2P), 곡률오차(Curv), 방향오차(Dir)를 완전히 분리하여 4개의 값으로 반환합니다.
        return loss_unified, p2p_distance.mean(), diff_curvature.mean(), diff_direction.mean()

# =====================================================================
# 위상 로스 및 Focal 로스
# =====================================================================

def compute_structural_loss(pred_coords, gt_coords):
    """
    [구조적 제약(Structural Constraint) 로스]
    - [의도]: 랜드마크들 사이의 상대적인 거리(뼈대 비율)가 정답과 일치하도록 제약합니다.
    """
    pred_dist_matrix = torch.cdist(pred_coords, pred_coords)
    gt_dist_matrix = torch.cdist(gt_coords, gt_coords)
    loss_struct = F.l1_loss(pred_dist_matrix, gt_dist_matrix)
    return loss_struct

def dynamic_focal_l1_loss(pred_coords, gt_coords, gamma=2.0):
    """
    [Focal L1 Loss]
    - [의도]: 배치의 평균 오차보다 못 맞추는 악성(Hard) 샘플에 가중치를 동적으로 높게 주어 집중 훈련합니다.
    """
    l1_errors = torch.norm(pred_coords - gt_coords, p=1, dim=-1)
    dynamic_threshold = l1_errors.mean().detach() + 1e-5
    
    focal_weights = torch.pow(l1_errors.detach() / dynamic_threshold, gamma)
    focal_weights = torch.clamp(focal_weights, min=0.1, max=5.0) 
    
    weighted_loss = (l1_errors * focal_weights).mean()
    return weighted_loss