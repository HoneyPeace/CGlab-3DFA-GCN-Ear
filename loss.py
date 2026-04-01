import torch
from torch import nn
import torch.nn.functional as F

'''
Adaptive Wing Loss from 
Wang X, Bo L, Fuxin L. Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression. ICCV2019.
The following module is based on https://github.com/protossw512/AdaptiveWingLoss
'''
class AdaptiveWingLoss(nn.Module):
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
    """ 예측된 히트맵에서 상위 K개의 확률을 이용해 3D 무게중심 좌표(Soft-argmax)를 추출 """
    # points: (B, N, 3), heatmap: (B, K_lm, N)
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
        
    return torch.stack(pred_coords, dim=1) # (B, K_lm, 3)

def find_knn_points(pred_coords, points, k=10):
    """ 예측된 좌표와 가장 가까운 표면의 점 K개를 찾음 """
    # pred_coords: (B, K_lm, 3), points: (B, N, 3)
    dist_matrix = torch.cdist(pred_coords, points) # (B, K_lm, N)
    _, knn_indices = torch.topk(dist_matrix, k, dim=2, largest=False) 
    
    idx_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    points_expanded = points.unsqueeze(1).expand(-1, pred_coords.size(1), -1, -1)
    knn_points = torch.gather(points_expanded, 2, idx_expanded) # (B, K_lm, k, 3)
    
    return knn_points

# =====================================================================
# [신규 추가] 단위 통일(Unit-Aligned) 및 방향성(Eigenvector)이 결합된 하이브리드 표면 로스
# =====================================================================

class CurvatureSurfaceLoss(nn.Module):
    def __init__(self, k_p2p=5, k_curv=30, alpha=10.0, dir_weight=1.0):
        super().__init__()
        self.k_p2p = k_p2p
        self.k_curv = k_curv
        self.alpha = alpha
        self.dir_weight = dir_weight # 방향성 로스 가중치

    def forward(self, pred_coords, gt_coords, points):
        # -------------------------------------------------------------
        # 1. P2P 투영용 법선 추출 (좁은 영역)
        # -------------------------------------------------------------
        knn_p2p_gt = find_knn_points(gt_coords, points, self.k_p2p)
        center_p2p_gt = knn_p2p_gt.mean(dim=2, keepdim=True)
        
        centered_p2p_gt = knn_p2p_gt - center_p2p_gt
        cov_p2p_gt = torch.matmul(centered_p2p_gt.transpose(2, 3), centered_p2p_gt)
        _, eigvec_p2p_gt = torch.linalg.eigh(cov_p2p_gt)
        normal_gt = eigvec_p2p_gt[..., 0] 

        # -------------------------------------------------------------
        # 2. 공분산 행렬 생성 및 고유 분해 (넓은 영역)
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
        # 3. 곡률 크기 (Magnitude) 추출
        # -------------------------------------------------------------
        sum_eig_pred = torch.sum(eigval_pred, dim=-1) + 1e-6
        sum_eig_gt = torch.sum(eigval_gt, dim=-1) + 1e-6

        c_pred = eigval_pred[..., 0] / sum_eig_pred
        c_gt = eigval_gt[..., 0] / sum_eig_gt

        # -------------------------------------------------------------
        # 4. 가장 큰 아이겐벡터의 코사인 유사도 방향성(Direction) 추출
        # -------------------------------------------------------------
        primary_dir_pred = eigvec_pred[..., 2] # (B, K_lm, 3)
        primary_dir_gt = eigvec_gt[..., 2].detach() # (B, K_lm, 3)

        # PCA 부호 모호성(Sign Ambiguity) 해결을 위한 절댓값 코사인 유사도
        cos_sim = torch.abs(torch.sum(primary_dir_pred * primary_dir_gt, dim=-1))

        # -------------------------------------------------------------
        # 5. 🌟 최종 Loss 조합 (단위 통일: 모든 무차원을 mm 거리의 페널티로 흡수)
        # -------------------------------------------------------------
        c_gt = c_gt.detach()
        normal_gt = normal_gt.detach()
        center_p2p_gt = center_p2p_gt.squeeze(2).detach()

        # [1] 무차원(Dimensionless) 오차들 계산 (비율 & 각도)
        diff_curvature = torch.abs(c_pred - c_gt)  # 곡률 형태 오차 (0 ~ 1)
        diff_direction = 1.0 - cos_sim             # 곡선 방향 오차 (0 ~ 1)

        # [2] 물리적 거리(Length) 오차 계산 (mm 단위)
        vector_to_plane = pred_coords - center_p2p_gt
        p2p_distance = torch.abs(torch.sum(vector_to_plane * normal_gt, dim=-1))

        # [3] 단일 단위로의 융합 (Unit Alignment)
        # 무차원 오차들을 P2P 거리(mm)를 뻥튀기하는 '페널티 가중치'로 묶어버립니다.
        # 수식: 거리(mm) * (1 + 알파*곡률오차 + 베타*방향오차)
        weight_multiplier = 1.0 + (self.alpha * diff_curvature) + (self.dir_weight * diff_direction)
        
        # 🌟 최종 반환: 오직 '거리 차원(mm)' 하나만을 가지는 통합 로스
        loss_unified = (p2p_distance * weight_multiplier).mean()

        return loss_unified

# =====================================================================
# 구조적 위상 로스 (Structural / Pairwise Distance Loss)
# =====================================================================

def compute_structural_loss(pred_coords, gt_coords):
    """
    랜드마크 간의 상대적 거리(뼈대 비율) 오차를 계산하는 로스
    pred_coords: (B, K_lm, 3)
    gt_coords: (B, K_lm, 3)
    """
    # 1. 예측된 랜드마크들 사이의 모든 쌍(Pairwise) 거리 계산 -> (B, K_lm, K_lm)
    pred_dist_matrix = torch.cdist(pred_coords, pred_coords)
    
    # 2. 실제 정답 랜드마크들 사이의 모든 쌍 거리 계산 -> (B, K_lm, K_lm)
    gt_dist_matrix = torch.cdist(gt_coords, gt_coords)
    
    # 3. 예측 거리와 실제 거리의 차이(절댓값) 평균 반환
    loss_struct = F.l1_loss(pred_dist_matrix, gt_dist_matrix)
    
    return loss_struct

def dynamic_focal_l1_loss(pred_coords, gt_coords, gamma=2.0):
    """
    [CVPR 2022+ 트렌드] 배치의 평균 오차를 동적 기준선으로 삼아,
    평균보다 못 맞추는 악성 랜드마크에 기하급수적 패널티를 부여하는 로스
    """
    # 1. 40개 랜드마크의 L1 오차 계산 -> (B, 40)
    l1_errors = torch.norm(pred_coords - gt_coords, p=1, dim=-1)
    
    # 2. 현재 배치의 '평균 오차'를 동적 기준선(Dynamic Threshold)으로 설정
    # detach()를 붙여서 기준선 자체로는 역전파가 흐르지 않게 고정
    dynamic_threshold = l1_errors.mean().detach() + 1e-5
    
    # 3. 평균 대비 얼마나 더 틀렸는지 비율을 구하고, gamma 제곱으로 휘어버림 (Focal 효과)
    # 에러가 평균보다 크면(비율 > 1) 패널티 폭발, 평균보다 작으면(비율 < 1) 패널티 축소
    focal_weights = torch.pow(l1_errors.detach() / dynamic_threshold, gamma)
    
    # 4. 가중치가 너무 폭발해서 NaN 에러가 나는 것을 방지 (최대 5배까지만 허용)
    focal_weights = torch.clamp(focal_weights, min=0.1, max=5.0)
    
    # 5. 기존 L1 오차에 동적 가중치 곱하기
    weighted_loss = (l1_errors * focal_weights).mean()
    
    return weighted_loss