import torch
from torch import nn
import torch.nn.functional as F

'''
[Module: Adaptive Wing Loss]
- 논문: Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression (ICCV 2019)
- 목적: 히트맵 기반 랜드마크 회귀(Regression) 시, 정답에 가까운(오차가 작은) 지점의 그래디언트를 증폭시켜 더 정밀한 위치를 학습하도록 유도함.
- 수학적 원리: 오차가 임계값(theta)보다 작을 때는 비선형적인 Logarithmic 함수를 사용하여 오차에 매우 민감하게 반응하게 하고, 오차가 클 때는 선형(Linear) 함수를 사용하여 발산을 막고 학습 안정성을 확보함.
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
        
        # 오차(delta_y) 크기에 따라 임계값(theta)을 기준으로 두 영역으로 분리
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        
        # [1] 오차가 작은 영역: Log 함수 적용 (미세 조정)
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        
        # [2] 오차가 큰 영역: Linear 함수 적용 (안정적 수렴)
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
    - 예측된 히트맵에서 확률이 가장 높은 상위 K개의 점을 이용해 3D 무게중심 좌표(Soft-argmax)를 추출함.
    - 의도: 이산적(Discrete)인 최대값 인덱스 추출(argmax)은 미분이 불가능하므로, 
            확률값을 가중치로 삼아 연속적인 공간의 좌표를 계산하여 Backpropagation이 가능하게 만듦.
    """
    # points: (B, N, 3), heatmap: (B, K_lm, N)
    B, K_lm, N = heatmap.shape
    pred_coords = []
    
    for i in range(K_lm):
        heat = heatmap[:, i, :] 
        topk_weights, topk_indices = torch.topk(heat, k, dim=1) 
        topk_weights_norm = F.softmax(topk_weights, dim=1) # 상위 K개의 확률을 정규화
        
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, 3)
        topk_points = torch.gather(points, 1, idx_expanded) 
        
        # 정규화된 확률을 가중치로 하여 3D 점들의 가중 평균(무게중심)을 산출
        coord = torch.sum(topk_points * topk_weights_norm.unsqueeze(-1), dim=1) 
        pred_coords.append(coord)
        
    return torch.stack(pred_coords, dim=1) # (B, K_lm, 3)

def find_knn_points(pred_coords, points, k=10):
    """ 
    [K-Nearest Neighbors 기반 로컬 패치 추출]
    - 예측된 좌표와 가장 가까운 표면의 점 K개를 찾음.
    - 의도: 단일 점이 아닌 주변 점들을 모아 '표면(Surface)' 단위의 기하학적 분석(곡률, 법선 벡터 등)을 수행하기 위함.
    """
    # pred_coords: (B, K_lm, 3), points: (B, N, 3)
    dist_matrix = torch.cdist(pred_coords, points) # (B, K_lm, N) 유클리디안 거리 행렬
    _, knn_indices = torch.topk(dist_matrix, k, dim=2, largest=False) 
    
    idx_expanded = knn_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
    points_expanded = points.unsqueeze(1).expand(-1, pred_coords.size(1), -1, -1)
    knn_points = torch.gather(points_expanded, 2, idx_expanded) # (B, K_lm, k, 3)
    
    return knn_points

def compute_input_geometric_features(points, k=15):
    """
    [입력 데이터의 기하학적 특징(주방향) 사전 계산]
    모든 입력 포인트(B, N, 3)에 대해 주변 k개의 점을 모아 
    공분산 행렬을 구하고, 주방향(Principal Direction) 벡터를 추출합니다.
    """
    B, N, _ = points.shape
    
    # 1. 이웃 점 추출 (B, N, k, 3)
    knn_points = find_knn_points(points, points, k=k)
    
    # 2. 공분산 행렬(Covariance Matrix) 생성 및 아이겐 분해(Eigen Decomposition)
    center = knn_points.mean(dim=2, keepdim=True)
    centered = knn_points - center
    cov = torch.matmul(centered.transpose(2, 3), centered)
    _, eigvec = torch.linalg.eigh(cov)
    
    # 3. 가장 큰 고유값에 해당하는 아이겐벡터 (주방향 능선 벡터, 능선/골짜기의 뻗는 방향) 추출 -> (B, N, 3)
    principal_dir = eigvec[..., 2]
    
    # 4. PCA 부호 모호성(Sign Ambiguity) 해결
    # - PCA로 얻은 벡터는 방향성이 반대일 수 있으므로, 무게중심 벡터와의 내적을 통해 항상 바깥쪽(또는 일관된 방향)을 향하도록 부호를 통일함
    dot_product = torch.sum(principal_dir * center.squeeze(2), dim=-1, keepdim=True)
    principal_dir = principal_dir * torch.sign(dot_product)
    
    return principal_dir # (B, N, 3)

# =====================================================================
# [업그레이드 완료] 단위 통일(Unit-Aligned) 및 방향성(Eigenvector) 결합 하이브리드 표면 로스
# =====================================================================

class CurvatureSurfaceLoss(nn.Module):
    """
    [하이브리드 표면 손실 함수 (Hybrid Surface Loss)]
    - 목적: 예측된 랜드마크가 정답 랜드마크의 단순 좌표뿐만 아니라, 해당 위치의 3D 표면 특성(곡률, 방향)까지 모사하도록 강제함.
    - 특징: 무차원인 곡률/방향 오차를 물리적 거리(mm) 오차에 곱셈 형태로 결합하여, 단위를 통일하고 로스 스케일을 안정화함.
    """
    def __init__(self, k_p2p=5, k_curv=30, alpha=10.0, dir_weight=1.0):
        super().__init__()
        self.k_p2p = k_p2p
        self.k_curv = k_curv
        self.alpha = alpha
        self.dir_weight = dir_weight # 방향성 로스 가중치

    # 🔥 disable_norm 인자 추가 완료! (학습 단계별 동적 적용 목적)
    def forward(self, pred_coords, gt_coords, points, disable_norm=False):
        # -------------------------------------------------------------
        # 1. Point-to-Plane (P2P) 투영용 법선(Normal Vector) 추출 (좁은 영역)
        # - K_p2p 크기의 좁은 이웃을 통해 미세한 로컬 평면을 구성하고, 그 평면의 직교 벡터(법선)를 찾음.
        # -------------------------------------------------------------
        knn_p2p_gt = find_knn_points(gt_coords, points, self.k_p2p)
        center_p2p_gt = knn_p2p_gt.mean(dim=2, keepdim=True)
        
        centered_p2p_gt = knn_p2p_gt - center_p2p_gt
        cov_p2p_gt = torch.matmul(centered_p2p_gt.transpose(2, 3), centered_p2p_gt)
        _, eigvec_p2p_gt = torch.linalg.eigh(cov_p2p_gt)
        normal_gt = eigvec_p2p_gt[..., 0] # 가장 작은 고유값의 벡터 = 평면의 법선

        # -------------------------------------------------------------
        # 2. 곡률/방향 분석용 공분산 행렬 생성 및 고유 분해 (넓은 영역)
        # - K_curv 크기의 넓은 이웃을 통해 해당 부위의 거시적인 굴곡과 주름의 방향을 파악함.
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
        # 3. 곡률 크기 (Curvature Magnitude) 추출
        # - Surface Variation = λ0 / (λ0 + λ1 + λ2)
        # - 값이 클수록 뾰족/복잡한 표면, 작을수록 평탄한 표면임을 의미함.
        # -------------------------------------------------------------
        sum_eig_pred = torch.sum(eigval_pred, dim=-1) + 1e-6
        sum_eig_gt = torch.sum(eigval_gt, dim=-1) + 1e-6

        c_pred = eigval_pred[..., 0] / sum_eig_pred
        c_gt = eigval_gt[..., 0] / sum_eig_gt

        # -------------------------------------------------------------
        # 4. 가장 큰 아이겐벡터의 코사인 유사도 방향성(Direction) 추출
        # - 특정 구조(예: 귀의 대이륜) 등 뚜렷한 주름이 있는 경우, 랜드마크가 그 뼈대의 방향을 정확히 따라가도록 유도.
        # -------------------------------------------------------------
        primary_dir_pred = eigvec_pred[..., 2] # (B, K_lm, 3)
        primary_dir_gt = eigvec_gt[..., 2].detach() # (B, K_lm, 3)

        # PCA 부호 모호성(Sign Ambiguity) 해결을 위한 절댓값 코사인 유사도
        cos_sim = torch.abs(torch.sum(primary_dir_pred * primary_dir_gt, dim=-1))

        # -------------------------------------------------------------
        # 5. 🌟 최종 Loss 조합 (단위 통일: 모든 무차원을 mm 거리의 페널티로 흡수)
        # -------------------------------------------------------------
        # 정답 데이터(GT)는 학습되어 변경되지 않도록 detach() 처리
        c_gt = c_gt.detach()
        normal_gt = normal_gt.detach()
        center_p2p_gt = center_p2p_gt.squeeze(2).detach()

        # [1] 무차원(Dimensionless) 오차들 계산 (비율 & 각도)
        diff_curvature = torch.abs(c_pred - c_gt)  # 곡률 형태 오차 (0 ~ 1)
        diff_direction = 1.0 - cos_sim             # 곡선 방향 오차 (0 ~ 1, 코사인 거리)

        # [2] 물리적 거리(Length) 오차 계산 (mm 단위)
        # 예측점 벡터를 GT 평면 법선에 투영(Dot Product)하여 평면까지의 최단 거리를 산출함
        vector_to_plane = pred_coords - center_p2p_gt
        p2p_distance = torch.abs(torch.sum(vector_to_plane * normal_gt, dim=-1))

        # 🔥 [3] 정규화 스위치 적용 로직 추가!
        if disable_norm:
            weight_multiplier = 1.0 # 1단계(PAConv) 학습 시에는 랜드마크가 대략적인 위치부터 찾도록 곡률/방향 페널티 없이 순수 거리만 줄임
        else:
            # 2단계(DeepLA 등) 정밀 학습 시, 예측점이 GT의 곡률이나 방향을 벗어날수록 P2P 거리 오차를 증폭(Multiplier)시켜 페널티를 강하게 줌
            weight_multiplier = 1.0 + (self.alpha * diff_curvature) + (self.dir_weight * diff_direction)
        
        # 🌟 최종 반환: 오직 '거리 차원(mm)' 하나만을 가지는 통합 로스
        loss_unified = (p2p_distance * weight_multiplier).mean()

        return loss_unified

# =====================================================================
# 구조적 위상 로스 (Structural / Pairwise Distance Loss)
# =====================================================================

def compute_structural_loss(pred_coords, gt_coords):
    """
    [구조적 제약(Structural Constraint) 유지 로스]
    - 랜드마크 간의 상대적 거리(뼈대 비율) 오차를 계산하는 로스.
    - 각 랜드마크의 절대 위치뿐만 아니라, 랜드마크들 사이의 전체적인 위상(Topology)이 정답과 일치하도록 강제함.
    """
    # pred_coords: (B, K_lm, 3)
    # gt_coords: (B, K_lm, 3)
    pred_dist_matrix = torch.cdist(pred_coords, pred_coords)
    gt_dist_matrix = torch.cdist(gt_coords, gt_coords)
    loss_struct = F.l1_loss(pred_dist_matrix, gt_dist_matrix)
    return loss_struct

def dynamic_focal_l1_loss(pred_coords, gt_coords, gamma=2.0):
    """
    [CVPR 2022+ 트렌드 반영 Focal Loss]
    - 배치의 평균 오차를 동적 기준선(Dynamic Threshold)으로 삼아,
      평균보다 못 맞추는 악성(Hard) 랜드마크에 기하급수적 패널티를 부여하는 로스.
    - 모델이 쉬운 랜드마크에 안주하지 않고 어려운 부분을 집중적으로 교정하도록 유도함.
    """
    l1_errors = torch.norm(pred_coords - gt_coords, p=1, dim=-1)
    dynamic_threshold = l1_errors.mean().detach() + 1e-5
    
    # 오차가 평균(threshold)보다 크면 가중치 > 1, 작으면 가중치 < 1
    focal_weights = torch.pow(l1_errors.detach() / dynamic_threshold, gamma)
    focal_weights = torch.clamp(focal_weights, min=0.1, max=5.0) # 폭발 방지용 클램핑
    
    weighted_loss = (l1_errors * focal_weights).mean()
    return weighted_loss