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
# [신규 추가] 자동 스케일링이 탑재된 곡률 기반 하이브리드 표면 로스
# (기존 compute_point_to_plane_loss를 완전히 대체 및 업그레이드)
# =====================================================================

class AutoScaledCurvatureSurfaceLoss(nn.Module):
    def __init__(self, target_norm_val=1.88, k_p2p=5, k_curv=15, alpha=10.0):
        """
        target_norm_val: 첫 배치에서 Raw Loss를 이 값에 맞춰 자동으로 배수(Scale Factor)를 고정합니다. (ex: PAConv 로그 기준 1.88)
        k_p2p: Point-to-Plane 평면 투영 및 법선 추출용 (좁은 영역, 기본 5)
        k_curv: 곡률(Surface Variation) 계산용 (넓은 영역, 기본 15)
        alpha: 곡률이 극심한 곳(이륜/대이륜)의 오차에 부여할 페널티 가중치
        """
        super().__init__()
        self.target_norm_val = target_norm_val
        self.k_p2p = k_p2p
        self.k_curv = k_curv
        self.alpha = alpha
        
        # 모델의 버퍼로 등록 (저장/로드 시 유지되며 역전파되지 않음, 초기값 -1)
        self.register_buffer('scale_factor', torch.tensor(-1.0))

    def forward(self, pred_coords, gt_coords, points):
        # ---------------------------------------------------------
        # 1. P2P 투영용 법선 추출 (k_p2p 사용: 좁은 영역)
        # ---------------------------------------------------------
        # GT 표면 주변의 좁은 이웃을 찾아 정확한 평면 법선을 구합니다.
        knn_p2p_gt = find_knn_points(gt_coords, points, self.k_p2p)
        center_p2p_gt = knn_p2p_gt.mean(dim=2, keepdim=True)
        
        centered_p2p_gt = knn_p2p_gt - center_p2p_gt
        cov_p2p_gt = torch.matmul(centered_p2p_gt.transpose(2, 3), centered_p2p_gt)
        _, eigvec_p2p_gt = torch.linalg.eigh(cov_p2p_gt)
        
        # 평면에 수직인 법선 벡터 (가장 작은 고윳값 방향)
        normal_gt = eigvec_p2p_gt[..., 0] 

        # ---------------------------------------------------------
        # 2. 곡률(Surface Variation) 추출 (k_curv 사용: 넓은 영역)
        # ---------------------------------------------------------
        # 예측점과 GT점 각각에 대해 조금 더 넓은 영역의 형태(곡률)를 파악합니다.
        knn_curv_pred = find_knn_points(pred_coords, points, self.k_curv)
        knn_curv_gt = find_knn_points(gt_coords, points, self.k_curv)

        center_curv_pred = knn_curv_pred.mean(dim=2, keepdim=True)
        center_curv_gt = knn_curv_gt.mean(dim=2, keepdim=True)

        centered_curv_pred = knn_curv_pred - center_curv_pred
        centered_curv_gt = knn_curv_gt - center_curv_gt

        cov_curv_pred = torch.matmul(centered_curv_pred.transpose(2, 3), centered_curv_pred)
        cov_curv_gt = torch.matmul(centered_curv_gt.transpose(2, 3), centered_curv_gt)

        eigval_curv_pred, _ = torch.linalg.eigh(cov_curv_pred)
        eigval_curv_gt, _ = torch.linalg.eigh(cov_curv_gt)

        # 표면 변화량 c 계산 (0 ~ 0.33)
        sum_eig_pred = torch.sum(eigval_curv_pred, dim=-1) + 1e-6
        sum_eig_gt = torch.sum(eigval_curv_gt, dim=-1) + 1e-6

        c_pred = eigval_curv_pred[..., 0] / sum_eig_pred
        c_gt = eigval_curv_gt[..., 0] / sum_eig_gt

        # [VRAM 최적화] GT 관련 연산은 역전파를 차단하여 메모리를 절약합니다.
        c_gt = c_gt.detach()
        normal_gt = normal_gt.detach()
        center_p2p_gt = center_p2p_gt.squeeze(2).detach()

        # ---------------------------------------------------------
        # 3. 로스 조합
        # ---------------------------------------------------------
        # [Loss A] 곡률 일관성 (예측된 점 주변의 뾰족함이 실제 뾰족함과 일치하는가)
        loss_curvature = F.l1_loss(c_pred, c_gt)

        # [Loss B] 곡률 가중치 P2P (뾰족한 능선에 있을수록 P2P 오차 폭발)
        vector_to_plane = pred_coords - center_p2p_gt
        p2p_distance = torch.abs(torch.sum(vector_to_plane * normal_gt, dim=-1))
        
        weight_curv = 1.0 + self.alpha * c_gt
        loss_p2p_weighted = (p2p_distance * weight_curv).mean()

        # 순수 기하학적 오차 합산
        raw_surface_loss = loss_p2p_weighted + loss_curvature

        # ---------------------------------------------------------
        # 4. [핵심] 첫 배치 자동 스케일링 고정 로직
        # ---------------------------------------------------------
        if self.training and self.scale_factor.item() < 0:
            # 첫 번째 배치의 raw_loss를 기준으로 타겟 스케일에 맞추기 위한 상수 도출
            calculated_scale = self.target_norm_val / (raw_surface_loss.item() + 1e-6)
            self.scale_factor.fill_(calculated_scale)
            print(f"\n[Auto-Scaler] Surface Loss 상수 자동 세팅 완료: x{self.scale_factor.item():.2f}")
            print(f" -> Raw: {raw_surface_loss.item():.5f} => Target: {self.target_norm_val}\n")

        # 평가 모드가 먼저 돌 경우를 대비해 abs() 처리 후 스케일 팩터 곱셈
        final_loss = raw_surface_loss * torch.abs(self.scale_factor)
        
        return final_loss


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