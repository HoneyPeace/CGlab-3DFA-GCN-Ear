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
# [신규 추가] 미분 가능한 3D 좌표 추출 및 Point-to-Plane Loss
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

def compute_point_to_plane_loss(pred_coords, points, k=5):
    """ 예측된 좌표를 로컬 가상 평면에 수직 투영(Projection)시키는 거리 반환 """
    # 1. 주변 표면 점 5개 찾기
    knn_points = find_knn_points(pred_coords, points, k) # (B, K_lm, k, 3)
    
    # 2. 로컬 평면의 중심점
    local_center = knn_points.mean(dim=2) # (B, K_lm, 3)
    
    # 3. 공분산 행렬(Covariance Matrix) 계산
    centered_knn = knn_points - local_center.unsqueeze(2)
    cov_matrix = torch.matmul(centered_knn.transpose(2, 3), centered_knn) 
    
    # 4. 고윳값 분해 (eigh) -> 법선 벡터 추출
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    normal_vector = eigenvectors[..., 0] # (B, K_lm, 3)
    
    # 5. Point-to-Plane 거리 계산
    vector_to_plane = pred_coords - local_center
    distance = torch.abs(torch.sum(vector_to_plane * normal_vector, dim=-1)) # (B, K_lm)
    
    return distance.mean()

# =====================================================================
# [신규 추가] 구조적 위상 로스 (Structural / Pairwise Distance Loss)
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

def dynamic_focal_l1_loss(pred_coords, gt_coords, gamma=2.0, max_weight=5.0):
    """
    [CVPR 2022+ 트렌드] 배치의 평균 오차를 동적 기준선으로 삼아,
    평균보다 못 맞추는 악성 랜드마크에 기하급수적 패널티를 부여하는 로스
    """
    # 1. 40개 랜드마크의 L1 오차 계산 -> (B, 40)
    l1_errors = torch.norm(pred_coords - gt_coords, p=1, dim=-1)
    
    # 2. 현재 배치의 '평균 오차'를 동적 기준선(Dynamic Threshold)으로 설정
    dynamic_threshold = l1_errors.mean().detach() + 1e-5
    
    # 3. 평균 대비 얼마나 더 틀렸는지 비율을 구하고, gamma 제곱으로 휘어버림 (Focal 효과)
    focal_weights = torch.pow(l1_errors.detach() / dynamic_threshold, gamma)
    
    # 👇👇👇 [숫자 5.0 대신 max_weight 변수로 변경] 👇👇👇
    # 4. 가중치가 너무 폭발해서 NaN 에러가 나는 것을 방지 (focal_max 적용)
    focal_weights = torch.clamp(focal_weights, min=0.1, max=max_weight)
    
    # 5. 기존 L1 오차에 동적 가중치 곱하기
    weighted_loss = (l1_errors * focal_weights).mean()
    
    return weighted_loss