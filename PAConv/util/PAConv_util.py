# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: PAConv_util.py
# @Description:
# [PAConv 유틸리티 모듈 - Edge Feature 생성 및 KNN 추출기]
# - 🌟 3채널(10ch Edge), 6채널(12ch Edge), 7채널(14ch Edge) 완벽 호환 패치
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# 🚀 초고속 C++ KNN 커널 (PyTorch3D) 로드 시도
# ==========================================================
try:
    from pytorch3d.ops import knn_points
    USE_PYTORCH3D = True
    print(">>> [SUCCESS] 🚀 PAConv: PyTorch3D C++ KNN 커널을 사용합니다! (학습 속도 대폭 향상)")
except ImportError:
    USE_PYTORCH3D = False
    print(">>> [WARNING] ⚠️ PAConv: PyTorch3D가 없어 기존 파이토치 KNN을 사용합니다.")

def knn(x, k):
    """
    x: (B, C, N) 형태의 입력
    """
    if USE_PYTORCH3D:
        # PyTorch3D는 (B, N, C) 형태를 요구하므로 축 변환
        points = x.transpose(1, 2).contiguous() 
        # C++ CUDA 커널 연산 (메모리 낭비 제로, 속도 10배 이상)
        dists, idx, _ = knn_points(points, points, K=k)
        # 원래 코드와 호환성을 위해 음수 형태로 거리 반환
        return idx, -dists.transpose(1, 2)
        
    else:
        # 기존 순수 파이토치 연산 (설치 실패 시 Fallback)
        B, _, N = x.size()                                   
        inner = -2 * torch.matmul(x.transpose(2, 1), x)      
        xx = torch.sum(x ** 2, dim=1, keepdim=True)          
        pairwise_distance = -xx - inner - xx.transpose(2, 1) 
        _, idx = pairwise_distance.topk(k=k, dim=-1)         
        return idx, pairwise_distance                 

def get_graph_feature(x, k=20, idx=None):
    # C_in 대신 raw_channels로 명칭 변경 (실제 입력되는 3, 6, 7)
    batch_size, raw_channels, num_points = x.size()             
    
    xyz = x[:, :3, :]
    if idx is None:
        idx, _ = knn(xyz, k=k)                          

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  
    idx = idx + idx_base                                
    idx = idx.view(-1)                                  

    x_trans = x.transpose(2, 1).contiguous()            
    
    neighbor = x_trans.view(batch_size * num_points, -1)[idx, :]   
    neighbor = neighbor.view(batch_size, num_points, k, raw_channels)  
    center = x_trans.view(batch_size, num_points, 1, raw_channels).repeat(1, 1, k, 1)  

    # 공통 10채널 기하 특징: 상대(3) + 이웃(3) + 중심(3) + 거리(1)
    neighbor_xyz = neighbor[..., :3]
    center_xyz = center[..., :3]
    relative_xyz = neighbor_xyz - center_xyz
    dist = torch.linalg.vector_norm(relative_xyz, dim=3, keepdim=True)

    # =====================================================================
    # 🌟 원시 입력 채널을 목표 엣지 채널(Target Edge Channels)로 직관적 매핑
    # =====================================================================
    if raw_channels == 7:
        target_edge_channels = 14 # 곡률+방향 포함
    elif raw_channels == 6:
        target_edge_channels = 12 # 🌟 [복구] 방향(Eigenvector)만 포함
    else:
        target_edge_channels = 10 # 기본 XYZ 전용

    # =====================================================================
    # 🌟 14채널 / 12채널 / 10채널 완벽 분기 처리
    # =====================================================================
    if target_edge_channels == 14:
        # 곡률 및 방향(4채널) 차이 계산 -> 엣지 피처 14채널
        relative_geom = neighbor[..., 3:] - center[..., 3:]
        feature = torch.cat((relative_xyz, neighbor_xyz, center_xyz, dist, relative_geom), dim=3)
        
    elif target_edge_channels == 12:
        # 🌟 [복구] 방향 벡터(3채널) 차이 계산 -> 엣지 피처 12채널
        relative_geom = neighbor[..., 3:] - center[..., 3:]
        feature = torch.cat((relative_xyz, neighbor_xyz, center_xyz, dist, relative_geom), dim=3)

    elif target_edge_channels == 10:
        # 순수 XYZ 전용 -> 엣지 피처 10채널
        feature = torch.cat((relative_xyz, neighbor_xyz, center_xyz, dist), dim=3)

    return feature.permute(0, 3, 1, 2).contiguous()     

def get_scorenet_input(x, idx=None, k=20):
    """
    텐서 차원 검사 (연산량 튜닝 방어 코드)
    PAConv_model.py에서 이미 연산된 엣지 피처(4차원)가 들어오면 중복 연산 없이 패스
    """
    if len(x.shape) == 4:
        return x
    return get_graph_feature(x, k=k, idx=idx)

def feat_trans_dgcnn(point_input, kernel, m):
    B, _, N = point_input.size()                          
    point_output = torch.matmul(
        point_input.permute(0, 2, 1).repeat(1, 1, 2),     
        kernel                                            
    ).view(B, N, m, -1)                                   

    center_output = torch.matmul(
        point_input.permute(0, 2, 1),                     
        kernel[:point_input.size(1)]                      
    ).view(B, N, m, -1)                                   

    return point_output, center_output                    

def feat_trans_pointnet(point_input, kernel, m):
    B, _, N = point_input.size()  
    point_output = torch.matmul(point_input.permute(0, 2, 1), kernel).view(B, N, m, -1)  
    return point_output

class ScoreNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[16], last_bn=False):
        super(ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()

        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs_nohidden = nn.Conv2d(in_channel, out_channel, 1, bias=not last_bn)
            if self.last_bn:
                self.mlp_bns_nohidden = nn.BatchNorm2d(out_channel)
        else:
            self.mlp_convs_hidden.append(nn.Conv2d(in_channel, hidden_unit[0], 1, bias=False))  
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):  
                self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1, bias=False))
                self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[-1], out_channel, 1, bias=not last_bn))  
            self.mlp_bns_hidden.append(nn.BatchNorm2d(out_channel))

    def forward(self, xyz, calc_scores='softmax', bias=0):
        B, _, N, K = xyz.size()  
        scores = xyz             

        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            if self.last_bn:
                scores = self.mlp_bns_nohidden(self.mlp_convs_nohidden(scores))  
            else:
                scores = self.mlp_convs_nohidden(scores)                         
        else:
            for i, conv in enumerate(self.mlp_convs_hidden):
                if i == len(self.mlp_convs_hidden)-1:  
                    if self.last_bn:
                        bn = self.mlp_bns_hidden[i]
                        scores = bn(conv(scores))      
                    else:
                        scores = conv(scores)          
                else:
                    bn = self.mlp_bns_hidden[i]
                    scores = F.relu(bn(conv(scores)))  

        if calc_scores == 'softmax':
            scores = F.softmax(scores, dim=1)+bias      
        elif calc_scores == 'sigmoid':
            scores = torch.sigmoid(scores)+bias         
        else:
            raise ValueError('Not Implemented!')

        scores = scores.permute(0, 2, 3, 1)             
        return scores                                   

class Attention_Layer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(Attention_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, N = x.size()
        y = self.avg_pool(x)           
        y = y.view(B, C)               
        y = self.fc(y).view(B, C, 1)   
        return x * y.expand_as(x)