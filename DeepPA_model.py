import torch
import torch.nn as nn
from deeppa_semseg import DeepPA_semseg

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "utils" / "pointnet2_ops_lib"))

# 🔥 다시 초고속 C++ 커널을 켭니다!
try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample as fps_cpp
    USE_CPP_FPS = True
    print(">>> [INFO] DeepPA 모드: 초고속 C++ FPS 커널을 사용합니다.")
except ImportError:
    USE_CPP_FPS = False

def farthest_point_sample(xyz, npoint):
    """
    [Farthest Point Sampling (FPS) 래퍼 함수]
    DeepPA의 계층적 구조(U-Net 스타일)를 위해 점의 개수를 점진적으로 줄일 때 사용합니다.
    C++ 커널이 있으면 O(N) 최적화로 초고속 처리하고, 없으면 PyTorch 기본 텐서 연산을 통해 O(N^2)로 계산합니다.
    """
    if USE_CPP_FPS:
        xyz = xyz.contiguous() 
        idx = fps_cpp(xyz, npoint)
        return idx.long()
    else:
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

class DeepPA_Wrapper(nn.Module):
    """
    [DeepPA 인터페이스 (Wrapper)]
    - 역할: 외부(train.py 등)와 내부(DeepPA_semseg.py)를 연결하는 다리입니다.
    - NotebookLM을 위한 분석 포인트: DeepPA 네트워크 자체가 120층이 넘는 엄청나게 깊은(Deep) 모델이기 때문에, 
      메모리 초과(OOM)를 방지하기 위해 계층(Stage) 간의 연결 고리(KNN, FPS)를 여기서 미리 전부 계산하여 인덱스 리스트로 전달하는 방식을 취합니다.
    """
    def __init__(self, args, landmark_num=None):
        super(DeepPA_Wrapper, self).__init__()
        
        # 내부 모델(DeepPA_semseg)이 요구하는 설정 구조체(Config)를 만듭니다.
        class Config: pass
        self.dl_args = dl_args = Config()
        
        dl_args.num_classes = landmark_num if landmark_num is not None else args.landmark_num   
        num_points = args.num_points # 보통 2048 또는 8192
        
        dl_args.bn_momentum = 0.1
        dl_args.act = nn.GELU     
        dl_args.head_dim = 256    
        dl_args.mlp_ratio = 1.0               
        
        # 🌟 모델의 극단적인 깊이 (총 120층)
        dl_args.depths = [20, 20, 60, 20] 
        
        # 깊이가 120층 이상일 경우, VRAM 폭발을 막기 위해 PyTorch의 Checkpoint 기능을 활성화
        total_depth = sum(dl_args.depths)
        if total_depth >= 120:
            dl_args.use_cp = True
        else:
            dl_args.use_cp = False
            
        # 각 계층(Stage)별 채널(차원), 이웃 개수(K), 헤드 차원 설정
        dl_args.dims = [64, 128, 256, 512]   
        dl_args.ks = [24, 24, 24, 24]         
        dl_args.nbr_dims = [64, 128, 256, 512] 
        dl_args.cor_std = [1.0, 1.0, 1.0, 1.0] 
        dl_args.head_drops = [0.1, 0.1, 0.1, 0.1] 
        
        # Stochastic Depth (DropPath): 레이어가 너무 깊을 때 일부 레이어를 건너뛰게 하여 학습을 돕고 과적합을 방지
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        dl_args.drop_paths = []
        cur = 0
        for d in dl_args.depths:
            dl_args.drop_paths.append(dpr[cur:cur + d])
            cur += d
            
        # 점의 개수가 계층마다 1/4씩 줄어듦 (Hierarchical Downsampling)
        dl_args.ns = [num_points, num_points // 4, num_points // 16, num_points // 64]
        
        # 🔥 14채널 엣지 구성을 위한 7채널(XYZ 3 + 주방향 3 + 곡률 1) 입력 고정
        dl_args.in_channels = 7 
        
        self.k = dl_args.ks[0]
        self.stage_count = len(dl_args.depths)
        
        # 실질적인 핵심 모델 객체 생성
        self.model = DeepPA_semseg(dl_args)

    def forward(self, x, prior_heatmap=None):
        B, C, N = x.shape
        
        # ---------------------------------------------------------------------
        # 🌟 [해결책 1. VRAM 및 연산 폭발 방지 메커니즘]
        # 공간상 거리를 구하는 K-NN 연산과 FPS 샘플링에 사용되는 xyz 좌표는 
        # 학습에 의해 업데이트될 대상이 아니므로 .detach()를 통해 미분 그래프에서 끊어냅니다.
        # 또한, 공간 거리를 잴 때는 7채널 피처가 아니라 앞의 3채널(물리적 XYZ)만 사용합니다.
        # ---------------------------------------------------------------------
        xyz = x[:, :3, :].permute(0, 2, 1).contiguous().detach()
        
        # ---------------------------------------------------------------------
        # 🌟 [해결책 2. 120층 딥 네트워크 학습 활성화]
        # 반면, 실제 모델 내부에서 합성곱 연산을 거칠 피처(feature)는 7채널 전체를 넘겨주고,
        # .requires_grad_(True)를 통해 미분 추적을 강제로 활성화하여 깊은 망의 가중치들이 정상적으로 업데이트되도록 유도합니다.
        # ---------------------------------------------------------------------
        feature = x.permute(0, 2, 1).contiguous() 
        if self.training:
            feature.requires_grad_(True)  
            
        if prior_heatmap is not None:
            prior_heatmap = prior_heatmap.permute(0, 2, 1).contiguous()
            if self.training:
                prior_heatmap.requires_grad_(True) # 1단계(PAConv)에서 넘어온 힌트도 학습 사이클에 포함
            
        device = x.device
        up_idx_list = []
        down_knn_list = []
        
        cur_xyz = xyz
        
        # ---------------------------------------------------------------------
        # [Step 3: 계층적 인덱스(Hierarchical Indexing) 사전 계산]
        # 내부 모델이 포워딩 중에 일일이 계산하면 병목이 생기므로, 
        # 밖에서 미리 모든 단계(Stage)의 이웃 구조(KNN)와 축소/확장 인덱스를 전부 구해 리스트로 넘겨줍니다.
        # ---------------------------------------------------------------------
        for d in range(self.stage_count):
            num_points = cur_xyz.shape[1]
            safe_k = min(self.dl_args.ks[d], num_points)
            
            # [3-1] 현재 해상도에서의 K-NN 이웃 인덱스 계산
            if safe_k < 1:
                idx = torch.zeros((B, num_points, self.dl_args.ks[d]), device=device, dtype=torch.long)
            else:
                dist = torch.cdist(cur_xyz, cur_xyz)
                idx = torch.topk(dist, safe_k, dim=-1, largest=False)[1]
                if safe_k < self.dl_args.ks[d]: # 점 개수가 k보다 모자라면 마지막 점 복제 (패딩)
                    idx = torch.cat([idx, idx[:, :, -1:].expand(-1, -1, self.dl_args.ks[d] - safe_k)], dim=-1)
                    
            down_knn_list.append(idx)
            
            # [3-2] 다음 단계로 넘어가기 위한 다운샘플링(FPS) 인덱스 계산
            if d < self.stage_count - 1:
                next_points = self.dl_args.ns[d+1]
                
                down_idx = farthest_point_sample(cur_xyz, next_points)
                
                down_knn_list.append(down_idx)
                
                # 다운샘플링된 점들의 실제 좌표 추출
                batch_indices = torch.arange(B, dtype=torch.long, device=device).view(-1, 1).repeat(1, next_points)
                next_xyz = cur_xyz[batch_indices, down_idx, :]
                
                # [3-3] 나중에 업샘플링(U-Net의 디코더)할 때 어느 점으로 돌아갈지(Nearest Neighbor) 미리 계산
                dist_up = torch.cdist(cur_xyz, next_xyz)
                up_idx = torch.argmin(dist_up, dim=2)
                up_idx_list.append(up_idx)
                
                cur_xyz = next_xyz
                
        # 리스트 순서를 스택(Stack)처럼 뒤집고 조합하여 모델이 .pop()으로 하나씩 꺼내 쓰기 좋게 만듦
        down_knn_list = down_knn_list[::-1]
        indices = up_idx_list + down_knn_list
        
        # ---------------------------------------------------------------------
        # [Step 4: 실제 모델 실행 (Forward Pass)]
        # ---------------------------------------------------------------------
        out = self.model(xyz, feature, indices, prior_heatmap=prior_heatmap)
        
        # 학습 중에는 보조 손실(spa, sem)이 튜플로 반환되므로, 결과 텐서만 분리
        if isinstance(out, tuple):
            out = out[0]
            
        # (B, N, C) 형태로 연산된 것을 다시 외부 표준 (B, C, N)으로 되돌려 반환
        return out.permute(0, 2, 1).contiguous()