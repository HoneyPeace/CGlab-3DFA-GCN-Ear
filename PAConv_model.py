import torch
import torch.nn as nn
import torch.nn.functional as F
from My_args import *
from PAConv.util.PAConv_util import knn, get_graph_feature, get_scorenet_input, feat_trans_dgcnn, ScoreNet, Attention_Layer
from PAConv.cuda_lib.functional import assign_score_withk as assemble_dgcnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
[Module: PAConv (Position Adaptive Convolution) Backbone]
- 역할: 3D Point Cloud 데이터를 입력받아 각 점이 특정 랜드마크일 확률(Heatmap)을 계산하는 핵심 딥러닝 모델.
- 핵심 논리: 기존 CNN처럼 고정된 커널(가중치)을 쓰지 않고, 
           ScoreNet을 통해 '점과 점 사이의 기하학적 위치 관계'를 분석하여 
           여러 개의 가중치 행렬(Weight Banks)을 동적으로 조합해 사용하는 최신 3D 합성곱 기법입니다.
'''
class PAConv(nn.Module):
    def __init__(self, args, landmark_num):
        super(PAConv, self).__init__()
        self.args = args
        self.k = args.k
        self.landmark_num = landmark_num
        self.calc_scores = args.calc_scores
        self.hidden = args.hidden
        self.m2, self.m3, self.m4, self.m5 = args.num_matrices # Weight Banks(가중치 행렬)의 개수
        
        # 🌟 [채널 동기화 핵심 로직]
        # util.py에서 만든 7채널(XYZ 3 + 주방향 3 + 곡률 1) 데이터가 여기서 처리됩니다.
        # 중심점(7)과 이웃점과의 차이(7)를 이어붙여(Concat) 총 14차원의 Edge Feature(간선 특징)를 만듭니다.
        in_channels = getattr(args, 'in_channels', 3)
        if in_channels == 7:
            self.edge_channels = 14
        elif in_channels == 6:
            self.edge_channels = 13
        else:
            self.edge_channels = 10 # 기본 3채널의 경우 (3*2 + 3 + 1 등 설계에 따라 다름)
        
        # [1. ScoreNets 정의] 
        # 위치 관계를 보고 "어떤 가중치 행렬(Weight matrix)을 얼마만큼 섞어 쓸 것인가?"(Score)를 계산하는 네트워크
        self.scorenet2 = ScoreNet(self.edge_channels, self.m2, hidden_unit=self.hidden[0])
        self.scorenet3 = ScoreNet(self.edge_channels, self.m3, hidden_unit=self.hidden[1])
        self.scorenet4 = ScoreNet(self.edge_channels, self.m4, hidden_unit=self.hidden[2])
        self.scorenet5 = ScoreNet(self.edge_channels, self.m5, hidden_unit=self.hidden[3])
        
        i2 = 64       
        o2 = i3 = 64  
        o3 = i4 = 64  
        o4 = i5 = 64  
        o5 = 64       

        # [2. Weight Banks (가중치 행렬 풀) 초기화]
        # ScoreNet이 계산한 비율(Score)에 따라 섞이게 될 기본 블록들입니다.
        tensor2 = nn.init.kaiming_normal_(torch.empty(self.m2, i2 * 2, o2), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i2 * 2, self.m2 * o2)
        tensor3 = nn.init.kaiming_normal_(torch.empty(self.m3, i3 * 2, o3), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i3 * 2, self.m3 * o3)
        tensor4 = nn.init.kaiming_normal_(torch.empty(self.m4, i4 * 2, o4), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i4 * 2, self.m4 * o4)
        tensor5 = nn.init.kaiming_normal_(torch.empty(self.m5, i5 * 2, o5), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i4 * 2, self.m5 * o5)

        self.matrice2 = nn.Parameter(tensor2, requires_grad=True)
        self.matrice3 = nn.Parameter(tensor3, requires_grad=True)
        self.matrice4 = nn.Parameter(tensor4, requires_grad=True)
        self.matrice5 = nn.Parameter(tensor5, requires_grad=True)

        self.bn2 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn3 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn4 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn5 = nn.BatchNorm1d(64, momentum=0.1)

        self.bnt = nn.BatchNorm1d(1024, momentum=0.1)
        self.bnc = nn.BatchNorm1d(64, momentum=0.1)

        self.bn6 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn7 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn8 = nn.BatchNorm1d(128, momentum=0.1)

        # 첫 번째 특징 추출 레이어 (Edge Feature -> 64차원 변환)
        self.conv1 = nn.Sequential(nn.Conv2d(self.edge_channels, 64, kernel_size=1, bias=True), 
                                   nn.BatchNorm2d(64, momentum=0.1))
                                   
        # [3. Global Feature & Output MLP 레이어]
        self.convt = nn.Sequential(nn.Conv1d(64*5, 1024, kernel_size=1, bias=False),
                                   self.bnt)

        self.conv6 = nn.Sequential(nn.Conv1d(1024+64*5, 256, kernel_size=1, bias=False),
                                   self.bn6)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn7)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn8)
        # 최종 히트맵 차원(landmark_num)으로 매핑하는 Projection 레이어
        self.conv9 = nn.Conv1d(128, landmark_num, kernel_size=1, bias=True)

        
    def forward(self, x):
        B, C, N = x.size()
        
        # ---------------------------------------------------------------------
        # [Step 1: 3D 물리적 공간 기반 이웃 탐색 (K-NN)]
        # 🌟 핵심: 채널이 7개(기하특징 포함)이더라도, 이웃을 찾을 때는 물리적 거리인 앞의 3채널(XYZ)만 사용합니다.
        # 이렇게 해야 기하학적 형태가 꼬이지 않고 정확한 로컬 패치(Local Patch)가 형성됩니다.
        # ---------------------------------------------------------------------
        physical_xyz = x[:, :3, :].contiguous()
        idx, _ = knn(physical_xyz, k=self.k)
        
        # ---------------------------------------------------------------------
        # [Step 2: 동적 합성곱(PAConv)을 위한 그래프 특징 및 Score 입력 추출]
        # ---------------------------------------------------------------------
        scorenet_input = get_scorenet_input(x, k=self.k, idx=idx)  
        x_edge_feat = get_graph_feature(x, k=self.k, idx=idx) # (B, 14, N, k)

        # Layer 1: 기본 Edge Feature 임베딩
        x_out = F.relu(self.conv1(x_edge_feat))
        x1 = x_out.max(dim=-1, keepdim=False)[0] # K개의 이웃 특징 중 가장 강한 특징만 남김 (Max Pooling)
        
        # ---------------------------------------------------------------------
        # [Step 3: PAConv 레이어 연쇄 통과 (Layer 2 ~ 5)]
        # 동작 방식: 
        # 1) feat_trans_dgcnn: 가중치 풀(matrice)을 이용해 다양한 변환 생성
        # 2) scorenet: 현재 기하학적 구조를 보고 가중치를 얼마나 섞을지 점수(Score) 계산
        # 3) assemble_dgcnn: 계산된 Score를 바탕으로 특징을 조합(Assemble)하여 동적 합성곱 수행
        # ---------------------------------------------------------------------
        x2, center2 = feat_trans_dgcnn(point_input=x1, kernel=self.matrice2, m=self.m2)
        score2 = self.scorenet2(scorenet_input, calc_scores=self.calc_scores, bias=0)
        x_asm = assemble_dgcnn(score=score2, point_input=x2, center_input=center2, knn_idx=idx, aggregate='sum')
        x2 = F.relu(self.bn2(x_asm))

        x3, center3 = feat_trans_dgcnn(point_input=x2, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(scorenet_input, calc_scores=self.calc_scores, bias=0)
        x_asm = assemble_dgcnn(score=score3, point_input=x3, center_input=center3, knn_idx=idx, aggregate='sum')
        x3 = F.relu(self.bn3(x_asm))

        x4, center4 = feat_trans_dgcnn(point_input=x3, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(scorenet_input, calc_scores=self.calc_scores, bias=0)
        x_asm = assemble_dgcnn(score=score4, point_input=x4, center_input=center4, knn_idx=idx, aggregate='sum')
        x4 = F.relu(self.bn4(x_asm))

        x5, center5 = feat_trans_dgcnn(point_input=x4, kernel=self.matrice5, m=self.m5)
        score5 = self.scorenet5(scorenet_input, calc_scores=self.calc_scores, bias=0)
        x_asm = assemble_dgcnn(score=score5, point_input=x5, center_input=center5, knn_idx=idx, aggregate='sum')
        x5 = F.relu(self.bn5(x_asm))

        # ---------------------------------------------------------------------
        # [Step 4: 로컬(Local) 및 글로벌(Global) 특징 융합]
        # ---------------------------------------------------------------------
        # 모든 레이어의 로컬 특징을 연결(Concatenation) -> (B, 320, N)
        xx = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # 글로벌 맥락(Global Context) 추출: 3D 형태 전체를 아우르는 1024차원 특징 벡터 생성
        xc = F.relu(self.convt(xx))
        xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)
        
        # 각 개별 점의 로컬 특징(xx)에 전체 모델의 글로벌 특징(cls)을 붙여줌 (PointNet 아키텍처 철학)
        cls = xc.view(B, 1024, 1).repeat(1, 1, N)
        x_concat = torch.cat((xx, cls), dim=1)
        
        # ---------------------------------------------------------------------
        # [Step 5: Heatmap 출력 (Regression Point)]
        # ---------------------------------------------------------------------
        x_res = F.relu(self.conv6(x_concat))
        x_res = self.dp1(x_res)
        x_res = F.relu(self.conv7(x_res))
        x_res = self.dp2(x_res)
        x_res = F.relu(self.conv8(x_res))
        
        # 최종 채널 수를 랜드마크 개수(landmark_num)로 맞춤
        x_res = self.conv9(x_res) 
        
        # 🌟 핵심: Softmax를 채널(랜드마크) 방향이 아닌, 점(Point, dim=1) 방향으로 적용
        # 이유: "이 점이 무슨 랜드마크인가?"(Classification)를 찾는 것이 아니라,
        # "1번 랜드마크가 이 N개의 점들 중 어디에 있을 확률이 높은가?"(Heatmap Regression)를 찾는 것이기 때문임.
        x_res = F.softmax(x_res, dim=1)  
        
        return x_res