import torch
import torch.nn as nn
import torch.nn.functional as F
# [주의]: My_args와 PAConv_util 등은 연구자님의 로컬 환경 경로에 맞게 임포트 유지
from My_args import *
from PAConv.util.PAConv_util import knn, get_graph_feature, get_scorenet_input, feat_trans_dgcnn, ScoreNet, Attention_Layer
from PAConv.cuda_lib.functional import assign_score_withk as assemble_dgcnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
================================================================================
[NotebookLM을 위한 학술적 의도 및 모듈 요약]
- 모듈명: PAConv (Position Adaptive Convolution) Feature Extractor
- 논문 내 역할: DeepPA(2단계) 아키텍처의 전반부(Stage 1)를 담당하며, 
              점과 점 사이의 기하학적 위치 관계를 동적으로 학습하여 
              전역적인(Global) 맥락을 파악하는 최전방 탐색기(Vanguard)입니다.
- 핵심 구조 변화 (Heatmap -> Latent Feature & Anchor):
  기존에는 최종 레이어에서 랜드마크의 '확률(Heatmap)' 하나만 내뱉었으나, 
  개선된 본 구조에서는 [분기형(Two-Head) 아키텍처]를 도입했습니다.
  1) DeepPA 게이트 융합을 위한 128차원의 순수 고차원 특징(Raw Latent Feature) 방출
  2) 최종 좌표 공간 제어를 위한 36차원 앵커 히트맵(Anchor Heatmap) 동시 방출
================================================================================
'''
class PAConv(nn.Module):
    def __init__(self, args, landmark_num):
        super(PAConv, self).__init__()
        self.args = args
        self.k = args.k
        self.landmark_num = landmark_num
        self.calc_scores = args.calc_scores
        self.hidden = args.hidden
        self.m2, self.m3, self.m4, self.m5 = args.num_matrices # Weight Banks(가중치 행렬) 개수
        
        # [채널 동기화 핵심 로직]
        in_channels = getattr(args, 'in_channels', 3)
        if in_channels == 7:
            self.edge_channels = 14
        elif in_channels == 6:
            self.edge_channels = 13
        else:
            self.edge_channels = 10 
        
        # [1. ScoreNets 정의] 
        self.scorenet2 = ScoreNet(self.edge_channels, self.m2, hidden=self.hidden)
        self.scorenet3 = ScoreNet(self.edge_channels, self.m3, hidden=self.hidden)
        self.scorenet4 = ScoreNet(self.edge_channels, self.m4, hidden=self.hidden)
        self.scorenet5 = ScoreNet(self.edge_channels, self.m5, hidden=self.hidden)

        # [2. 초기 특징 추출용 MLP]
        self.bn1 = nn.BatchNorm2d(64)
        if in_channels == 7:
            self.conv1 = nn.Sequential(nn.Conv2d(14, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        elif in_channels == 6:
            self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))

        # [3. PAConv 기반 특징 추출 레이어]
        self.matrice2 = nn.Parameter(torch.FloatTensor(64, 64, self.m2))
        nn.init.kaiming_normal_(self.matrice2, mode='fan_out', nonlinearity='relu')
        self.bn2 = nn.BatchNorm1d(64)

        self.matrice3 = nn.Parameter(torch.FloatTensor(64, 64, self.m3))
        nn.init.kaiming_normal_(self.matrice3, mode='fan_out', nonlinearity='relu')
        self.bn3 = nn.BatchNorm1d(64)

        self.matrice4 = nn.Parameter(torch.FloatTensor(64, 64, self.m4))
        nn.init.kaiming_normal_(self.matrice4, mode='fan_out', nonlinearity='relu')
        self.bn4 = nn.BatchNorm1d(64)

        self.matrice5 = nn.Parameter(torch.FloatTensor(64, 64, self.m5))
        nn.init.kaiming_normal_(self.matrice5, mode='fan_out', nonlinearity='relu')
        self.bn5 = nn.BatchNorm1d(64)

        # [4. 글로벌 맥락(Global Context) 추출 레이어]
        self.convt = nn.Conv1d(320, 1024, kernel_size=1)
        
        # [5. 융합 및 출력 (Feature Fusion & Output Heads)]
        self.conv6 = nn.Conv1d(1344, 512, 1)
        self.dp1 = nn.Dropout(p=0.5)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.dp2 = nn.Dropout(p=0.5)
        
        # 🌟 [디펜스 포인트: Two-Head 출력 정의]
        # 1) DeepPA 게이트 융합을 위한 고밀도 힌트 생성기 (256 -> 128)
        self.conv8 = nn.Conv1d(256, 128, 1)
        
        # 2) 최종 회귀 헤드 앵커용 히트맵 로짓 생성기 (128 -> 36)
        # Softmax를 제거하고 선형 로짓(Logit) 상태로 뱉어냅니다.
        self.conv9 = nn.Conv1d(128, self.landmark_num, 1) 

    def forward(self, xyz, feature=None):
        B, N, C = xyz.shape

        # 채널 동기화 기반 초기 Feature 세팅
        if C == 7:
            x_input = xyz.clone().permute(0, 2, 1)
        elif C == 6:
            x_input = xyz.clone().permute(0, 2, 1)
        else:
            x_input = xyz.clone().permute(0, 2, 1)
        
        xyz = xyz[:, :, :3] # K-NN 검색용 순수 좌표 추출
        
        # ---------------------------------------------------------------------
        # [Step 1: 입력 데이터 구성 (Edge Feature 추출)]
        # ---------------------------------------------------------------------
        idx = knn(xyz, self.k) # 각 점 주변의 K개 이웃 인덱스
        x1 = get_graph_feature(x_input, k=self.k, idx=idx) 
        scorenet_input = get_scorenet_input(x1) # PAConv 가중치 스코어링용 별도 입력 생성

        # ---------------------------------------------------------------------
        # [Step 2: 초기 특징 추출]
        # ---------------------------------------------------------------------
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0] # PointNet 기반 Max Pooling

        # ---------------------------------------------------------------------
        # [Step 3: PAConv 레이어 통과 (위치-적응형 특징 추출)]
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
        xx = torch.cat((x1, x2, x3, x4, x5), dim=1) # (B, 320, N)

        xc = F.relu(self.convt(xx))
        xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)
        
        cls = xc.view(B, 1024, 1).repeat(1, 1, N)
        x_concat = torch.cat((xx, cls), dim=1) # (B, 1344, N)
        
        # ---------------------------------------------------------------------
        # [Step 5: 🌟 분기형 Two-Head 출력 (Feature Extraction Point)]
        # ---------------------------------------------------------------------
        x_res = F.relu(self.conv6(x_concat)) # 1344 -> 512
        x_res = self.dp1(x_res)
        x_res = F.relu(self.conv7(x_res))    # 512 -> 256
        x_res = self.dp2(x_res)
        
        # 1. DeepPA 게이트 잔차 융합을 위한 고밀도 라텐트 힌트 (128채널)
        latent_hint = F.relu(self.conv8(x_res)) # (B, 128, N)
        
        # 2. 최종단 앵커(Anchor) 연결을 위한 거시적 히트맵 (36채널)
        # Softmax 삭제 (정보 병목 방지)
        heatmap_anchor = self.conv9(latent_hint) # (B, 36, N)
        
        # DeepPA_model.py 규격에 맞게 (B, N, C) 형태로 트랜스포즈 후 방출
        latent_hint = latent_hint.permute(0, 2, 1).contiguous()
        heatmap_anchor = heatmap_anchor.permute(0, 2, 1).contiguous()
        
        return latent_hint, heatmap_anchor