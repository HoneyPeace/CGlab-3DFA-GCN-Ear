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
        self.m2, self.m3, self.m4, self.m5 = args.num_matrices
        
        # 🌟 [수정 1] 채널 동기화 핵심 로직 (17채널 롤백 -> 14채널로 완벽 고정)
        in_channels = getattr(args, 'in_channels', 3)
        if in_channels == 7:
            self.edge_channels = 14 # 7채널(XYZ+방향+곡률)일 경우 엣지는 14채널
        elif in_channels == 6:
            self.edge_channels = 13 
        else:
            self.edge_channels = 10 
        
        self.scorenet2 = ScoreNet(self.edge_channels, self.m2, hidden_unit=self.hidden[0])
        self.scorenet3 = ScoreNet(self.edge_channels, self.m3, hidden_unit=self.hidden[1])
        self.scorenet4 = ScoreNet(self.edge_channels, self.m4, hidden_unit=self.hidden[2])
        self.scorenet5 = ScoreNet(self.edge_channels, self.m5, hidden_unit=self.hidden[3])

        # 🌟 [수정 2] 하드코딩 제거: 동적으로 self.edge_channels를 받도록 수정
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.edge_channels, 64, kernel_size=1, bias=False), 
            self.bn1, 
            nn.LeakyReLU(negative_slope=0.2)
        )

        # ---------------------------------------------------------------------
        # [3. PAConv 기반 특징 추출 레이어]
        # ---------------------------------------------------------------------
        self.matrice2 = nn.Parameter(torch.FloatTensor(64 * 2, 64 * self.m2))
        nn.init.kaiming_normal_(self.matrice2, mode='fan_out', nonlinearity='relu')
        self.bn2 = nn.BatchNorm1d(64)

        self.matrice3 = nn.Parameter(torch.FloatTensor(64 * 2, 64 * self.m3))
        nn.init.kaiming_normal_(self.matrice3, mode='fan_out', nonlinearity='relu')
        self.bn3 = nn.BatchNorm1d(64)

        self.matrice4 = nn.Parameter(torch.FloatTensor(64 * 2, 64 * self.m4))
        nn.init.kaiming_normal_(self.matrice4, mode='fan_out', nonlinearity='relu')
        self.bn4 = nn.BatchNorm1d(64)

        self.matrice5 = nn.Parameter(torch.FloatTensor(64 * 2, 64 * self.m5))
        nn.init.kaiming_normal_(self.matrice5, mode='fan_out', nonlinearity='relu')
        self.bn5 = nn.BatchNorm1d(64)

        # [4. 글로벌 맥락 추출 레이어]
        self.convt = nn.Conv1d(320, 1024, kernel_size=1)
        
        # [5. 융합 및 출력]
        self.conv6 = nn.Conv1d(1344, 512, 1)
        self.dp1 = nn.Dropout(p=0.5)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.dp2 = nn.Dropout(p=0.5)
        
        # Two-Head
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, self.landmark_num, 1) 

    def forward(self, xyz, feature=None):
        B, C, N = xyz.shape

        xyz_coords = xyz[:, :3, :].contiguous() 
        idx, _ = knn(xyz_coords, self.k) 

        # 🌟 [수정 3] get_scorenet_input 파라미터 에러 수정! (원본 xyz, idx, k 모두 전달)
        x_edge_feat = get_graph_feature(xyz, k=self.k, idx=idx) 
        scorenet_input = get_scorenet_input(xyz, idx, self.k)

        x1 = self.conv1(x_edge_feat) 
        x1 = x1.max(dim=-1, keepdim=False)[0] 

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

        xx = torch.cat((x1, x2, x3, x4, x5), dim=1) 

        xc = F.relu(self.convt(xx))
        xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)
        
        cls = xc.view(B, 1024, 1).repeat(1, 1, N)
        x_concat = torch.cat((xx, cls), dim=1) 
        
        x_res = F.relu(self.conv6(x_concat)) 
        x_res = self.dp1(x_res)
        x_res = F.relu(self.conv7(x_res))    
        x_res = self.dp2(x_res)
        
        latent_hint = F.relu(self.conv8(x_res)) 
        heatmap_anchor = self.conv9(latent_hint) 
        
        #latent_hint = latent_hint.permute(0, 2, 1).contiguous()
        #heatmap_anchor = heatmap_anchor.permute(0, 2, 1).contiguous()
        
        return latent_hint, heatmap_anchor