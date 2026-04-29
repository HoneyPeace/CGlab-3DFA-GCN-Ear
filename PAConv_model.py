import torch
import torch.nn as nn
import torch.nn.functional as F
from My_args import *
from PAConv.util.PAConv_util import knn, get_graph_feature, get_scorenet_input, feat_trans_dgcnn, ScoreNet, Attention_Layer
from PAConv.cuda_lib.functional import assign_score_withk as assemble_dgcnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
================================================================================
[PAConv Feature Extractor - Universal Injection판]
- 모듈명: PAConv (Position Adaptive Convolution) 
- 🌟 핵심 업데이트: 
  args.use_raw_injection 옵션에 따라 정보 전달 방식을 우아하게 전환합니다.
  True: 1344차원 날것 그대로(Raw) DeepPA에 전달 (교수님 권장/순수 지식)
  False: 기존 4단계(64,128,256,512) 정제 후 전달 (VRAM 절약/기존 방식)
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
        
        # 🌟 옵션 스위치 장착
        self.use_raw_injection = getattr(args, 'use_raw_injection', False)
        
        # 채널 동기화 로직
        in_channels = getattr(args, 'in_channels', 3)
        if in_channels == 14: 
            self.edge_channels = 14 
        elif in_channels == 10: 
            self.edge_channels = 10 
        else: 
            self.edge_channels = 6 
        
        self.scorenet2 = ScoreNet(self.edge_channels, self.m2, hidden_unit=self.hidden[0])
        self.scorenet3 = ScoreNet(self.edge_channels, self.m3, hidden_unit=self.hidden[1])
        self.scorenet4 = ScoreNet(self.edge_channels, self.m4, hidden_unit=self.hidden[2])
        self.scorenet5 = ScoreNet(self.edge_channels, self.m5, hidden_unit=self.hidden[3])

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.edge_channels, 64, kernel_size=1, bias=False), 
            self.bn1, 
            nn.LeakyReLU(negative_slope=0.2)
        )

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

        self.convt = nn.Sequential(
            nn.Conv1d(320, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024)
        )
        
        # ====================================================================
        # 🌟 [뿌리 수정 1] Raw Injection이 "아닐 때만" 투영 레이어를 생성합니다.
        # 이렇게 하면 불필요한 파라미터가 모델에 등재되지 않아 매우 깔끔해집니다.
        # ====================================================================
        if not self.use_raw_injection:
            def make_refinement_proj(in_c, out_c):
                return nn.Sequential(
                    nn.Conv1d(in_c, out_c, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_c),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(out_c, out_c, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_c),
                    nn.LeakyReLU(0.2, inplace=True)
                )

            self.proj_st1 = make_refinement_proj(1344, 64)
            self.proj_st2 = make_refinement_proj(1344, 128)
            self.proj_st3 = make_refinement_proj(1344, 256)
            self.proj_st4 = make_refinement_proj(1344, 512)
        
        # ====================================================================
        # [기존 유지] 자체 학습 및 히트맵 닻(Anchor) 생성을 위한 경로
        # ====================================================================
        self.conv6 = nn.Sequential(nn.Conv1d(1344, 512, kernel_size=1, bias=False), nn.BatchNorm1d(512))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv7 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False), nn.BatchNorm1d(256))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128))
        self.conv9 = nn.Conv1d(128, self.landmark_num, kernel_size=1, bias=True) 

    def forward(self, xyz, feature=None):
        B, C, N = xyz.shape

        xyz_coords = xyz[:, :3, :].contiguous() 
        idx, _ = knn(xyz_coords, self.k)[cite: 5]

        x_edge_feat = get_graph_feature(xyz, k=self.k, idx=idx)[cite: 5]
        scorenet_input = get_scorenet_input(x_edge_feat, idx=idx, k=self.k)[cite: 5]

        x1 = self.conv1(x_edge_feat)[cite: 5]
        x1 = x1.max(dim=-1, keepdim=False)[0][cite: 5]

        x2, center2 = feat_trans_dgcnn(point_input=x1, kernel=self.matrice2, m=self.m2)[cite: 5]
        score2 = self.scorenet2(scorenet_input, calc_scores=self.calc_scores, bias=0)[cite: 5]
        x_asm = assemble_dgcnn(score=score2, point_input=x2, center_input=center2, knn_idx=idx, aggregate='sum')[cite: 5]
        x2 = F.relu(self.bn2(x_asm))[cite: 5]

        x3, center3 = feat_trans_dgcnn(point_input=x2, kernel=self.matrice3, m=self.m3)[cite: 5]
        score3 = self.scorenet3(scorenet_input, calc_scores=self.calc_scores, bias=0)[cite: 5]
        x_asm = assemble_dgcnn(score=score3, point_input=x3, center_input=center3, knn_idx=idx, aggregate='sum')[cite: 5]
        x3 = F.relu(self.bn3(x_asm))[cite: 5]

        x4, center4 = feat_trans_dgcnn(point_input=x3, kernel=self.matrice4, m=self.m4)[cite: 5]
        score4 = self.scorenet4(scorenet_input, calc_scores=self.calc_scores, bias=0)[cite: 5]
        x_asm = assemble_dgcnn(score=score4, point_input=x4, center_input=center4, knn_idx=idx, aggregate='sum')[cite: 5]
        x4 = F.relu(self.bn4(x_asm))[cite: 5]

        x5, center5 = feat_trans_dgcnn(point_input=x4, kernel=self.matrice5, m=self.m5)[cite: 5]
        score5 = self.scorenet5(scorenet_input, calc_scores=self.calc_scores, bias=0)[cite: 5]
        x_asm = assemble_dgcnn(score=score5, point_input=x5, center_input=center5, knn_idx=idx, aggregate='sum')[cite: 5]
        x5 = F.relu(self.bn5(x_asm))[cite: 5]

        xx = torch.cat((x1, x2, x3, x4, x5), dim=1)[cite: 5]

        xc = F.relu(self.convt(xx))[cite: 5]
        xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)[cite: 5]
        
        cls = xc.view(B, 1024, 1).repeat(1, 1, N)[cite: 5]
        x_concat = torch.cat((xx, cls), dim=1)[cite: 5]
        
        # 🌟 [뿌리 수정 2] 조건에 따라 넘겨주는 포맷을 분기 처리합니다.
        if self.use_raw_injection:
            # 교수님 권장 방식: 날것(1344 채널) 통째로 전달
            prior_hints = x_concat
        else:
            # 기존 방식: 4단계 다중 해상도로 깎아서 리스트로 전달
            hint_st1 = self.proj_st1(x_concat) 
            hint_st2 = self.proj_st2(x_concat) 
            hint_st3 = self.proj_st3(x_concat) 
            hint_st4 = self.proj_st4(x_concat) 
            prior_hints = [hint_st1, hint_st2, hint_st3, hint_st4]
        
        # 자체 학습 및 히트맵 예측용 라텐트 압축
        x_res = F.relu(self.conv6(x_concat))[cite: 5]
        x_res = self.dp1(x_res)[cite: 5]
        x_res = F.relu(self.conv7(x_res))   [cite: 5]
        x_res = self.dp2(x_res)[cite: 5]
        latent_hint = F.relu(self.conv8(x_res))[cite: 5]
        heatmap_anchor = self.conv9(latent_hint)[cite: 5]
        
        return prior_hints, heatmap_anchor