# @Author: Yuan Wang (Modified by Researcher Park Pyeong-hwa & AI Assistant)
# @File: PAConv_model.py
# @Description: 
# [PAConv Feature Extractor - Universal Injection 완결본]
# - 🌟 핵심 업데이트: My_args의 latent_injection_type 파라미터와 완벽 동기화
# - 🌟 [안정성] 'none' 모드 시 None 대신 빈 리스트([]) 반환하여 크래시 방지
# ================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from PAConv.util.PAConv_util import (
    knn,
    get_edge_feature_channels,
    get_graph_feature,
    get_scorenet_input,
    feat_trans_dgcnn,
    ScoreNet,
    Attention_Layer,
)
from PAConv.cuda_lib.functional import assign_score_withk as assemble_dgcnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PAConv(nn.Module):
    def __init__(self, args, landmark_num):
        super(PAConv, self).__init__()
        self.args = args
        self.k = args.k
        self.landmark_num = landmark_num
        self.calc_scores = args.calc_scores
        self.hidden = args.hidden
        self.m2, self.m3, self.m4, self.m5 = args.num_matrices
        
        # 🌟 파라미터 단일화 반영
        self.injection_type = getattr(args, 'latent_injection_type', 'raw').lower()
        
        in_channels = getattr(args, 'in_channels', 3)
        self.edge_channels = get_edge_feature_channels(in_channels)
        
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
        
        # 'compressed' 모드 전용 압축 헤드
        if self.injection_type == 'compressed':
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
        
        self.conv6 = nn.Sequential(nn.Conv1d(1344, 256, kernel_size=1, bias=False), nn.BatchNorm1d(256))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False), nn.BatchNorm1d(256))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False), nn.BatchNorm1d(128))
        self.conv9 = nn.Conv1d(128, self.landmark_num, kernel_size=1, bias=True) 

    def forward(self, xyz, feature=None):
        B, C, N = xyz.shape
        xyz_coords = xyz[:, :3, :].contiguous() 
        idx, _ = knn(xyz_coords, self.k)

        x_edge_feat = get_graph_feature(xyz, k=self.k, idx=idx)
        scorenet_input = get_scorenet_input(x_edge_feat, idx=idx, k=self.k)

        x1 = self.conv1(x_edge_feat).max(dim=-1, keepdim=False)[0]

        x2, center2 = feat_trans_dgcnn(point_input=x1, kernel=self.matrice2, m=self.m2)
        score2 = self.scorenet2(scorenet_input, calc_scores=self.calc_scores, bias=0)
        x2 = F.relu(self.bn2(assemble_dgcnn(score=score2, point_input=x2, center_input=center2, knn_idx=idx, aggregate='sum')))

        x3, center3 = feat_trans_dgcnn(point_input=x2, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(scorenet_input, calc_scores=self.calc_scores, bias=0)
        x3 = F.relu(self.bn3(assemble_dgcnn(score=score3, point_input=x3, center_input=center3, knn_idx=idx, aggregate='sum')))

        x4, center4 = feat_trans_dgcnn(point_input=x3, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(scorenet_input, calc_scores=self.calc_scores, bias=0)
        x4 = F.relu(self.bn4(assemble_dgcnn(score=score4, point_input=x4, center_input=center4, knn_idx=idx, aggregate='sum')))

        x5, center5 = feat_trans_dgcnn(point_input=x4, kernel=self.matrice5, m=self.m5)
        score5 = self.scorenet5(scorenet_input, calc_scores=self.calc_scores, bias=0)
        x5 = F.relu(self.bn5(assemble_dgcnn(score=score5, point_input=x5, center_input=center5, knn_idx=idx, aggregate='sum')))

        xx = torch.cat((x1, x2, x3, x4, x5), dim=1)
        xc = F.relu(self.convt(xx))
        xc = F.adaptive_max_pool1d(xc, 1).view(B, -1)
        cls = xc.view(B, 1024, 1).repeat(1, 1, N)
        x_concat = torch.cat((xx, cls), dim=1)
        
        # 🌟 분기 처리 (Return 값 안정화)
        if self.injection_type == 'raw':
            prior_hints = x_concat
        elif self.injection_type == 'compressed':
            hint_st1 = self.proj_st1(x_concat) 
            hint_st2 = self.proj_st2(x_concat) 
            hint_st3 = self.proj_st3(x_concat) 
            hint_st4 = self.proj_st4(x_concat) 
            prior_hints = [hint_st1, hint_st2, hint_st3, hint_st4]
        else: # 'none'
            prior_hints = [] # <--- 💥 None 대신 빈 리스트 반환하여 에러 방지
        
        # 히트맵 예측
        x_res = F.relu(self.conv6(x_concat))
        x_res = self.dp1(x_res)
        x_res = F.relu(self.conv7(x_res))   
        x_res = self.dp2(x_res)
        latent_hint = F.relu(self.conv8(x_res))
        heatmap_logits = self.conv9(latent_hint)
        heatmap_activation = getattr(self.args, 'paconv_heatmap_activation_mode', 'raw').lower()
        if heatmap_activation == 'softmax':
            heatmap_anchor = F.softmax(heatmap_logits, dim=2)
        elif heatmap_activation == 'sigmoid':
            heatmap_anchor = torch.sigmoid(heatmap_logits)
        elif heatmap_activation == 'raw':
            heatmap_anchor = heatmap_logits
        else:
            raise ValueError(f"Unsupported paconv_heatmap_activation_mode: {heatmap_activation}")
        
        return prior_hints, heatmap_anchor
