import torch
import torch.nn as nn
import torch.nn.functional as F
from My_args import *
from PAConv.util.PAConv_util import knn, get_graph_feature, get_scorenet_input, feat_trans_dgcnn, ScoreNet, Attention_Layer
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
        
        # 🌟 7채널(14엣지) 동기화
        in_channels = getattr(args, 'in_channels', 3)
        if in_channels == 7:
            self.edge_channels = 14
        elif in_channels == 6:
            self.edge_channels = 13
        else:
            self.edge_channels = 10
        
        self.scorenet2 = ScoreNet(self.edge_channels, self.m2, hidden_unit=self.hidden[0])
        self.scorenet3 = ScoreNet(self.edge_channels, self.m3, hidden_unit=self.hidden[1])
        self.scorenet4 = ScoreNet(self.edge_channels, self.m4, hidden_unit=self.hidden[2])
        self.scorenet5 = ScoreNet(self.edge_channels, self.m5, hidden_unit=self.hidden[3])
        
        i2 = 64       
        o2 = i3 = 64  
        o3 = i4 = 64  
        o4 = i5 = 64  
        o5 = 64       

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

        self.conv1 = nn.Sequential(nn.Conv2d(self.edge_channels, 64, kernel_size=1, bias=True), 
                                   nn.BatchNorm2d(64, momentum=0.1))
                                   
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
        self.conv9 = nn.Conv1d(128, landmark_num, kernel_size=1, bias=True)

        
    def forward(self, x):
        B, C, N = x.size()
        
        physical_xyz = x[:, :3, :].contiguous()
        idx, _ = knn(physical_xyz, k=self.k)
        
        scorenet_input = get_scorenet_input(x, k=self.k, idx=idx)  
        x_edge_feat = get_graph_feature(x, k=self.k, idx=idx)

        x_out = F.relu(self.conv1(x_edge_feat))
        x1 = x_out.max(dim=-1, keepdim=False)[0]
        
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
        x_res = F.relu(self.conv8(x_res))
        
        x_res = self.conv9(x_res) 
        x_res = F.softmax(x_res, dim=1)  
        
        return x_res