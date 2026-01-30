import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    B, _, N = x.size()                                   # x: (B, C, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)      # x^T x: (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)          # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # (B, N, N)

    _, idx = pairwise_distance.topk(k=k, dim=-1)         # idx: (B, N, K)

    return idx, pairwise_distance                        # idx:(B,N,K), dist:(B,N,N)



def get_graph_feature(x, k=30, idx=None):
    """
    x: input points (B, C, N) -> C=6 (XYZ+Normal)
    return: edge features (B, 19, N, K) 
    Structure: [XYZ_Feat(10)] + [Normal_Feat(9)]
    """
    batch_size, num_dims, num_points = x.size()         
    
    # [1] 데이터 분리
    xyz = x[:, :3, :]      # (B, 3, N)
    normals = x[:, 3:, :]  # (B, 3, N)

    # [2] KNN은 오직 'XYZ 좌표'로만 수행 (공간적 이웃 찾기)
    if idx is None:
        idx, _ = knn(xyz, k=k)                  # idx: (B, N, K)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    # -----------------------------------------------------------------
    # [Part A] XYZ Geometric Features (10채널)
    # -----------------------------------------------------------------
    xyz_t = xyz.transpose(2, 1).contiguous()   # (B, N, 3)
    
    neighbor_xyz = xyz_t.view(batch_size * num_points, -1)[idx, :]
    neighbor_xyz = neighbor_xyz.view(batch_size, num_points, k, 3) 
    
    center_xyz = xyz_t.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1) 
    
    dist = torch.linalg.vector_norm(neighbor_xyz - center_xyz, dim=3, keepdim=True)
    
    # XYZ 10ch: [Diff(3), Neighbor(3), Center(3), Dist(1)]
    feat_xyz = torch.cat((neighbor_xyz - center_xyz, neighbor_xyz, center_xyz, dist), dim=3)

    # -----------------------------------------------------------------
    # [Part B] Normal Geometric Features (9채널) - 복구됨!
    # -----------------------------------------------------------------
    norm_t = normals.transpose(2, 1).contiguous() # (B, N, 3)
    
    # 1. Neighbor Normal
    neighbor_norm = norm_t.view(batch_size * num_points, -1)[idx, :]
    neighbor_norm = neighbor_norm.view(batch_size, num_points, k, 3) 

    # 2. Center Normal
    center_norm = norm_t.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1)
    
    # 3. Normal Diff (곡률 정보 반영)
    diff_norm = neighbor_norm - center_norm
    
    # Normal 9ch: [Diff(3), Neighbor(3), Center(3)]
    feat_norm = torch.cat((diff_norm, neighbor_norm, center_norm), dim=3)

    # -----------------------------------------------------------------
    # [Part C] 최종 결합 (10 + 9 = 19채널)
    # -----------------------------------------------------------------
    feature = torch.cat((feat_xyz, feat_norm), dim=3) 
    
    return feature.permute(0, 3, 1, 2).contiguous()     # (B, 19, N, K)


def get_scorenet_input(x, idx, k):
    """
    x: input points (B, C, N)
    return: (B, 19, N, K)
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    # [1] 데이터 분리
    xyz = x[:, :3, :]
    normals = x[:, 3:, :]

    # [Part A] XYZ Features (10ch)
    xyz_t = xyz.transpose(2, 1).contiguous()
    neighbor_xyz = xyz_t.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, k, 3)
    center_xyz = xyz_t.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1)
    
    dist = torch.linalg.vector_norm(neighbor_xyz - center_xyz, dim=3, keepdim=True)
    feat_xyz = torch.cat((neighbor_xyz - center_xyz, neighbor_xyz, center_xyz, dist), dim=3)

    # [Part B] Normal Features (9ch)
    norm_t = normals.transpose(2, 1).contiguous()
    neighbor_norm = norm_t.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, k, 3)
    center_norm = norm_t.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1)
    
    feat_norm = torch.cat((neighbor_norm - center_norm, neighbor_norm, center_norm), dim=3)

    # [Part C] Combine (19ch)
    feature = torch.cat((feat_xyz, feat_norm), dim=3)
    
    return feature.permute(0, 3, 1, 2).contiguous()     # (B, 19, N, K)



def feat_trans_dgcnn(point_input, kernel, m):
    """transforming features using weight matrices"""
    # following get_graph_feature in DGCNN: torch.cat((neighbor - center, neighbor), dim=3)
    B, _, N = point_input.size()                          # point_input: (B, Cin, N)
    # kernel: (2*Cin, m*Cout)  으로 초기화되어 있음

    point_output = torch.matmul(
        point_input.permute(0, 2, 1).repeat(1, 1, 2),     # (B, N, 2*Cin)
        kernel                                            # (2*Cin, m*Cout)
    ).view(B, N, m, -1)                                   # (B, N, m, Cout)

    center_output = torch.matmul(
        point_input.permute(0, 2, 1),                     # (B, N, Cin)
        kernel[:point_input.size(1)]                      # (Cin, m*Cout)
    ).view(B, N, m, -1)                                   # (B, N, m, Cout)

    return point_output, center_output                    # 둘 다 (B, N, m, Cout)


def feat_trans_pointnet(point_input, kernel, m):
    """transforming features using weight matrices"""
    # no feature concat, following PointNet
    B, _, N = point_input.size()  # b, cin, n
    point_output = torch.matmul(point_input.permute(0, 2, 1), kernel).view(B, N, m, -1)  # b,n,m,cout
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
            self.mlp_convs_hidden.append(nn.Conv2d(in_channel, hidden_unit[0], 1, bias=False))  # from in_channel to first hidden
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):  # from 2nd hidden to next hidden to last hidden
                self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1, bias=False))
                self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[-1], out_channel, 1, bias=not last_bn))  # from last hidden to out_channel
            self.mlp_bns_hidden.append(nn.BatchNorm2d(out_channel))

    def forward(self, xyz, calc_scores='softmax', bias=0):
        B, _, N, K = xyz.size()  # (B,in_ch,N,K)
        scores = xyz             # (B,in_ch,N,K)

        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            if self.last_bn:
                scores = self.mlp_bns_nohidden(self.mlp_convs_nohidden(scores))  # (B,out_ch,N,K)
            else:
                scores = self.mlp_convs_nohidden(scores)                         # (B,out_ch,N,K)
        else:
            for i, conv in enumerate(self.mlp_convs_hidden):
                if i == len(self.mlp_convs_hidden)-1:  # if the output layer, no ReLU
                    if self.last_bn:
                        bn = self.mlp_bns_hidden[i]
                        scores = bn(conv(scores))      # (B,out_ch,N,K)
                    else:
                        scores = conv(scores)          # (B,out_ch,N,K)
                else:
                    bn = self.mlp_bns_hidden[i]
                    scores = F.relu(bn(conv(scores)))  # (B,hidden_ch,N,K)

        if calc_scores == 'softmax':
            scores = F.softmax(scores, dim=1)+bias      # (B,out_ch,N,K)
        elif calc_scores == 'sigmoid':
            scores = torch.sigmoid(scores)+bias         # (B,out_ch,N,K)
        else:
            raise ValueError('Not Implemented!')

        scores = scores.permute(0, 2, 3, 1)             # (B,N,K,out_ch)

        return scores                                   # (B,N,K,out_ch)

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
        # x shape: (B, C, N)
        B, C, N = x.size()
        y = self.avg_pool(x)           # shape: (B, C, 1)
        y = y.view(B, C)               # shape: (B, C)
        y = self.fc(y).view(B, C, 1)   # shape: (B, C, 1)
        return x * y.expand_as(x)      # attention applied