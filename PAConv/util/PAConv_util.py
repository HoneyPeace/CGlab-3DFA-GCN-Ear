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



def get_graph_feature(x, k=20, idx=None):
    """
    x: input points (B, C, N)
    return: edge features (B, 2*C, N, K)
    """
    batch_size, num_dims, num_points = x.size()         # (B, C, N)
    if idx is None:
        idx, _ = knn(x, k=k)                            # idx: (B, N, K)

    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # (B,1,1)
    idx = idx + idx_base                                # (B, N, K) + base offset
    idx = idx.view(-1)                                  # (B*N*K,)

    x = x.transpose(2, 1).contiguous()                  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]   # (B*N*K, C) 인덱싱
    feature = feature.view(batch_size, num_points, k, num_dims)  # (B, N, K, C) = neighbor

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # center: (B, N, K, C)

    dist = torch.linalg.vector_norm(feature - x, dim = 3, keepdim=True)
    #feature = torch.cat((feature - x, x), dim=3)  # (B, N, K, 2*C) 6채널
    feature = torch.cat((feature - x, feature, x, dist), dim=3)  # 결과: (B, N, K, 10) 10채널
    """
    feature = torch.cat((
        feature - x,  # (B, N, K, 3)
        feature,      # (B, N, K, 3)
        x,            # (B, N, K, 3)
        dist          # (B, N, K, 1)
    ), dim=3)         # 결과: (B, N, K, 10)
    """
    return feature.permute(0, 3, 1, 2).contiguous()     # (B, 10, N, K)


def get_scorenet_input(x, idx, k):

    batch_size = x.size(0)                                # B
    num_points = x.size(2)                                # N
    x = x.view(batch_size, -1, num_points)                # (B, C, N)

    device = torch.device('cuda')  # 기존 코드 유지

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # (B,1,1)
    idx = idx + idx_base                                 # (B, N, K)
    idx = idx.view(-1)                                   # (B*N*K,)

    _, num_dims, _ = x.size()                            # num_dims = C (=3)

    x = x.transpose(2, 1).contiguous()                   # (B, N, C)
    neighbor = x.view(batch_size * num_points, -1)[idx, :]\
                .view(batch_size, num_points, k, num_dims)   # (B, N, K, C)
    center = x.view(batch_size, num_points, 1, num_dims)\
             .repeat(1, 1, k, 1)                         # (B, N, K, C)
    dist = torch.linalg.vector_norm(neighbor - center, dim = 3, keepdim=True)
    #feature = torch.cat((neighbor - center, neighbor), dim=3)  # (B, N, K, 2*C) 6채널
    feature = torch.cat((neighbor - center, neighbor, center, dist), dim=3)  # 결과: (B, N, K, 10) 10채널
    """
    feature = torch.cat((
        neighbor - center, # (B, N, K, 3)
        neighbor,          # (B, N, K, 3)
        center,            # (B, N, K, 3)
        dist               # (B, N, K, 1)
    ), dim=3)              # 결과: (B, N, K, 10)
    """
    return feature.permute(0, 3, 1, 2).contiguous()     # (B, 2*C, N, K) = (B, 6, N, K)



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