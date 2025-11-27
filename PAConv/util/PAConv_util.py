import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    B, _, N = x.size()
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)

    return idx, pairwise_distance



def get_graph_feature(x, k=20, idx=None):
    """
    x: input points (B, C, N)
    return: edge features (B, 2*C, N, k)
    """
    batch_size, num_dims, num_points = x.size()
    if idx is None:
        idx, _ = knn(x, k=k)  # (B, N, k)

    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)  # (B, 2*C, N, k)
    return feature


def get_scorenet_input(x, idx, k):
    """
    x: [B, C, N]
    return: [B, 10, N, K]   # 2025.06.06 변경됨 (기존: [B, 6, N, K])
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    device = torch.device('cuda')  # 기존 그대로 유지

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    neighbor = x.view(batch_size * num_points, -1)[idx, :].view(batch_size, num_points, k, num_dims)
    center = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((neighbor - center, neighbor), dim=3)

    return feature.permute(0, 3, 1, 2).contiguous()  # (B, 6, N, K) 



def feat_trans_dgcnn(point_input, kernel, m):
    """transforming features using weight matrices"""
    # following get_graph_feature in DGCNN: torch.cat((neighbor - center, neighbor), dim=3)
    B, _, N = point_input.size()  # b, 2cin, n
    point_output = torch.matmul(point_input.permute(0, 2, 1).repeat(1, 1, 2), kernel).view(B, N, m, -1)  # b,n,m,cout
    center_output = torch.matmul(point_input.permute(0, 2, 1), kernel[:point_input.size(1)]).view(B, N, m, -1)  # b,n,m,cout
    return point_output, center_output


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
        B, _, N, K = xyz.size()
        scores = xyz

        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            if self.last_bn:
                scores = self.mlp_bns_nohidden(self.mlp_convs_nohidden(scores))
            else:
                scores = self.mlp_convs_nohidden(scores)
        else:
            for i, conv in enumerate(self.mlp_convs_hidden):
                if i == len(self.mlp_convs_hidden)-1:  # if the output layer, no ReLU
                    if self.last_bn:
                        bn = self.mlp_bns_hidden[i]
                        scores = bn(conv(scores))
                    else:
                        scores = conv(scores)
                else:
                    bn = self.mlp_bns_hidden[i]
                    scores = F.relu(bn(conv(scores)))

        if calc_scores == 'softmax':
            scores = F.softmax(scores, dim=1)+bias  # B*m*N*K, where bias may bring larger gradient
        elif calc_scores == 'sigmoid':
            scores = torch.sigmoid(scores)+bias  # B*m*N*K
        else:
            raise ValueError('Not Implemented!')

        scores = scores.permute(0, 2, 3, 1)  # B*N*K*m

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
        # x shape: (B, C, N)
        B, C, N = x.size()
        y = self.avg_pool(x)           # shape: (B, C, 1)
        y = y.view(B, C)               # shape: (B, C)
        y = self.fc(y).view(B, C, 1)   # shape: (B, C, 1)
        return x * y.expand_as(x)      # attention applied