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
    x: input points (B, C, N) - 3채널 또는 6채널 입력
    return: edge features (B, 10, N, K) 또는 (B, 16, N, K)
    """
    batch_size, C_in, num_points = x.size()             # C_in은 3 또는 6
    
    # 🌟 KNN 거리 계산은 무조건 앞의 3채널(순수 xyz 물리 좌표)로만 수행!
    xyz = x[:, :3, :]
    if idx is None:
        idx, _ = knn(xyz, k=k)                          # idx: (B, N, K)

    device = x.device
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # (B,1,1)
    idx = idx + idx_base                                # (B, N, K) + base offset
    idx = idx.view(-1)                                  # (B*N*K,)

    x_trans = x.transpose(2, 1).contiguous()            # (B, N, C_in)
    
    # (B, N, K, C_in) 형태로 이웃 점과 중심 점 전개
    neighbor = x_trans.view(batch_size * num_points, -1)[idx, :]   
    neighbor = neighbor.view(batch_size, num_points, k, C_in)  
    center = x_trans.view(batch_size, num_points, 1, C_in).repeat(1, 1, k, 1)  

    # 물리 좌표(xyz) 추출 및 거리 계산
    neighbor_xyz = neighbor[..., :3]
    center_xyz = center[..., :3]
    relative_xyz = neighbor_xyz - center_xyz
    dist = torch.linalg.vector_norm(relative_xyz, dim=3, keepdim=True)

    # 🌟 6채널(좌표+방향) 처리 로직
    if C_in == 6:
        neighbor_v = neighbor[..., 3:]
        center_v = center[..., 3:]
        relative_v = neighbor_v - center_v # 이웃으로 갈 때 곡률의 꺾임 정도
        
        # 16채널 조립: 상대위치(3) + 이웃위치(3) + 중심위치(3) + 거리(1) + 중심방향(3) + 방향꺾임(3)
        feature = torch.cat((relative_xyz, neighbor_xyz, center_xyz, dist, center_v, relative_v), dim=3)
    else:
        # 기존 10채널 조립
        feature = torch.cat((relative_xyz, neighbor_xyz, center_xyz, dist), dim=3)

    return feature.permute(0, 3, 1, 2).contiguous()     # (B, feature_C, N, K)


def get_scorenet_input(x, idx, k):
    batch_size, C_in, num_points = x.size()

    device = x.device  

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # (B,1,1)
    idx = idx + idx_base                                 # (B, N, K)
    idx = idx.view(-1)                                   # (B*N*K,)

    x_trans = x.transpose(2, 1).contiguous()                   # (B, N, C_in)
    neighbor = x_trans.view(batch_size * num_points, -1)[idx, :]\
                .view(batch_size, num_points, k, C_in)   # (B, N, K, C_in)
    center = x_trans.view(batch_size, num_points, 1, C_in)\
             .repeat(1, 1, k, 1)                         # (B, N, K, C_in)
             
    # 물리 좌표(xyz) 추출 및 거리 계산
    neighbor_xyz = neighbor[..., :3]
    center_xyz = center[..., :3]
    relative_xyz = neighbor_xyz - center_xyz
    dist = torch.linalg.vector_norm(relative_xyz, dim=3, keepdim=True)

    # 🌟 6채널(좌표+방향) 처리 로직
    if C_in == 6:
        neighbor_v = neighbor[..., 3:]
        center_v = center[..., 3:]
        relative_v = neighbor_v - center_v # 이웃으로 갈 때 곡률의 꺾임 정도
        
        # 16채널 조립: 상대위치(3) + 이웃위치(3) + 중심위치(3) + 거리(1) + 중심방향(3) + 방향꺾임(3)
        feature = torch.cat((relative_xyz, neighbor_xyz, center_xyz, dist, center_v, relative_v), dim=3)
    else:
        # 기존 10채널 조립
        feature = torch.cat((relative_xyz, neighbor_xyz, center_xyz, dist), dim=3)

    return feature.permute(0, 3, 1, 2).contiguous()     # (B, feature_C, N, K)



def feat_trans_dgcnn(point_input, kernel, m):
    """transforming features using weight matrices"""
    # point_input은 6채널이든 3채널이든 이 행렬 곱셈 레이어에서는 알아서 처리됩니다.
    B, _, N = point_input.size()                          # point_input: (B, Cin, N)

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
            self.mlp_convs_hidden.append(nn.Conv2d(in_channel, hidden_unit[0], 1, bias=False))  
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):  
                self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1, bias=False))
                self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[-1], out_channel, 1, bias=not last_bn))  
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
                if i == len(self.mlp_convs_hidden)-1:  
                    if self.last_bn:
                        bn = self.mlp_bns_hidden[i]
                        scores = bn(conv(scores))      
                    else:
                        scores = conv(scores)          
                else:
                    bn = self.mlp_bns_hidden[i]
                    scores = F.relu(bn(conv(scores)))  

        if calc_scores == 'softmax':
            scores = F.softmax(scores, dim=1)+bias      
        elif calc_scores == 'sigmoid':
            scores = torch.sigmoid(scores)+bias         
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