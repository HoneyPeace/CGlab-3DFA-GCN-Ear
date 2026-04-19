import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    B, _, N = x.size()                                   
    inner = -2 * torch.matmul(x.transpose(2, 1), x)      
    xx = torch.sum(x ** 2, dim=1, keepdim=True)          
    pairwise_distance = -xx - inner - xx.transpose(2, 1) 
    _, idx = pairwise_distance.topk(k=k, dim=-1)         
    return idx, pairwise_distance                        

def get_graph_feature(x, k=20, idx=None):
    batch_size, C_in, num_points = x.size()             
    
    xyz = x[:, :3, :]
    if idx is None:
        idx, _ = knn(xyz, k=k)                          

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  
    idx = idx + idx_base                                
    idx = idx.view(-1)                                  

    x_trans = x.transpose(2, 1).contiguous()            
    
    neighbor = x_trans.view(batch_size * num_points, -1)[idx, :]   
    neighbor = neighbor.view(batch_size, num_points, k, C_in)  
    center = x_trans.view(batch_size, num_points, 1, C_in).repeat(1, 1, k, 1)  

    neighbor_xyz = neighbor[..., :3]
    center_xyz = center[..., :3]
    relative_xyz = neighbor_xyz - center_xyz
    dist = torch.linalg.vector_norm(relative_xyz, dim=3, keepdim=True)

    # 🌟 [수정 1] 상대적 기하 정보 복원 (1.7mm 성능의 핵심)
    # 단순히 중심점의 기하정보(center_geom)만 쓰는 것이 아니라,
    # 이웃 점과의 주방향(Tangent) 및 곡률 차이(relative_geom)를 사용하여 엣지 특징을 극대화합니다.
    if C_in == 7:
        relative_geom = neighbor[..., 3:] - center[..., 3:]
        feature = torch.cat((relative_xyz, neighbor_xyz, center_xyz, dist, relative_geom), dim=3)
    elif C_in == 6:
        relative_v = neighbor[..., 3:] - center[..., 3:]
        feature = torch.cat((relative_xyz, neighbor_xyz, center_xyz, dist, relative_v), dim=3)
    else:
        feature = torch.cat((relative_xyz, neighbor_xyz, center_xyz, dist), dim=3)

    return feature.permute(0, 3, 1, 2).contiguous()     

def get_scorenet_input(x, idx=None, k=20):
    """
    🌟 [수정 2] 텐서 차원 검사 (연산량 반토막 튜닝)
    PAConv_model.py에서 4차원 텐서(이미 연산된 엣지 피처)가 들어오면 
    중복 연산 없이 그대로 패스합니다.
    """
    if len(x.shape) == 4:
        return x
    return get_graph_feature(x, k=k, idx=idx)

def feat_trans_dgcnn(point_input, kernel, m):
    B, _, N = point_input.size()                          
    point_output = torch.matmul(
        point_input.permute(0, 2, 1).repeat(1, 1, 2),     
        kernel                                            
    ).view(B, N, m, -1)                                   

    center_output = torch.matmul(
        point_input.permute(0, 2, 1),                     
        kernel[:point_input.size(1)]                      
    ).view(B, N, m, -1)                                   

    return point_output, center_output                    

def feat_trans_pointnet(point_input, kernel, m):
    B, _, N = point_input.size()  
    point_output = torch.matmul(point_input.permute(0, 2, 1), kernel).view(B, N, m, -1)  
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
        B, _, N, K = xyz.size()  
        scores = xyz             

        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            if self.last_bn:
                scores = self.mlp_bns_nohidden(self.mlp_convs_nohidden(scores))  
            else:
                scores = self.mlp_convs_nohidden(scores)                         
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

        scores = scores.permute(0, 2, 3, 1)             
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
        B, C, N = x.size()
        y = self.avg_pool(x)           
        y = y.view(B, C)               
        y = self.fc(y).view(B, C, 1)   
        return x * y.expand_as(x)