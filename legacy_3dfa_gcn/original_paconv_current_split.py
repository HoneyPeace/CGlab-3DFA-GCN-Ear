import argparse
import ast
import os
import random
import subprocess
import sys
import time
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PAConv.util.PAConv_util import ScoreNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_literal(value):
    if isinstance(value, (list, tuple)):
        return value
    return ast.literal_eval(value)


def set_seed(seed):
    seed = int(seed)
    if seed < 0:
        print("[INFO] Training seed free: skip random/np/torch seed fix")
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Training seed fixed: {seed}")


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight)


class OriginalAdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        delta_y = (target - pred).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = target[delta_y < self.theta]
        y2 = target[delta_y >= self.theta]

        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = (
            self.omega
            * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2)))
            * (self.alpha - y2)
            * torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)
        )
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.omega, self.alpha - y2)
        )
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def normalize_points(batch_data, landmark=None):
    xyz = batch_data[:, :, :3]
    centroid = torch.mean(xyz, axis=1, keepdim=True)
    centered = xyz - centroid
    scale = torch.max(torch.sqrt(torch.sum(centered ** 2, axis=2)), axis=1)[0].view(-1, 1, 1)
    normalized = centered / scale
    if landmark is None:
        return normalized
    landmark_norm = (landmark - centroid) / scale
    return normalized, landmark_norm, centroid, scale


class PointcloudScaleAndTranslate:
    def __init__(self, scale_low=2.0 / 3.0, scale_high=3.0 / 2.0, translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        bsize = pc.size(0)
        for i in range(bsize):
            scale = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            translate = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
            pc[i, :, 0:3] = (
                torch.mul(pc[i, :, 0:3], torch.from_numpy(scale).float().to(pc.device))
                + torch.from_numpy(translate).float().to(pc.device)
            )
        return pc


class CurrentSplitHeatmapDataset(Dataset):
    def __init__(self, data_root, data_name, partition, shuffle_points=False):
        self.data_root = data_root
        self.data_name = data_name
        self.partition = partition
        self.shuffle_points = shuffle_points

        base_dir = os.path.join(data_root, f"{data_name}-npy")
        self.shape = np.load(os.path.join(base_dir, f"shape_3ch_{partition}.npy"), allow_pickle=True)
        self.landmark = np.load(os.path.join(base_dir, f"landmark_{partition}.npy"), allow_pickle=True)
        self.heatmap = np.load(os.path.join(base_dir, f"Heat_data_{partition}.npy"), allow_pickle=True)
        name_path = os.path.join(base_dir, f"name_{partition}.npy")
        if os.path.exists(name_path):
            self.names = np.load(name_path, allow_pickle=True)
        else:
            self.names = np.array([f"S{i:03d}" for i in range(len(self.shape))])

    def __len__(self):
        return len(self.shape)

    def __getitem__(self, item):
        face = torch.tensor(self.shape[item], dtype=torch.float32)
        landmark = torch.tensor(self.landmark[item], dtype=torch.float32)
        heatmap = torch.tensor(self.heatmap[item], dtype=torch.float32)
        name = str(self.names[item])

        if self.shuffle_points:
            indices = torch.randperm(face.size(0))
            face = face[indices]
            heatmap = heatmap[indices]

        return face, landmark, heatmap, name


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)
    return idx, pairwise_distance


def gather_neighbors(x, idx):
    batch_size, channels, num_points = x.size()
    k = idx.size(-1)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx_flat = (idx + idx_base).view(-1)
    x_trans = x.transpose(2, 1).contiguous()
    neighbor = x_trans.view(batch_size * num_points, channels)[idx_flat, :]
    neighbor = neighbor.view(batch_size, num_points, k, channels)
    center = x_trans.view(batch_size, num_points, 1, channels).repeat(1, 1, k, 1)
    return neighbor, center


def get_original_edge_feature(x, idx):
    neighbor, center = gather_neighbors(x, idx)
    feature = torch.cat((neighbor - center, center), dim=3)
    return feature.permute(0, 3, 1, 2).contiguous()


def get_original_scorenet_input(x, idx):
    neighbor, center = gather_neighbors(x, idx)
    relative = neighbor - center
    dist = torch.linalg.vector_norm(relative, dim=3, keepdim=True)
    feature = torch.cat((relative, neighbor, center, dist), dim=3)
    return feature.permute(0, 3, 1, 2).contiguous()


def feat_trans_dgcnn(point_input, kernel, m):
    batch_size, _, num_points = point_input.size()
    point_output = torch.matmul(
        point_input.permute(0, 2, 1).repeat(1, 1, 2),
        kernel,
    ).view(batch_size, num_points, m, -1)
    center_output = torch.matmul(
        point_input.permute(0, 2, 1),
        kernel[:point_input.size(1)],
    ).view(batch_size, num_points, m, -1)
    return point_output, center_output


def assemble_dgcnn_pytorch(score, point_input, center_input, knn_idx, aggregate='sum'):
    # Slower fallback for cases where the PAConv CUDA extension is locked by
    # another Python process. The CUDA extension is still available by flag.
    batch_size, num_points, num_matrices, out_channels = point_input.shape
    k = knn_idx.size(-1)
    idx_base = torch.arange(batch_size, device=knn_idx.device).view(-1, 1, 1) * num_points
    idx_flat = (knn_idx + idx_base).reshape(-1)
    neighbor_point = point_input.reshape(batch_size * num_points, num_matrices, out_channels)[idx_flat]
    neighbor_point = neighbor_point.view(batch_size, num_points, k, num_matrices, out_channels)
    weighted = (neighbor_point - center_input.unsqueeze(2)) * score.unsqueeze(-1)
    if aggregate == 'sum':
        output = weighted.sum(dim=(2, 3))
    elif aggregate == 'avg':
        output = weighted.mean(dim=2).sum(dim=2)
    else:
        raise ValueError(f"Unsupported aggregate mode: {aggregate}")
    return output.permute(0, 2, 1).contiguous()


def get_assemble_dgcnn(use_cuda_extension):
    if not use_cuda_extension:
        print("[INFO] PAConv assemble op: PyTorch fallback")
        return assemble_dgcnn_pytorch
    from PAConv.cuda_lib.functional import assign_score_withk
    print("[INFO] PAConv assemble op: CUDA extension")
    return assign_score_withk


class OriginalPAConv(nn.Module):
    def __init__(self, args, landmark_num):
        super().__init__()
        self.k = args.k
        self.landmark_num = landmark_num
        self.calc_scores = args.calc_scores
        self.hidden = args.hidden
        self.m2, self.m3, self.m4, self.m5 = args.num_matrices
        self.assemble_dgcnn = args.assemble_dgcnn
        self.return_latent = getattr(args, "return_latent", False)
        self.in_channels = int(getattr(args, "in_channels", 3))
        edge_channels = self.in_channels * 2
        scorenet_channels = self.in_channels * 3 + 1

        self.scorenet2 = ScoreNet(scorenet_channels, self.m2, hidden_unit=self.hidden[0])
        self.scorenet3 = ScoreNet(scorenet_channels, self.m3, hidden_unit=self.hidden[1])
        self.scorenet4 = ScoreNet(scorenet_channels, self.m4, hidden_unit=self.hidden[2])
        self.scorenet5 = ScoreNet(scorenet_channels, self.m5, hidden_unit=self.hidden[3])

        tensor2 = nn.init.kaiming_normal_(torch.empty(self.m2, 64 * 2, 64), nonlinearity='relu')
        tensor3 = nn.init.kaiming_normal_(torch.empty(self.m3, 64 * 2, 64), nonlinearity='relu')
        tensor4 = nn.init.kaiming_normal_(torch.empty(self.m4, 64 * 2, 64), nonlinearity='relu')
        tensor5 = nn.init.kaiming_normal_(torch.empty(self.m5, 64 * 2, 64), nonlinearity='relu')

        self.matrice2 = nn.Parameter(tensor2.permute(1, 0, 2).contiguous().view(64 * 2, self.m2 * 64))
        self.matrice3 = nn.Parameter(tensor3.permute(1, 0, 2).contiguous().view(64 * 2, self.m3 * 64))
        self.matrice4 = nn.Parameter(tensor4.permute(1, 0, 2).contiguous().view(64 * 2, self.m4 * 64))
        self.matrice5 = nn.Parameter(tensor5.permute(1, 0, 2).contiguous().view(64 * 2, self.m5 * 64))

        self.bn2 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn3 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn4 = nn.BatchNorm1d(64, momentum=0.1)
        self.bn5 = nn.BatchNorm1d(64, momentum=0.1)
        self.bnt = nn.BatchNorm1d(1024, momentum=0.1)
        self.bn6 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn7 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn8 = nn.BatchNorm1d(128, momentum=0.1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(edge_channels, 64, kernel_size=1, bias=True),
            nn.BatchNorm2d(64, momentum=0.1),
        )
        self.convt = nn.Sequential(nn.Conv1d(64 * 5, 1024, kernel_size=1, bias=False), self.bnt)
        self.conv6 = nn.Sequential(nn.Conv1d(1024 + 64 * 5, 256, kernel_size=1, bias=False), self.bn6)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv7 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False), self.bn7)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False), self.bn8)
        self.conv9 = nn.Conv1d(128, landmark_num, kernel_size=1, bias=True)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        idx, _ = knn(x, k=self.k)
        score_input = get_original_scorenet_input(x, idx)

        edge = get_original_edge_feature(x, idx)
        x1 = F.relu(self.conv1(edge)).max(dim=-1, keepdim=False)[0]

        x2, center2 = feat_trans_dgcnn(point_input=x1, kernel=self.matrice2, m=self.m2)
        score2 = self.scorenet2(score_input, calc_scores=self.calc_scores, bias=0)
        x2 = F.relu(self.bn2(self.assemble_dgcnn(score=score2, point_input=x2, center_input=center2, knn_idx=idx, aggregate='sum')))

        x3, center3 = feat_trans_dgcnn(point_input=x2, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(score_input, calc_scores=self.calc_scores, bias=0)
        x3 = F.relu(self.bn3(self.assemble_dgcnn(score=score3, point_input=x3, center_input=center3, knn_idx=idx, aggregate='sum')))

        x4, center4 = feat_trans_dgcnn(point_input=x3, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(score_input, calc_scores=self.calc_scores, bias=0)
        x4 = F.relu(self.bn4(self.assemble_dgcnn(score=score4, point_input=x4, center_input=center4, knn_idx=idx, aggregate='sum')))

        x5, center5 = feat_trans_dgcnn(point_input=x4, kernel=self.matrice5, m=self.m5)
        score5 = self.scorenet5(score_input, calc_scores=self.calc_scores, bias=0)
        x5 = F.relu(self.bn5(self.assemble_dgcnn(score=score5, point_input=x5, center_input=center5, knn_idx=idx, aggregate='sum')))

        multi_scale = torch.cat((x1, x2, x3, x4, x5), dim=1)
        global_feature = F.relu(self.convt(multi_scale))
        global_feature = F.adaptive_max_pool1d(global_feature, 1).view(batch_size, -1)
        global_feature = global_feature.view(batch_size, 1024, 1).repeat(1, 1, num_points)
        x = torch.cat((multi_scale, global_feature), dim=1)
        latent = x
        x = F.relu(self.conv6(x))
        x = self.dp1(x)
        x = F.relu(self.conv7(x))
        x = self.dp2(x)
        x = F.relu(self.conv8(x))
        heatmap = self.conv9(x)
        if self.return_latent:
            return latent, heatmap
        return heatmap


def topk_coords(points, heatmaps, k):
    topk_vals, topk_idx = torch.topk(heatmaps, k, dim=2)
    idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    points_expanded = points.unsqueeze(1).expand(-1, heatmaps.size(1), -1, -1)
    topk_points = torch.gather(points_expanded, 2, idx_expanded)
    weights = F.softmax(topk_vals, dim=2)
    return torch.sum(topk_points * weights.unsqueeze(-1), dim=2)


def get_rigid(src, dst):
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    h_matrix = reduce(
        lambda s, p: s + np.outer(p[0], p[1]),
        zip(src - src_mean, dst - dst_mean),
        np.zeros((3, 3)),
    )
    u, _, v = np.linalg.svd(h_matrix)
    rotation = v.T.dot(u.T)
    translation = -rotation.dot(src_mean) + dst_mean
    return np.hstack((rotation, translation[:, np.newaxis]))


def landmark_regression_mds(shape, heatmap, regression_point_num, random_state=None, distance_weight="paper"):
    shape_np = shape.detach().cpu().numpy()
    heatmap_np = heatmap.detach().cpu().numpy()
    sort_idx = np.argsort(heatmap_np, 0)

    selected_shape = np.array([
        shape_np[sort_idx[-regression_point_num:, lm_idx]]
        for lm_idx in range(heatmap_np.shape[1])
    ])
    selected_heatmap = np.array([
        heatmap_np[sort_idx[-regression_point_num:, lm_idx], lm_idx]
        for lm_idx in range(heatmap_np.shape[1])
    ]).reshape(-1, regression_point_num, 1)

    selected_rep = np.expand_dims(selected_shape, axis=-1).repeat(regression_point_num, axis=-1)
    pairwise_delta = selected_rep.transpose(0, 1, 3, 2) - selected_rep.transpose(0, 3, 1, 2)
    distance_matrix = np.linalg.norm(pairwise_delta, axis=3)
    if distance_weight in ("paper", "symmetric"):
        heatmap_pair_weight = 0.5 * (
            selected_heatmap.repeat(regression_point_num, axis=-1)
            + selected_heatmap.transpose(0, 2, 1).repeat(regression_point_num, axis=1)
        )
        distance_matrix = heatmap_pair_weight * distance_matrix
    elif distance_weight != "none":
        raise ValueError(f"Unsupported MDS distance weight mode: {distance_weight}")

    if random_state is None:
        mds = MDS(n_components=2, dissimilarity='precomputed')
    else:
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=random_state)
    shape_mds = np.array([mds.fit_transform(distance_matrix[i]) for i in range(heatmap_np.shape[1])])
    shape_mds = np.concatenate(
        (shape_mds, np.zeros((heatmap_np.shape[1], regression_point_num, 1))),
        axis=2,
    )
    landmark_2d = (
        np.sum(selected_heatmap.repeat(3, axis=2) * shape_mds, axis=1)
        / selected_heatmap.sum(1)
    )

    neighbor_count = min(6, regression_point_num)
    neigh = NearestNeighbors(n_neighbors=neighbor_count)
    neighbor_indices = []
    for i in range(heatmap_np.shape[1]):
        neigh.fit(shape_mds[i])
        neighbor_indices.append(neigh.kneighbors(landmark_2d[i].reshape(1, -1))[1])
    neighbor_indices = np.array(neighbor_indices)

    shape_ext = np.array([
        shape_mds[i, neighbor_indices[i], :].reshape(-1, 3)
        - landmark_2d[i].reshape(1, -1).repeat(neighbor_count, axis=0)
        for i in range(heatmap_np.shape[1])
    ])
    shape_ext_target = np.array([
        selected_shape[i, neighbor_indices[i], :]
        for i in range(heatmap_np.shape[1])
    ]).reshape(-1, neighbor_count, 3)

    w1 = shape_ext - np.repeat(shape_ext.mean(1, keepdims=True), neighbor_count, axis=1)
    w2 = shape_ext_target - np.repeat(shape_ext_target.mean(1, keepdims=True), neighbor_count, axis=1)
    w1 = np.linalg.norm(w1.reshape(heatmap_np.shape[1], -1), axis=1).reshape(-1, 1, 1)
    w2 = np.linalg.norm(w2.reshape(heatmap_np.shape[1], -1), axis=1).reshape(-1, 1, 1)
    shape_ext = shape_ext * w2 / w1

    landmark_3d = np.array([
        get_rigid(shape_ext[i], shape_ext_target[i])[:, 3]
        for i in range(heatmap_np.shape[1])
    ])
    return torch.from_numpy(landmark_3d).unsqueeze(0).to(shape.device, dtype=shape.dtype)


def unique_run_dir(args, train_len):
    project_dir = os.path.join(args.output_root, args.exp_name)
    os.makedirs(project_dir, exist_ok=True)
    base = f"FPS{args.num_points}_sigma{args.sigma}_batch{args.batch_size}_train{train_len}"
    setting = f"{base}_{args.user_tag}" if args.user_tag else base
    run_id = 1
    while True:
        run_dir = os.path.join(project_dir, f"{setting}_{run_id}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            os.makedirs(os.path.join(run_dir, "models"))
            break
        run_id += 1
    return run_dir


def save_command(run_dir, file_name):
    path = os.path.join(run_dir, file_name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("[Working Directory]\n")
        handle.write(os.getcwd() + "\n\n")
        handle.write("[Command]\n")
        handle.write(subprocess.list2cmdline([os.sys.executable] + os.sys.argv) + "\n")
    print(f"[INFO] Command saved to: {path}")


def evaluate(model, loader, args, method, max_batches=0):
    model.eval()
    errors = []
    per_landmark = []
    random_state = None if args.mds_random_state < 0 else args.mds_random_state
    with torch.no_grad():
        for batch_idx, (point, landmark, _, _) in enumerate(tqdm(loader, desc=f"Eval {method}")):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            point = point.to(device)
            landmark = landmark.to(device)
            point_normal, landmark_normal, centroid, scale = normalize_points(point, landmark)
            point_input = point_normal.permute(0, 2, 1).contiguous()
            heatmap = model(point_input)
            points_norm = point_input.permute(0, 2, 1).contiguous()
            if method == "mds":
                pred_norm = torch.cat([
                    landmark_regression_mds(
                        points_norm[i],
                        heatmap[i].permute(1, 0).contiguous(),
                        args.regression_point_num,
                        random_state=random_state,
                        distance_weight=args.mds_distance_weight,
                    )
                    for i in range(points_norm.size(0))
                ], dim=0)
            else:
                pred_norm = topk_coords(points_norm, heatmap, args.regression_point_num)
            pred = pred_norm * scale + centroid
            dists = torch.linalg.vector_norm(pred - landmark, dim=2).detach().cpu().numpy()
            errors.extend(dists.reshape(-1).tolist())
            per_landmark.append(dists)

    per_landmark_np = np.concatenate(per_landmark, axis=0)
    errors_np = np.array(errors, dtype=np.float64)
    return {
        "average_me": float(errors_np.mean()),
        "std": float(errors_np.std()),
        "p95": float(np.percentile(errors_np, 95)),
        "per_landmark_me": per_landmark_np.mean(axis=0),
        "per_landmark_std": per_landmark_np.std(axis=0),
    }


def write_eval_summary(run_dir, summaries):
    path = os.path.join(run_dir, "original_paconv_eval_summary.txt")
    with open(path, "w", encoding="utf-8") as handle:
        for method, summary in summaries.items():
            handle.write(f"[{method.upper()}]\n")
            handle.write(
                f"Average ME: {summary['average_me']:.4f} ± {summary['std']:.4f} mm "
                f"(95%ile: {summary['p95']:.4f} mm)\n"
            )
            handle.write("Per-landmark ME ± STD\n")
            for idx, (me, std) in enumerate(
                zip(summary["per_landmark_me"], summary["per_landmark_std"]),
                start=1,
            ):
                handle.write(f"LM {idx:02d}: {me:.4f} ± {std:.4f} mm\n")
            handle.write("\n")
    print(f"[INFO] Evaluation summary saved to: {path}")


def train(args):
    set_seed(args.seed)
    args.hidden = parse_literal(args.hidden)
    args.num_matrices = parse_literal(args.num_matrices)
    args.assemble_dgcnn = get_assemble_dgcnn(args.use_cuda_extension)

    train_dataset = CurrentSplitHeatmapDataset(
        args.data_root, args.train_dataset_name, "train", shuffle_points=args.shuffle_train_points
    )
    val_dataset = CurrentSplitHeatmapDataset(
        args.data_root, args.val_dataset_name, args.val_partition, shuffle_points=False
    )
    test_dataset = CurrentSplitHeatmapDataset(
        args.data_root, args.test_dataset_name, args.eval_datatype, shuffle_points=False
    )

    run_dir = unique_run_dir(args, len(train_dataset))
    save_command(run_dir, "command_original_paconv_train_eval.txt")
    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Train/Val/Test sizes: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")
    print("[INFO] Original-like PAConv: XYZ input, 6ch edge conv, raw heatmap logits, heatmap-only loss")
    print(
        f"[INFO] Paper settings: k={args.k}, r={args.regression_point_num}, "
        f"val_method={args.val_method}, mds_distance_weight={args.mds_distance_weight}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    model = OriginalPAConv(args, args.landmark_num).to(device)
    model.apply(weight_init)
    criterion = OriginalAdaptiveWingLoss().to(device) if args.loss == "adaptive_wing" else nn.MSELoss().to(device)
    scale_and_translate = PointcloudScaleAndTranslate()

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(
            model.parameters(),
            lr=args.lr * 100,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)

    scheduler = (
        CosineAnnealingLR(opt, T_max=100, eta_min=0.0001)
        if args.scheduler == "cos"
        else StepLR(opt, step_size=40, gamma=0.9)
    )

    train_log_path = os.path.join(run_dir, "training_log.csv")
    with open(train_log_path, "w", encoding="utf-8") as handle:
        handle.write(f"epoch,train_heatmap_loss,val_{args.val_method}_me,val_{args.val_method}_std,lr\n")

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for point, _, heatmap, _ in tqdm(train_loader, desc=f"Train Ep {epoch + 1:03d}"):
            if args.max_train_batches > 0 and train_batches >= args.max_train_batches:
                break
            point = point.to(device)
            heatmap = torch.where(torch.isnan(heatmap), torch.full_like(heatmap, 0), heatmap).to(device)
            point_normal = normalize_points(point)
            point_normal = scale_and_translate(point_normal)
            point_input = point_normal.permute(0, 2, 1).contiguous()
            pred_heatmap = model(point_input)
            loss = criterion(pred_heatmap, heatmap.permute(0, 2, 1).contiguous())

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += float(loss.item())
            train_batches += 1

        if args.scheduler == "cos":
            scheduler.step()
        else:
            if opt.param_groups[0]["lr"] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]["lr"] < 1e-5:
                for param_group in opt.param_groups:
                    param_group["lr"] = 1e-5

        val_summary = evaluate(
            model,
            val_loader,
            args,
            method=args.val_method,
            max_batches=args.max_val_batches,
        )
        avg_train_loss = train_loss / max(train_batches, 1)
        lr = opt.param_groups[0]["lr"]
        print(
            f"[ORIG_PACONV Ep {epoch + 1:03d}/{args.epochs}] "
            f"HM_Loss: {avg_train_loss:.6f} | Val_{args.val_method}_ME: "
            f"{val_summary['average_me']:.4f} ± {val_summary['std']:.4f} | LR: {lr:.6f}"
        )

        with open(train_log_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch + 1},{avg_train_loss:.8f},{val_summary['average_me']:.8f},"
                f"{val_summary['std']:.8f},{lr:.8f}\n"
            )

        if val_summary["average_me"] < best_val:
            best_val = val_summary["average_me"]
            torch.save(model.state_dict(), os.path.join(run_dir, "models", "Original_PAConv_best.t7"))
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                model.state_dict(),
                os.path.join(run_dir, "models", f"Original_PAConv_epoch_{epoch + 1}.t7"),
            )

    last_path = os.path.join(run_dir, "models", "Original_PAConv_last.t7")
    torch.save(model.state_dict(), last_path)
    print(f"[INFO] Last checkpoint saved to: {last_path}")

    if args.skip_final_eval:
        print("[INFO] Final evaluation skipped by --skip_final_eval")
        return

    summaries = {
        "topk": evaluate(model, test_loader, args, method="topk", max_batches=args.max_eval_batches),
        "mds": evaluate(model, test_loader, args, method="mds", max_batches=args.max_eval_batches),
    }
    write_eval_summary(run_dir, summaries)
    print(
        "[FINAL] TopK Average ME: "
        f"{summaries['topk']['average_me']:.4f} ± {summaries['topk']['std']:.4f} mm"
    )
    print(
        "[FINAL] MDS Average ME: "
        f"{summaries['mds']['average_me']:.4f} ± {summaries['mds']['std']:.4f} mm"
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Original 3DFA-GCN PAConv on current Earlandmark splits")
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--output_root", type=str, default="../results")
    parser.add_argument("--exp_name", type=str, default="S2G_Original3DFAGCN_PAConv_CurrentSplit")
    parser.add_argument("--user_tag", type=str, default="orig3dfa_paconv_xyz_hmonly_mds_currsplit")
    parser.add_argument("--train_dataset_name", type=str, default="train")
    parser.add_argument("--val_dataset_name", type=str, default="valiation")
    parser.add_argument("--val_partition", type=str, default="val")
    parser.add_argument("--test_dataset_name", type=str, default="test")
    parser.add_argument("--eval_datatype", type=str, default="test")
    parser.add_argument("--num_points", type=int, default=8192)
    parser.add_argument("--sigma", type=float, default=2.5)
    parser.add_argument("--landmark_num", type=int, default=36)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--scheduler", type=str, default="step", choices=["cos", "step"])
    parser.add_argument("--loss", type=str, default="adaptive_wing", choices=["adaptive_wing", "mse"])
    parser.add_argument("--use_sgd", action="store_true")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--calc_scores", type=str, default="softmax")
    parser.add_argument("--hidden", type=str, default="[[16], [16], [16], [16]]")
    parser.add_argument("--num_matrices", type=str, default="[8, 8, 8, 8]")
    parser.add_argument("--regression_point_num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dataset_seed", type=int, default=1)
    parser.add_argument("--shuffle_train_points", action="store_true")
    parser.add_argument("--mds_random_state", type=int, default=-1)
    parser.add_argument("--mds_distance_weight", type=str, default="paper", choices=["paper", "symmetric", "none"])
    parser.add_argument("--val_method", type=str, default="mds", choices=["mds", "topk"])
    parser.add_argument("--use_cuda_extension", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--skip_final_eval", action="store_true")
    parser.add_argument("--save_interval", type=int, default=50)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
