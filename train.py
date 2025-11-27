'''
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: train.py
@Time: 2021/12/02 09:59 AM
'''

import os
import math
import time
import numpy as np
from scipy import io
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from init import *
from My_args import *
from augmentations import *
from dataset import FaceLandmarkData
from loss import AdaptiveWingLoss
from util import main_sample
from PAConv_model import PAConv
from init import _init_

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
#from util import save_heatmap_3d

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 06.10 평화 ---> 외장 GPU 1번만 사용하도록 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#07-04  npy 테스트, 학습 데이터 분리 저장 ---> 07/05
def save_split_npy(dataset, prefix, dataset_name, num_points, sigma):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import os
    import numpy as np

    # 기본 위치 저장: ./Ear296_Korean-npy/
    base_dir = os.path.join(f"{dataset_name}-npy")
    os.makedirs(base_dir, exist_ok=True)

    shape_list, landmark_list, heatmap_list = [], [], []
    for i in range(len(dataset)):
        p, l, h = dataset[i]
        shape_list.append(p.numpy())
        landmark_list.append(l.numpy())
        heatmap_list.append(h.numpy())

    shape_arr = np.stack(shape_list)
    landmark_arr = np.stack(landmark_list)
    heatmap_arr = np.stack(heatmap_list)

    # 기본 저장
    np.save(os.path.join(base_dir, f"shape_{prefix}.npy"), shape_arr)
    np.save(os.path.join(base_dir, f"landmark_{prefix}.npy"), landmark_arr)
    np.save(os.path.join(base_dir, f"Heat_data_{prefix}.npy"), heatmap_arr)
    print(f" 기본 위치 저장 완료: {base_dir}")

    # 조건별 위치 저장
    cond_dir = os.path.join(base_dir, f"FPS{num_points}_sigma{sigma}")
    os.makedirs(cond_dir, exist_ok=True)

    np.save(os.path.join(cond_dir, f"shape_{prefix}.npy"), shape_arr)
    np.save(os.path.join(cond_dir, f"landmark_{prefix}.npy"), landmark_arr)
    np.save(os.path.join(cond_dir, f"Heat_data_{prefix}.npy"), heatmap_arr)
    print(f" 조건별 위치 저장 완료: {cond_dir}")

    # 🔍 GT 히트맵 시각화 저장 (train/test 모두 생성해서 저장)
    vis_folder_train = os.path.join(cond_dir, "GT_Trainheatmap")
    vis_folder_test = os.path.join(cond_dir, "GT_Testheatmap")
    os.makedirs(vis_folder_train, exist_ok=True)
    os.makedirs(vis_folder_test, exist_ok=True)

    for idx in range(len(shape_arr)):
        if idx % 100 != 0:
            continue

        points_np = shape_arr[idx]                  # (num_points, 3)
        heatmap_np = heatmap_arr[idx].T             # (landmarks, num_points)
        landmark_num = heatmap_np.shape[0]

        for landmark_id in range(min(40, landmark_num)):
            colors = heatmap_np[landmark_id]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=colors, cmap='jet', s=1)
            ax.view_init(elev=90, azim=-90)
            plt.colorbar(sc, label=f"GT Heatmap value for landmark {landmark_id}")
            plt.title(f"{prefix.capitalize()} Sample {idx:03d} - GT Heatmap (Landmark {landmark_id})")

            save_path = os.path.join(
                vis_folder_train if prefix.lower() == "train" else vis_folder_test,
                f"sample{idx:03d}_landmark{landmark_id}.png"
            )
            plt.savefig(save_path)
            plt.close()

    print(f" GT 히트맵 시각화 완료: {vis_folder_train if prefix.lower() == 'train' else vis_folder_test}")


def train(args):
    #writer = SummaryWriter('runs/3D_face_alignment')
    if args.need_resample:
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.dataset)

    # Dataset Random partition
    FaceLandmark = FaceLandmarkData(partition='trainval', data=args.dataset)
    train_size = int(len(FaceLandmark) * 0.7)
    test_size = len(FaceLandmark) - train_size
    torch.manual_seed(args.dataset_seed)
    # Prepare the dateset and dataloader 
    train_dataset, test_dataset = torch.utils.data.random_split(FaceLandmark, [train_size, test_size])
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=True, drop_last=True)
 
    # 07/05 추가: 분할된 데이터 저장
    save_split_npy(train_dataset, "train", args.dataset, args.num_points, args.sigma)
    save_split_npy(test_dataset, "test", args.dataset, args.num_points, args.sigma)

    # data argument
    ScaleAndTranslate = PointcloudScaleAndTranslate()
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5

    # select a model to train
    model = PAConv(args, args.landmark_num).to(device) #랜드마크 갯수 숫자 맞춤  # 68 in FaceScape; 8 in BU-3DFE and FRGC
    model.apply(weight_init)
    # model = nn.DataParallel(model) 06.10 평화 --> 외장지피유만 사용하도록

    print('let us use', torch.cuda.device_count(), 'GPUs')
    if args.loss == 'adaptive_wing':
        criterion = AdaptiveWingLoss()
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0.0001)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=40, gamma=0.9)

    loss_epoch = 0.0
    for epoch in range(args.epochs):
        iters = 0
        model.train()
        for point, landmark, seg in train_loader:
            seg = torch.where(torch.isnan(seg), torch.full_like(seg, 0), seg)
            iters = iters + 1
            if args.no_cuda == False:
                point = point.to(device)                   # point: (Batch * num_point * num_dim)
                landmark = landmark.to(device)             # landmark : (Batch * landmark * num_dim)
                seg = seg.to(device)                       # seg: (Batch * point_num * landmark)
            point_normal = normalize_data(point)           # point_normal : (Batch * num_point * num_dim)
            point_normal = ScaleAndTranslate(point_normal)
            opt.zero_grad()
            point_normal = point_normal.permute(0, 2, 1)   # point : (batch * num_dim * num_point)
            pred_heatmap = model(point_normal)

            # Compute the loss fucntion 
            loss = criterion(pred_heatmap, seg.permute(0, 2, 1).contiguous())
            loss.backward()
                     
            loss_epoch = loss_epoch + loss
            opt.step()
            print('Epoch: [%d / %d] Train_Iter: [%d /%d] loss: %.4f' % (epoch + 1, args.epochs, iters, len(train_loader), loss))

        #경로 따라가게 수정 ---> 07/05
        if (epoch + 1) % 5 == 0:
            model_dir = os.path.join(f"{args.dataset}-npy", f"FPS{args.num_points}_sigma{args.sigma}", "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.t7")
            torch.save(model.state_dict(), model_path)
#        if (epoch + 1) % 5 == 0:
#            torch.save(model.state_dict(), './checkpoints/%s/%s/models/model_epoch_%d.t7' % (args.exp_name, args.dataset, epoch+1))
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        #writer.add_scalar('3D_Face_Alignment_loss', loss_epoch / ((epoch + 1) * len(train_loader)), epoch + 1)


if __name__ == "__main__":
    # Training settings
    args = parser.parse_args()
    _init_(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    train(args)






