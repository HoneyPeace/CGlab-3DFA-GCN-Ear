import numpy as np
import matplotlib.pyplot as plt
import os

# 파일 경로 설정
data_dir = r"C:\Users\CGLAB\Desktop\Ear 3DFA-GCN-master\Ear296_Korean-npy"
heatmap_path = os.path.join(data_dir, "Heat_data_sample.npy")
shape_path = os.path.join(data_dir, "shape_sample.npy")

# 파일 불러오기
heatmap_sample = np.load(heatmap_path, allow_pickle=True)
shape_sample = np.load(shape_path, allow_pickle=True)

points = shape_sample[200]          # (2048, 3)
heatmap = heatmap_sample[200]       # (2048, N_landmarks)

# 저장 폴더 생성
output_dir = os.path.join(data_dir, "landmark_heatmaps_topview")
os.makedirs(output_dir, exist_ok=True)

# 모든 랜드마크에 대해 시각화 및 저장
for landmark_id in range(heatmap.shape[1]):
    colors = heatmap[:, landmark_id]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap='jet', s=1)
    
    # 위에서 내려다보는 시점으로 설정
    #ax.view_init(elev=90, azim=-90)

    plt.colorbar(sc, label=f"Heatmap value for landmark {landmark_id}")
    plt.title(f"Top View Heatmap for Landmark {landmark_id}")
    plt.savefig(os.path.join(output_dir, f"landmark_{landmark_id:03d}.png"))
    plt.close()
