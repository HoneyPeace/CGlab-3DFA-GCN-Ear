'''
@Author: Yuan Wang (Modified by Researcher 2)
@File: init.py
@Description: Initialize checkpoint folders & Backup source code
'''

import os
import shutil
import torch

def _init_(args):
    # args.output_root: ../Result
    # args.exp_name: Ear_Project (예시)
    
    # 1. 체크포인트 폴더 생성
    ckpt_dir = os.path.join(args.output_root, 'Models', args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. 소스코드 백업 (나중에 코드를 어떻게 짰는지 확인용)
    # 현재 폴더(.)의 주요 파이썬 파일들을 Result/Code_Backup/실험명/ 에 복사
    code_backup_dir = os.path.join(args.output_root, 'Code_Backup', args.exp_name)
    os.makedirs(code_backup_dir, exist_ok=True)
    
    # 백업할 파일 목록
    src_files = ['train.py', 'dataset.py', 'PAConv_model.py', 'util.py', 'loss.py', 'My_args.py', 'augmentations.py', 'init.py']
    
    for f in src_files:
        if os.path.exists(f):
            shutil.copy(f, os.path.join(code_backup_dir, f))
            
    print(f">> Initialized Experiment: {args.exp_name}")
    print(f">> Source Code Backed up to: {code_backup_dir}")

# weight_init 함수는 수정할 필요 없음 (그대로 사용)
def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)