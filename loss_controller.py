# @File: loss_controller.py
# @Description: 모델 분기, 가중치 스케줄링(Patience), 모델 맞춤형 로깅 전담

import torch
import pandas as pd

class DeepPALossController:
    def __init__(self, patience=5, base_hds=0.1):
        self.patience = patience
        self.base_hds = base_hds     # Aux(HDS) 기본 가중치 0.1 고정
        self.val_decay = 1.0         # Main HM 가중치. 1.0으로 시작 (Geom은 0.0)
        self.stagnation_counter = 0
        self.best_val_mm = float('inf')

    def update_patience(self, current_val_mm):
        """Val 정체 여부를 확인하고 가중치를 업데이트"""
        if current_val_mm < self.best_val_mm:
            self.best_val_mm = current_val_mm
            self.stagnation_counter = 0
            return False
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.patience:
                self.val_decay = max(0.1, self.val_decay - 0.05)
                self.stagnation_counter = 0
                return True 
        return False

    def compute_loss(self, model_name, epoch, L_main, L_aux, L_crd, L_srf, L_str):
        """모델별 분기에 맞추어 최종 로스와 동적 가중치를 계산"""
        m_name = model_name.lower()
        
        w_main = max(0.1, self.val_decay)
        w_geom = 1.0 - w_main  
        L_pred = L_crd + L_srf + L_str 

        # 🌟 1. 기존 공통 로직
        if m_name in ['single_deeppa', 'deeppa_frozen', 'frozen_aux_fixed']:
            w_hds = self.base_hds # 0.1 고정
            total_loss = (w_main * L_main) + (w_hds * L_aux) + (w_geom * L_pred)

        # 🌟 2. [수정됨] frozen_aux_drop: 1.0에서 시작하여 15에폭 동안 감소
        elif m_name == 'frozen_aux_drop':
            w_hds = max(0.0, 1.0 - (1.0 * (epoch / 15.0))) 
            total_loss = (w_main * L_main) + (w_hds * L_aux) + (w_geom * L_pred)

        # 🌟 3. Aux 배제 그룹 (deepla_progress 포함)
        elif m_name in ['frozen_no_aux', 'deepla_progress']:
            w_hds = 0.0 # Aux 끔
            total_loss = (w_main * L_main) + (w_geom * L_pred)

        elif m_name == 'deepla_ori':
            w_hds = self.val_decay 
            total_loss = (w_hds * L_aux) + (w_geom * L_pred)

        # 🌟 4. [명칭 변경] deepla_all -> deepla_decay
        elif m_name == 'deepla_decay': 
            w_hds = self.base_hds * (1.0 / (epoch + 1.0)) # 반비례 감소
            total_loss = (w_hds * L_aux) + (w_main * L_main) + (w_geom * L_pred)

        elif m_name == 'deepla_all_tied':
            w_hds = self.base_hds * self.val_decay
            total_loss = (w_hds * L_aux) + (w_main * L_main) + (w_geom * L_pred)

        else: # 기본값 방어
            w_hds = self.base_hds
            total_loss = (w_main * L_main) + (w_hds * L_aux) + (w_geom * L_pred)

        weights = {'w_main': w_main, 'w_hds': w_hds, 'w_geom': w_geom}
        return total_loss, weights

    def print_and_get_log(self, model_name, epoch, t_loss_n, t_mm, v_mm, weights, num_b, t_hm_main, t_hm_aux, t_crd, t_srf, t_str):
        """모델별로 불필요한 값은 가리고, Str(관계로스)는 포함시켜 터미널 출력 및 엑셀 데이터 반환"""
        m_name = model_name.lower()
        
        # 모델에 따른 텍스트 포맷 동적 생성
        if m_name in ['frozen_no_aux', 'deepla_progress']:
            w_str = f"Main_W: {weights['w_main']:.2f} | Geom_W: {weights['w_geom']:.2f}"
            l_str = f"Main_HM: {t_hm_main/num_b:.4f} | Crd: {t_crd/num_b:.4f} | Srf: {t_srf/num_b:.4f} | Str: {t_str/num_b:.4f}"
        else:
            w_str = f"Main_W: {weights['w_main']:.2f} | Aux_W: {weights['w_hds']:.3f} | Geom_W: {weights['w_geom']:.2f}"
            l_str = f"Main_HM: {t_hm_main/num_b:.4f} | Aux_HM: {t_hm_aux/num_b:.4f} | Crd: {t_crd/num_b:.4f} | Srf: {t_srf/num_b:.4f} | Str: {t_str/num_b:.4f}"

        # 터미널 프린팅
        print(f" [{model_name.upper()} Ep {epoch+1:03d}] Total_L: {t_loss_n/num_b:.4f} | Train_mm: {t_mm:.2f} || Val_mm: {v_mm:.2f} ")
        print(f"  ├─ ⚙️ {w_str}")
        print(f"  └─ 🎯 [Loss] {l_str}")

        # 엑셀 데이터 반환
        log_record = {
            'Epoch': epoch + 1, 'Total_Loss': t_loss_n/num_b, 'Train_mm': t_mm, 'Val_mm': v_mm,
            'W_Main': weights.get('w_main', 0), 'W_Geom': weights.get('w_geom', 0), 'W_Aux': weights.get('w_hds', 0),
            'L_main_hm': t_hm_main/num_b, 'L_aux_hm': t_hm_aux/num_b, 
            'L_coord': t_crd/num_b, 'L_surface': t_srf/num_b, 'L_struct': t_str/num_b
        }
        return log_record