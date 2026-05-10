# @File: loss_controller.py
# @Description: 모델 분기, 가중치 스케줄링(Patience), 모델 맞춤형 로깅 전담
# 🌟 [수정 완료] train.py가 전달하는 decay_step, L_pa 파라미터 수신부 완벽 동기화

import torch
import pandas as pd

class DeepPALossController:
    def __init__(self, patience=5, base_hds=0.1, decay_step=0.05, min_heatmap_warmup=30, use_rlw_for_pred=False, aux_drop_epochs=30, frozen_paconv_hm_weight=0.0): # 🌟 decay_step 수신부 추가
        self.patience = patience
        self.base_hds = base_hds     # Aux(HDS) 기본 가중치
        self.decay_step = decay_step # 가중치 감소 보폭
        self.min_heatmap_warmup = min_heatmap_warmup
        self.use_rlw_for_pred = use_rlw_for_pred
        if aux_drop_epochs <= 0:
            raise ValueError("aux_drop_epochs must be positive")
        self.aux_drop_epochs = aux_drop_epochs
        self.frozen_paconv_hm_weight = frozen_paconv_hm_weight
        self.val_decay = 1.0         # Main HM 가중치. 1.0으로 시작
        self.stagnation_counter = 0
        self.best_val_mm = float('inf')
        self.train_hm_history = []
        self.train_hm_plateau_counter = 0
        self.train_hm_plateau_best = float('inf')
        self.train_hm_plateau_first_epoch = None
        self.train_hm_plateau_trigger_epoch = None
        self.train_hm_plateau_ramp_start_epoch = None
        
        # E2E 전용 가중치 추가
        self.weight_PA = 1.0
        self.weight_DP = 0.0

    def update_patience(self, current_val_mm, epoch=None):
        """Val 정체 여부를 확인하고 가중치를 업데이트"""
        if getattr(self, 'loss_schedule', 'val_adaptive') in ['fixed_three_phase', 'linear_three_phase', 'train_hm_plateau']:
            if current_val_mm < self.best_val_mm:
                self.best_val_mm = current_val_mm
            self.stagnation_counter = 0
            return False

        if epoch is not None and epoch < self.min_heatmap_warmup:
            if current_val_mm < self.best_val_mm:
                self.best_val_mm = current_val_mm
            self.stagnation_counter = 0
            return False

        if current_val_mm < self.best_val_mm:
            self.best_val_mm = current_val_mm
            self.stagnation_counter = 0
            return False
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.patience:
                # 🌟 하드코딩(0.05) 대신 전달받은 decay_step 사용
                self.val_decay = max(0.1, self.val_decay - self.decay_step)
                
                # E2E 모드 가중치 전환 로직 포함
                self.weight_PA = max(0.1, self.weight_PA - self.decay_step)
                self.weight_DP = 1.0 - self.weight_PA
                
                self.stagnation_counter = 0
                return True 
        return False

    # 🌟 L_pa 수신부 추가 (단일 모델일 땐 None이 들어오도록 안전장치)
    def update_train_hm_plateau(self, train_main_hm, epoch):
        """Train Main_HM rolling loss plateau 기준으로 geometry transition 시작 시점을 정한다."""
        if getattr(self, 'loss_schedule', 'val_adaptive') != 'train_hm_plateau':
            return None
        if self.train_hm_plateau_trigger_epoch is not None:
            return None

        current_epoch = epoch + 1
        self.train_hm_history.append(float(train_main_hm))

        start_epoch = getattr(self, 'plateau_start_epoch', self.aux_drop_epochs)
        window = max(1, int(getattr(self, 'plateau_window', 20)))
        patience = max(1, int(getattr(self, 'plateau_patience', 10)))
        threshold = float(getattr(self, 'plateau_threshold', 0.01))

        if len(self.train_hm_history) < window or current_epoch <= start_epoch:
            return None

        rolling_hm = sum(self.train_hm_history[-window:]) / float(window)
        improved = (
            self.train_hm_plateau_best == float('inf')
            or rolling_hm < self.train_hm_plateau_best * (1.0 - threshold)
        )
        if improved:
            self.train_hm_plateau_best = rolling_hm
            self.train_hm_plateau_counter = 0
            self.train_hm_plateau_first_epoch = None
            return None

        self.train_hm_plateau_counter += 1
        if self.train_hm_plateau_first_epoch is None:
            self.train_hm_plateau_first_epoch = current_epoch

        if self.train_hm_plateau_counter >= patience:
            self.train_hm_plateau_trigger_epoch = current_epoch
            self.train_hm_plateau_ramp_start_epoch = epoch + 1
            return (
                f"[Train-HM Plateau Trigger] window={window}, patience={patience}, "
                f"threshold={threshold:.4f}, first_plateau_ep={self.train_hm_plateau_first_epoch}, "
                f"confirm_ep={current_epoch}, rolling_main_hm={rolling_hm:.6f}"
            )
        return None

    def compute_loss(self, model_name, epoch, L_main, L_aux, L_crd, L_srf, L_str, L_pa=None):
        """모델별 분기에 맞추어 최종 로스와 동적 가중치를 계산"""
        m_name = model_name.lower()
        if L_pa is None: L_pa = torch.tensor(0.0).to(L_crd.device)
        
        warmup_modes = [
            'single_deeppa', 'deeppa_frozen', 'frozen_aux_fixed', 'frozen_aux_drop',
            'frozen_no_aux', 'deepla_progress', 'deeppa_frozen_no_heat',
            'deepla_ori', 'deepla_decay', 'deepla_all', 'deepla_all_tied',
            'paconv_struct'
        ]
        loss_schedule = getattr(self, 'loss_schedule', 'val_adaptive')
        schedule_target_models = ['deeppa_finetune', 'deeppa_e2e']
        if loss_schedule in ['fixed_three_phase', 'linear_three_phase', 'train_hm_plateau'] and (m_name.startswith('frozen_') or m_name in schedule_target_models):
            fixed_heatmap_epochs = getattr(self, 'fixed_heatmap_epochs', 60)
            heatmap_only_active = epoch < fixed_heatmap_epochs
            final_main_weight = getattr(self, 'fixed_main_weight', 0.9)
            final_geom_weight = getattr(self, 'fixed_geom_weight', 0.1)
            if loss_schedule == 'train_hm_plateau':
                ramp_start_epoch = getattr(self, 'train_hm_plateau_ramp_start_epoch', None)
                if ramp_start_epoch is None or epoch < ramp_start_epoch:
                    w_main = 1.0
                    w_geom = 0.0
                else:
                    transition_epochs = max(1, int(getattr(self, 'plateau_transition_epochs', 30)))
                    transition_progress = min(1.0, max(0.0, (epoch - ramp_start_epoch + 1) / float(transition_epochs)))
                    w_main = 1.0 + (final_main_weight - 1.0) * transition_progress
                    w_geom = final_geom_weight * transition_progress
            elif heatmap_only_active:
                w_main = 1.0
                w_geom = 0.0
            elif loss_schedule == 'linear_three_phase':
                total_epochs = getattr(self, 'total_epochs', 500)
                transition_epochs = max(1, total_epochs - fixed_heatmap_epochs)
                transition_progress = min(1.0, max(0.0, (epoch - fixed_heatmap_epochs + 1) / float(transition_epochs)))
                w_main = 1.0 + (final_main_weight - 1.0) * transition_progress
                w_geom = final_geom_weight * transition_progress
            else:
                w_main = final_main_weight
                w_geom = final_geom_weight
        else:
            warmup_active = (m_name in warmup_modes) and (epoch < self.min_heatmap_warmup)
            w_main = 1.0 if warmup_active else max(0.1, self.val_decay)
            w_geom = 0.0 if warmup_active else 1.0 - w_main

        if self.use_rlw_for_pred:
            rand_weights = torch.rand(3, device=L_crd.device)
            rand_weights = rand_weights / rand_weights.sum().clamp_min(1e-8)
            L_pred = rand_weights[0] * L_crd + rand_weights[1] * L_srf + rand_weights[2] * L_str
        else:
            L_pred = L_crd + L_srf + L_str 
        
        # 기본값 세팅
        w_hds = self.base_hds
        w_pa = self.weight_PA
        w_dp = self.weight_DP
        frozen_pa_loss = self.frozen_paconv_hm_weight * L_pa if m_name.startswith('frozen_') else 0.0

        # =================================================================
        # 🌟 PAConv 및 기타 브랜치 완벽 대응
        # =================================================================
        if m_name in ['paconv_heat', 'single_paconv_heat']:
            total_loss = L_pa
            w_geom, w_main, w_hds = 0.0, 0.0, 0.0

        elif m_name == 'paconv_struct':
            total_loss = (w_main * L_pa) + (w_geom * L_pred)
            w_hds = 0.0
            
        elif m_name in ['paconv', 'single_paconv']:
            rho = 0.9
            total_loss = ((1.0 - rho) * L_pa) + (w_geom * L_pred)
            w_main, w_hds = 0.0, 0.0

        elif m_name == 'deeppa_finetune':
            finetune_aux_mode = getattr(self, 'finetune_aux_mode', 'fixed')
            if finetune_aux_mode == 'drop':
                w_hds = max(0.0, 1.0 - (1.0 * (epoch / float(self.aux_drop_epochs))))
            elif finetune_aux_mode == 'none':
                w_hds = 0.0
            w_pa = 0.1
            w_dp = 1.0
            total_loss = (w_pa * L_pa) + (w_main * L_main) + (w_hds * L_aux) + (w_geom * L_pred)
            
        elif m_name == 'deeppa_e2e':
            e2e_aux_mode = getattr(self, 'e2e_aux_mode', 'fixed')
            if e2e_aux_mode == 'drop':
                w_hds = max(0.0, 1.0 - (1.0 * (epoch / float(self.aux_drop_epochs))))
            elif e2e_aux_mode == 'none':
                w_hds = 0.0

            if getattr(self, 'e2e_staged_paconv', False):
                warmup_epochs = max(0, int(getattr(self, 'e2e_paconv_warmup_epochs', 30)))
                if epoch < warmup_epochs:
                    w_pa = 1.0
                    w_dp = 1.0
                    w_main = 0.0
                    w_geom = 0.0
                    w_hds = 0.0
                else:
                    w_pa = float(getattr(self, 'e2e_paconv_final_weight', 0.1))
                    w_dp = 1.0
                total_loss = (w_pa * L_pa) + (w_main * L_main) + (w_hds * L_aux) + (w_geom * L_pred)
            elif loss_schedule in ['fixed_three_phase', 'linear_three_phase', 'train_hm_plateau']:
                w_pa = w_main
                w_dp = 1.0
                total_loss = (w_pa * L_pa) + (w_main * L_main) + (w_hds * L_aux) + (w_geom * L_pred)
            else:
                rho = 0.9
                w_main = 1.0 - rho * w_geom
                total_loss = w_main * (w_pa * L_pa + w_dp * L_main) + (w_hds * L_aux) + (w_geom * L_pred)

        # =================================================================
        # 🌟 DeepPA & DeepLA Ablation 그룹
        # =================================================================
        elif m_name in ['single_deeppa', 'deeppa_frozen', 'frozen_aux_fixed']:
            total_loss = (w_main * L_main) + (w_hds * L_aux) + (w_geom * L_pred) + frozen_pa_loss

        elif m_name == 'frozen_aux_drop':
            w_hds = max(0.0, 1.0 - (1.0 * (epoch / float(self.aux_drop_epochs)))) 
            total_loss = (w_main * L_main) + (w_hds * L_aux) + (w_geom * L_pred) + frozen_pa_loss

        elif m_name in ['frozen_no_aux', 'deepla_progress', 'deeppa_frozen_no_heat']:
            w_hds = 0.0 
            if m_name == 'deeppa_frozen_no_heat':
                total_loss = L_pred
                w_main = 0.0
            else:
                total_loss = (w_main * L_main) + (w_geom * L_pred) + frozen_pa_loss

        elif m_name == 'deepla_ori':
            w_hds = self.val_decay 
            total_loss = (w_hds * L_aux) + (w_geom * L_pred)

        elif m_name in ['deepla_decay', 'deepla_all']: 
            w_hds = self.base_hds * (1.0 / (epoch + 1.0)) 
            total_loss = (w_hds * L_aux) + (w_main * L_main) + (w_geom * L_pred)

        elif m_name == 'deepla_all_tied':
            w_hds = self.base_hds * self.val_decay
            total_loss = (w_hds * L_aux) + (w_main * L_main) + (w_geom * L_pred)

        else: 
            total_loss = (w_main * L_main) + (w_hds * L_aux) + (w_geom * L_pred)

        weights = {'w_main': w_main, 'w_hds': w_hds, 'w_geom': w_geom, 'w_pa': w_pa, 'w_dp': w_dp, 'w_frozen_pa': self.frozen_paconv_hm_weight}
        return total_loss, weights

    # 🌟 t_hm_PA 수신부 추가
    def print_and_get_log(self, model_name, epoch, t_loss_n, t_mm, v_mm, weights, num_b, t_hm_main, t_hm_aux, t_crd, t_srf, t_str, t_hm_PA=0.0):
        """모델별로 불필요한 값은 가리고, Str(관계로스)는 포함시켜 터미널 출력 및 엑셀 데이터 반환"""
        m_name = model_name.lower()
        
        # 모델에 따른 텍스트 포맷 동적 생성
        if m_name in ['frozen_no_aux', 'deepla_progress']:
            w_str = f"Main_W: {weights['w_main']:.2f} | Geom_W: {weights['w_geom']:.2f}"
            l_str = f"Main_HM: {t_hm_main/num_b:.4f} | Crd: {t_crd/num_b:.4f} | Srf: {t_srf/num_b:.4f} | Str: {t_str/num_b:.4f}"
            
        elif m_name == 'deeppa_e2e' or m_name == 'deeppa_finetune':
            w_str = f"PA_W: {weights['w_pa']:.2f} | DP_Main_W: {weights['w_dp'] * weights['w_main']:.2f} | Aux_W: {weights['w_hds']:.2f} | Geom_W: {weights['w_geom']:.2f}"
            l_str = f"PA_HM: {t_hm_PA/num_b:.4f} | Main_HM: {t_hm_main/num_b:.4f} | Aux_HM: {t_hm_aux/num_b:.4f} | Geom: {(t_crd+t_srf+t_str)/num_b:.4f}"
            
        elif m_name == 'paconv_struct':
            w_str = f"PA_W: {weights['w_main']:.2f} | Geom_W: {weights['w_geom']:.2f}"
            l_str = f"PA_HM: {t_hm_PA/num_b:.4f} | Crd: {t_crd/num_b:.4f} | Srf: {t_srf/num_b:.4f} | Str: {t_str/num_b:.4f}"

        elif m_name in ['paconv_heat', 'single_paconv_heat', 'paconv', 'single_paconv']:
            w_str = f"PA_W: 1.0 | Geom_W: {weights['w_geom']:.2f}"
            l_str = f"PA_HM: {t_hm_PA/num_b:.4f} | Crd: {t_crd/num_b:.4f} | Srf: {t_srf/num_b:.4f} | Str: {t_str/num_b:.4f}"
            
        else:
            w_str = f"Main_W: {weights['w_main']:.2f} | Aux_W: {weights['w_hds']:.3f} | Geom_W: {weights['w_geom']:.2f}"
            l_str = f"Main_HM: {t_hm_main/num_b:.4f} | Aux_HM: {t_hm_aux/num_b:.4f} | Crd: {t_crd/num_b:.4f} | Srf: {t_srf/num_b:.4f} | Str: {t_str/num_b:.4f}"

        if m_name.startswith('frozen_') and weights.get('w_frozen_pa', 0.0) > 0.0:
            pa_hm_raw = t_hm_PA / num_b
            pa_hm_weighted = weights['w_frozen_pa'] * pa_hm_raw
            w_str = f"{w_str} | PA_W: {weights['w_frozen_pa']:.3f}"
            l_str = f"PA_HM: {pa_hm_raw:.4f} | PA_HM*w: {pa_hm_weighted:.4f} | {l_str}"

        # 터미널 프린팅
        print(f" [{model_name.upper()} Ep {epoch+1:03d}] Total_L: {t_loss_n/num_b:.4f} | Train_mm: {t_mm:.2f} || Val_mm: {v_mm:.2f} ")
        print(f"  ├─ ⚙️ {w_str}")
        print(f"  └─ 🎯 [Loss] {l_str}")

        # 엑셀 데이터 반환
        log_record = {
            'Epoch': epoch + 1, 'Total_Loss': t_loss_n/num_b, 'Train_mm': t_mm, 'Val_mm': v_mm,
            'W_Main': weights.get('w_main', 0), 'W_Geom': weights.get('w_geom', 0), 'W_Aux': weights.get('w_hds', 0),
            'L_main_hm': t_hm_main/num_b, 'L_aux_hm': t_hm_aux/num_b, 
            'L_coord': t_crd/num_b, 'L_surface': t_srf/num_b, 'L_struct': t_str/num_b,
            'L_pa_hm': t_hm_PA/num_b,
            'W_Frozen_PA': weights.get('w_frozen_pa', 0),
            'L_pa_hm_weighted': weights.get('w_frozen_pa', 0) * (t_hm_PA/num_b)
        }
        return log_record
