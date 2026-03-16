import optuna
import torch
from My_args import parser
from train import train

def objective(trial):
    args = parser.parse_args()
    
    # 1. 탐색할 하이퍼파라미터 범위 설정 (좁은 범위 권장)
    args.focal_gamma = trial.suggest_float("focal_gamma", 1.0, 2.5)
    args.focal_max = trial.suggest_float("focal_max", 2.0, 8.0)
    
    # 2. Optuna 탐색을 위한 전체 에포크 상한선 설정 (예: 100 에포크)
    args.epochs = 100 
    
    args.user_tag = f"Optuna_G{args.focal_gamma:.2f}_M{args.focal_max:.2f}"
    print(f"\n🚀 [Optuna Trial {trial.number}] Gamma: {args.focal_gamma:.2f}, Max: {args.focal_max:.2f}")
    
    # 3. train 함수에 trial 객체를 넘겨서 조기 종료(Pruning)가 가능하게 함
    best_val_mm = train(args, trial=trial)
    
    return best_val_mm

if __name__ == "__main__":
    # Optuna 스터디 생성 (Pruner 장착: 성적이 중간값 이하로 떨어지면 가차없이 자름)
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=20)
        # 해석: 처음 3개의 시도는 무조건 끝까지 지켜보고, 
        # 그 이후 시도부터는 최소 20 에포크(warmup)까지만 지켜본 뒤 가망 없으면 짤라버림!
    )
    
    study.optimize(objective, n_trials=20)
    
    print("\n==========================================")
    print("🏆 탐색 종료! 최고의 파라미터 조합:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value:.4f}")
    print(f"달성한 최저 오차: {study.best_value:.4f} mm")
    print("==========================================")