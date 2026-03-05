import os
import numpy as np
import torch
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pytorch_lightning as pl
from schedulefree import RAdamScheduleFree
from tqdm import tqdm

# ==========================================
# 設定箇所
# ==========================================
# NPZ_FILE_PATH = 'fused7111216_hdbscan-0.001-sample-5-MIN-20-step-under10.npz' 
NPZ_FILE_PATH = 'fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz' 
PROJECT_NAME = 'EnergyPredictor_kai_ultra_super_new_super_tani'
MANUAL_GROUP_PATH = None 

# ==========================================
# 1. モデル定義
# ==========================================
class EnergyPredictorPL(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-2, dropout: float = 0.5, energy_mean: float = 0.0, energy_std: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(31+768, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 1)
        )
        self.energy_mean = energy_mean
        self.energy_std = energy_std

    def forward(self, x):
        features, step = x
        em_features = torch.cat([features, step], dim=1)
        return self.model(em_features)

    def configure_optimizers(self):
        return RAdamScheduleFree(self.parameters(), lr=self.hparams.learning_rate)

# ==========================================
# 2. データロード関数
# ==========================================
def load_val_data(data, task_name, val_split=0.2):
    mask = data['task_name'] == task_name
    if np.sum(mask) == 0:
        return None, None, None, None
    
    robot_state = data['robot_state'][mask]
    predicted_pose = np.concatenate([
        data['soft_predicted_position'][mask],
        data['soft_predicted_rotation_quat'][mask]
    ], axis=1)
    grip = data['soft_predicted_gripper_state'][mask].reshape(-1, 1).astype(np.float32)
    coll = data['soft_predicted_collision_state'][mask].reshape(-1, 1).astype(np.float32)
    fused_feat = data['rvt_fused_feat'][mask]

    inputs = np.concatenate([robot_state, predicted_pose, grip, coll, fused_feat], axis=1)
    labels = data['energy'][mask].reshape(-1, 1)

    steps_raw = data['step'][mask].astype(np.float32)
    steps_norm = (steps_raw - 1) / (25 - 1)
    
    try:
        _, val_inputs, _, val_labels, _, val_steps_norm = train_test_split(
            inputs, labels, steps_norm, test_size=val_split, random_state=42, shuffle=True
        )
    except ValueError:
        return None, None, None, None
    
    val_steps_int = np.round(val_steps_norm * 24 + 1).astype(int)
    return val_inputs, val_labels, val_steps_norm, val_steps_int

# ==========================================
# 3. 可視化関数群
# ==========================================

def plot_boxplot(df, output_dir, task_name):
    """
    Boxplot (ステップごとの誤差分布)
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Step', y='Absolute Error', showfliers=False, color='skyblue')
    plt.title(f'Task: {task_name} - MAE Distribution per Step')
    plt.xlabel('Step Index (1-25)')
    plt.ylabel('Absolute Error (Unscaled)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{task_name}_1_boxplot.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()

def plot_step_wise_parity_with_mae(df, output_dir, task_name):
    """
    Parity Plot (正解 vs 予測) - タイトルにMAEとR2を表示
    """
    unique_steps = sorted(df['Step'].unique())
    n_steps = len(unique_steps)
    
    cols = 5
    rows = (n_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    global_min = min(df['True Value'].min(), df['Predicted Value'].min())
    global_max = max(df['True Value'].max(), df['Predicted Value'].max())
    margin = (global_max - global_min) * 0.05
    if margin == 0: margin = 1.0
    lims = [global_min - margin, global_max + margin]

    for i, step in enumerate(unique_steps):
        ax = axes[i]
        step_data = df[df['Step'] == step]
        
        if step_data.empty:
            continue

        y_true = step_data['True Value']
        y_pred = step_data['Predicted Value']
        
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0
        mae = np.mean(np.abs(y_true.values - y_pred.values))

        ax.scatter(y_true, y_pred, alpha=0.2, s=15, color='royalblue', edgecolor='none')
        ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=1.5, label='Ideal')
        
        ax.set_title(f'Step {step}\nMAE={mae:.4f} | R²={r2:.3f}', fontsize=10)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, alpha=0.3)
        
        if i % cols == 0:
            ax.set_ylabel('Predicted')
        if i >= (rows - 1) * cols:
            ax.set_xlabel('True')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Task: {task_name} - Step-wise Parity Plots', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    filename = f"{task_name}_2_parity.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=100)
    plt.close()

# ==========================================
# 4. メイン処理
# ==========================================
def process_and_evaluate_tasks(config_path, npz_path):
    print(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading data: {npz_path}")
    if not os.path.exists(npz_path):
        print(f"Error: NPZ file not found at {npz_path}")
        return
    raw_data = np.load(npz_path)

    task_configs = config['task_configs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Processing {len(task_configs)} tasks on {device}...")

    # --- ディレクトリ設定 ---
    # NPZファイル名(拡張子なし)を取得
    npz_filename = os.path.splitext(os.path.basename(npz_path))[0]
    
    # analysis_results/{NPZ_FILE_NAME}/
    base_output_dir = os.path.join('analysis_results', npz_filename)
    os.makedirs(base_output_dir, exist_ok=True)

    print(f"Results will be saved directly to: {base_output_dir}")

    task_performance_summary = []

    for task_name, params in tqdm(task_configs.items(), desc="Analyzing Tasks"):
        ckpt_path = params['ckpt_path']
        
        safe_task_name = str(task_name).replace('/', '_').replace('\\', '_')

        # パス補正
        if not os.path.exists(ckpt_path):
            base_dir = os.path.dirname(config_path)
            filename = os.path.basename(ckpt_path)
            potential_path = os.path.join(base_dir, filename)
            if os.path.exists(potential_path):
                ckpt_path = potential_path
            else:
                continue

        # 1. データロード
        val_inputs, val_labels_true, val_steps_norm, val_steps_int = load_val_data(raw_data, task_name)
        if val_inputs is None:
            continue

        val_inputs_t = torch.tensor(val_inputs, dtype=torch.float32).to(device)
        val_steps_norm_t = torch.tensor(val_steps_norm, dtype=torch.float32).unsqueeze(1).to(device)

        # 2. 推論
        try:
            model = EnergyPredictorPL.load_from_checkpoint(ckpt_path)
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                preds_norm = model((val_inputs_t, val_steps_norm_t))
                mu = params['normalization']['mean']
                sigma = params['normalization']['std']
                preds_real = preds_norm.cpu().numpy() * sigma + mu

            abs_errors = np.abs(preds_real - val_labels_true)
            
            df_task = pd.DataFrame({
                'Step': val_steps_int.flatten(),
                'Absolute Error': abs_errors.flatten(),
                'True Value': val_labels_true.flatten(),
                'Predicted Value': preds_real.flatten()
            })

            # 3. グラフ作成 (同じフォルダに直出し)
            plot_boxplot(df_task, base_output_dir, safe_task_name)
            plot_step_wise_parity_with_mae(df_task, base_output_dir, safe_task_name)

            # 4. ランキングデータ蓄積
            mean_mae = df_task['Absolute Error'].mean()
            r2_val = r2_score(df_task['True Value'], df_task['Predicted Value'])
            
            task_performance_summary.append({
                'Task': str(task_name),
                'MAE': mean_mae,
                'R2': r2_val
            })

        except Exception as e:
            print(f"Error processing task {task_name}: {e}")
            continue

    # 結果保存
    if task_performance_summary:
        summary_df = pd.DataFrame(task_performance_summary)
        summary_df = summary_df.sort_values('MAE', ascending=False)
        
        rank_path = os.path.join(base_output_dir, 'task_performance_ranking.csv')
        summary_df.to_csv(rank_path, index=False)
        
        print("\n" + "="*40)
        print("Analysis Complete!")
        print(f"Saved all files in: {base_output_dir}")
        print("="*40)
        print("Worst 5 Tasks (Highest MAE):")
        print(summary_df.head(5))
    else:
        print("No tasks were successfully processed.")

def main():
    config_path = None
    if MANUAL_GROUP_PATH:
        config_path = os.path.join(MANUAL_GROUP_PATH, 'energy_config.yaml')
    else:
        search_dir = os.path.join('checkpoints', PROJECT_NAME)
        if os.path.exists(search_dir):
            subdirs = [os.path.join(search_dir, d) for d in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, d))]
            if subdirs:
                latest_subdir = max(subdirs, key=os.path.getmtime)
                config_path = os.path.join(latest_subdir, 'energy_config.yaml')
                print(f"Auto-detected latest experiment: {latest_subdir}")

    if not config_path or not os.path.exists(config_path):
        print("Config file not found.")
        return

    process_and_evaluate_tasks(config_path, NPZ_FILE_PATH)

if __name__ == '__main__':
    main()