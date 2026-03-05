# visualize_results.py (タスクごと学習対応版)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split # <--- 変更点
import glob # <--- 追加

# --- 1. モデル構造の変更 ---
#      入力は (観測データ + ステップ数) = 28 + 1 = 29次元
class EnergyPredictorForVis(nn.Module):
    def __init__(self, input_dim=29, dropout=0.3): # <--- 変更点
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 入力は (features, step) のタプル
        features, step = x # <--- 変更点
        # 2つの特徴量を結合
        combined_features = torch.cat([features, step], dim=1) # <--- 変更点
        return self.model(combined_features)

# --- 2. 可視化のメイン処理 ---
def visualize():
    # --- 設定 ---
    NPZ_FILE_PATH = "low_200_energy.npz"
    CHECKPOINT_DIR = "./checkpoints/"
    VAL_SPLIT_RATIO = 0.2
    RANDOM_SEED = 42 # 学習時と同じシード
    EXPECTED_MAX_STEP = 25.0

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # --- 全データとタスク名の準備 ---
    if not os.path.exists(NPZ_FILE_PATH):
        print(f"エラー: データファイル '{NPZ_FILE_PATH}' が見つかりません。")
        return
    data = np.load(NPZ_FILE_PATH)
    all_inputs = np.concatenate([data['robot_state'], data['predicted_position'], data['predicted_rotation_quat']], axis=1)
    all_labels = data['energy'].reshape(-1, 1)
    all_task_names = data['task_name']
    all_steps = data['step'].astype(np.float32)
    all_steps = (all_steps - 1) / (EXPECTED_MAX_STEP - 1)
    all_steps = np.clip(all_steps, 0.0, 1.0)
    
    unique_tasks = np.unique(all_task_names)
    print(f"対象タスク: {unique_tasks}")

    # --- タスクごとのループ処理 ---
    for task_name in unique_tasks:
        print("\n" + "---" * 15)
        print(f"タスク '{task_name}' の可視化を開始します...")
        
        # --- 対応するモデルファイルを探す ---
        # 例: 'model-close_jar-best.ckpt' のようなファイル名を探す
        model_files = glob.glob(os.path.join(CHECKPOINT_DIR, f"*{task_name}*"))
        if not model_files:
            print(f"警告: タスク '{task_name}' のモデルファイルが見つかりません。スキップします。")
            continue
        MODEL_PATH = model_files[0] # 最初に見つかったファイルを使用
        print(f"使用モデル: {MODEL_PATH}")

        # --- モデルの読み込み ---
        model = EnergyPredictorForVis().to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['state_dict']
        model_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        model.model.load_state_dict(model_state_dict)
        model.eval()

        # --- 現在のタスクのデータのみを抽出 ---
        mask = (all_task_names == task_name)
        task_inputs = all_inputs[mask]
        task_labels = all_labels[mask]
        task_steps = all_steps[mask]

        # --- 学習時と全く同じ方法で検証データを抽出 ---
        _, X_val_features, _, y_val, _, X_val_steps = train_test_split(
            task_inputs,
            task_labels,
            task_steps,
            test_size=VAL_SPLIT_RATIO,
            random_state=RANDOM_SEED,
            shuffle=True
        )

        # --- 予測の実行 ---
        features_tensor = torch.tensor(X_val_features, dtype=torch.float32).to(device)
        steps_tensor = torch.tensor(X_val_steps, dtype=torch.float32).unsqueeze(1).to(device)
        
        with torch.no_grad():
            # 2つの入力をタプルで渡す
            y_pred_tensor = model((features_tensor, steps_tensor))
        
        y_pred = y_pred_tensor.cpu().numpy()
        y_true = y_val
        errors = y_pred - y_true
        
        actual_mse = np.mean(errors ** 2)
        actual_rmse = np.sqrt(actual_mse)
        
        print(f"タスク '{task_name}' の検証データでのRMSE: {actual_rmse:.2f} (J)")

        # --- 可視化 ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Visualization for Task: {task_name}", fontsize=16) # <--- 変更点
        
        # 1. 実績値 vs 予測値プロット
        axes[0].scatter(y_true, y_pred, alpha=0.5, label="Predictions")
        perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
        axes[0].plot(perfect_line, perfect_line, 'r--', label="Ideal Line (y=x)")
        axes[0].set_title("Actual vs. Predicted Plot")
        axes[0].set_xlabel("Actual Energy (J)")
        axes[0].set_ylabel("Predicted Energy (J)")
        axes[0].grid(True)
        axes[0].legend()
        axes[0].axis('equal')

        # 2. 誤差の分布（ヒストグラム）
        axes[1].hist(errors, bins=50)
        axes[1].set_title("Error Distribution (Histogram)")
        axes[1].set_xlabel("Error (Predicted - Actual)")
        axes[1].set_ylabel("Count")
        axes[1].grid(True)

        # 3. 残差プロット
        axes[2].scatter(y_true, errors, alpha=0.5)
        axes[2].axhline(y=0, color='r', linestyle='--')
        axes[2].set_title("Residual Plot")
        axes[2].set_xlabel("Actual Energy (J)")
        axes[2].set_ylabel("Error (Predicted - Actual)")
        axes[2].grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    visualize()