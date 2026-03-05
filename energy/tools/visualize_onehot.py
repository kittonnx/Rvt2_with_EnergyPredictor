# visualize_results.py (Step3: One-hot対応版)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F

# --- 1. 学習させたモデルと同じ構造のモデルを定義 ---
#      入力は (観測データ + OneHotタスクID + ステップ数) = 28 + 18 + 1 = 47次元
class EnergyPredictorForVis(nn.Module):
    def __init__(self, input_dim=47, dropout=0.3):
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
        # 入力は (features, task_one_hot, step) のタプル
        features, task_one_hot, step = x
        # 3つの特徴量をすべて結合
        combined_features = torch.cat([features, step, task_one_hot], dim=1)
        return self.model(combined_features)

# --- 2. 可視化のメイン処理 ---
def visualize():
    # --- 設定 ---
    NPZ_FILE_PATH = "low_200_energy.npz"
    # ★★★ .ckptファイル名を、あなたの学習済みモデルのパスに書き換えてください ★★★
    MODEL_PATH = "./checkpoints/model-LEARNING_RATE=0.00000-epoch=162-val_loss=47.2.ckpt" 
    VAL_SPLIT_RATIO = 0.2
    NUM_TASKS = 18
    EXPECTED_MAX_STEP = 25.0

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # --- モデルの読み込み ---
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。")
        return
    
    # 正しい構造でモデルの「器」を準備
    model = EnergyPredictorForVis().to(device)

    print(f"チェックポイントファイル '{MODEL_PATH}' を読み込んでいます...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['state_dict']
    
    # Lightningのチェックポイントから、モデル部分('model.')の重みだけを抽出
    model_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
    
    model.model.load_state_dict(model_state_dict)
    model.eval()

    # --- データの準備 ---
    if not os.path.exists(NPZ_FILE_PATH):
        print(f"エラー: データファイル '{NPZ_FILE_PATH}' が見つかりません。")
        return
        
    data = np.load(NPZ_FILE_PATH)
    # 観測データ
    inputs = np.concatenate([data['robot_state'], data['predicted_position'], data['predicted_rotation_quat']], axis=1)
    # ラベル
    labels = data['energy'].reshape(-1, 1)
    # タスクID
    task_names = data['task_name']
    unique_tasks = np.unique(task_names)
    task_to_id = {name: i for i, name in enumerate(unique_tasks)}
    task_ids = np.array([task_to_id[name] for name in task_names])
    # ステップ数
    steps = data['step'].astype(np.float32)
    steps = (steps - 1) / (EXPECTED_MAX_STEP - 1)
    steps = np.clip(steps, 0.0, 1.0)

    # 学習時と全く同じ方法で検証データを抽出
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT_RATIO, random_state=42)
    for _, val_idx in sss.split(np.zeros(len(task_ids)), task_ids):
        pass
    
    X_val_features = inputs[val_idx]
    X_val_task_ids = task_ids[val_idx]
    X_val_steps = steps[val_idx]
    y_val = labels[val_idx]

    # --- 予測の実行 ---
    # データをTensorに変換
    features_tensor = torch.tensor(X_val_features, dtype=torch.float32).to(device)
    task_ids_tensor = torch.tensor(X_val_task_ids, dtype=torch.long).to(device)
    steps_tensor = torch.tensor(X_val_steps, dtype=torch.float32).unsqueeze(1).to(device)
    
    # task_idをOne-hotベクトルに変換
    task_one_hot_tensor = F.one_hot(task_ids_tensor, num_classes=NUM_TASKS).float()

    with torch.no_grad():
        # 3つの入力をタプルで渡す
        # y_pred_tensor = model((features_tensor, task_one_hot_tensor, steps_tensor))
        y_pred_tensor = model((features_tensor, task_one_hot_tensor, steps_tensor))
    
    y_pred = y_pred_tensor.cpu().numpy()
    y_true = y_val
    errors = y_pred - y_true
    

    # このグラフを生成した検証データでの実際のMSEを計算
    actual_mse = np.mean(errors ** 2)
    actual_rmse = np.sqrt(actual_mse)
    
    print("---" * 15)
    print(f"このグラフを生成した検証データでの実際のMSE: {actual_mse:.2f}")
    print(f"このグラフを生成した検証データでの実際のRMSE: {actual_rmse:.2f} (J)")
    print("---" * 15)

    # --- 可視化（プロット部分は変更なし） ---
    plt.figure(figsize=(18, 5))
    # ... (以降のプロットコードは元のままでOK)
    
    # 1. 実績値 vs 予測値プロット
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, label="Predictions")
    perfect_line = np.linspace(min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()), 100)
    plt.plot(perfect_line, perfect_line, 'r--', label="Ideal Line (y=x)")
    plt.title("Actual vs. Predicted Plot")
    plt.xlabel("Actual Energy (J)")
    plt.ylabel("Predicted Energy (J)")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

    # 2. 誤差の分布（ヒストグラム）
    plt.subplot(1, 3, 2)
    plt.hist(errors, bins=50)
    plt.title("Error Distribution (Histogram)")
    plt.xlabel("Error (Predicted - Actual)")
    plt.ylabel("Count")
    plt.grid(True)

    # 3. 残差プロット
    plt.subplot(1, 3, 3)
    plt.scatter(y_true, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residual Plot")
    plt.xlabel("Actual Energy (J)")
    plt.ylabel("Error (Predicted - Actual)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize()