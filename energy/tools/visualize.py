# visualize_results.py (最終版)

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# --- 1. 実際に学習させた、正しいモデル定義に差し替え ---
class TorquePredictorMLP(nn.Module):
    def __init__(self, dropout=0.5): # ドロップアウト率は学習時と同じ値が理想
        super(TorquePredictorMLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(28, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(256, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)

# --- 2. 可視化のメイン処理 ---
def visualize():
    # --- 設定 ---
    # NPZ_FILE_PATH = "low_500_energy.npz"
    NPZ_FILE_PATH = "task1234.npz"
    # ★★★ .ckptファイル名を指定してください ★★★
    MODEL_PATH = "./checkpoints/model-epoch=1122-val_loss=182.2.ckpt" 
    VAL_SPLIT_RATIO = 0.2

    # --- デバイス設定 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # --- モデルの読み込み ---
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。")
        return
    
    # 正しい構造でモデルの「器」を準備
    model = TorquePredictorMLP().to(device)

    print(f"チェックポイントファイル '{MODEL_PATH}' を読み込んでいます...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # .ckptファイルから 'state_dict' を取り出して読み込む
    state_dict = checkpoint['state_dict']
    
    # (オプション) キーに余計な接頭辞('model.')がついていたら削除する
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("model.", "layers.") # 'model.'という接頭辞を削除
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval() # 評価モードに設定

    # --- データの準備 ---
    if not os.path.exists(NPZ_FILE_PATH):
        print(f"エラー: データファイル '{NPZ_FILE_PATH}' が見つかりません。")
        return
        
    data = np.load(NPZ_FILE_PATH)
    inputs = np.concatenate([
        data['robot_state'],
        data['predicted_position'],
        data['predicted_rotation_quat']
    ], axis=1)
    labels = data['energy'].reshape(-1, 1)
    
    _, X_val, _, y_val = train_test_split(
        inputs, labels, test_size=VAL_SPLIT_RATIO, random_state=42
    )
    X_val = inputs
    y_val = labels

    # --- 予測の実行 ---
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_tensor = model(X_val_tensor)
    
    y_pred = y_pred_tensor.cpu().numpy()
    y_true = y_val
    errors = y_pred - y_true
    
    # --- 可視化 ---
    plt.figure(figsize=(18, 5))
    # 日本語フォントの設定（お使いの環境に合わせてコメントアウトを解除・変更してください）
    # plt.rcParams["font.family"] = "Meiryo" # Windows
    # plt.rcParams["font.family"] = "Hiragino Maru Gothic Pro" # Mac

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