import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
import sys
import os
import re # ファイル名サニタイズ用
import matplotlib.pyplot as plt

# --- パラメータ ---
NPZ_FILE_PATH = 'fused711.npz'
OUTPUT_NPZ_PATH = 'fused711_hdbscan-0.001-MIN-5.npz' # 保存する新しいファイル名

# ★可視化グラフを保存するディレクトリ
VISUALIZATION_DIR = 'hdbscan_plots_true' 

# HDBSCANのパラメータ (タスクごとのデータ数に対応)
ABSOLUTE_MIN_SIZE = 5    # これより小さい集団はクラスタと認めない
PERCENTAGE_RATE = 0.001
# --------------------

def sanitize_filename(name):
    """ファイル名として使えない文字を除去または置換する"""
    return re.sub(r'[\\/*?:"<>|]',"_", name)

def plot_clusters(X_features, cluster_labels, core_mask, task_name, output_dir):
    """
    HDBSCANの結果を2種類の散布図で可視化し、保存する
    
    Args:
        X_features (np.array): 標準化する前の特徴量 (移動距離, エネルギー)
        cluster_labels (np.array): HDBSCANのラベル (-1, 0, 1, ...)
        core_mask (np.array): 外れ値(-1)以外がTrueのマスク
        task_name (str): タスク名 (ファイル名に使用)
        output_dir (str): 保存先ディレクトリ
    """
    
    # ファイル名に使えない文字を処理
    safe_task_name = sanitize_filename(task_name)
    
    # 軸ラベル
    xlabel = 'Movement Distance (m)'
    ylabel = 'Energy'

    # --- 1. 外れ値 vs 正常値プロット ---
    plt.figure(figsize=(10, 8))
    
    # 正常値 (Core)
    core_data = X_features[core_mask]
    core_count = len(core_data)
    if core_data.size > 0:
        plt.scatter(core_data[:, 0], core_data[:, 1], s=5, alpha=0.5, label=f'Core (Kept) N={core_count}')
    
    # 外れ値 (Outlier)
    outlier_data = X_features[~core_mask]
    outlier_count = len(outlier_data)
    if outlier_data.size > 0:
        plt.scatter(outlier_data[:, 0], outlier_data[:, 1], s=5, alpha=0.5, marker="x", label=f'Outlier (Removed) N={outlier_count}', c='black')

    plt.title(f'Outlier vs Core Data - Task: {task_name} N={core_count+outlier_count}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 保存
    plot_path_1 = os.path.join(output_dir, f"{safe_task_name}_1_outliers.png")
    plt.savefig(plot_path_1)
    plt.close() # メモリ解放

    # --- 2. 全クラスタープロット ---
    plt.figure(figsize=(10, 8))
    
    # クラスターラベルのユニークな値を取得 (-1, 0, 1, ...)
    unique_labels = np.unique(cluster_labels)

    # Colormapオブジェクトを取得 (新しい方法)
    cmap = plt.colormaps['hsv'] 
    # クラスターの最大IDを取得 (例: 0, 1, 2 があれば 2)
    # (-1 しかない場合は 0 にフォールバック)
    max_label = cluster_labels.max()
    if max_label <= 0:
        max_label = 1 # 0除算を避ける
    
    for label in unique_labels:
        mask = (cluster_labels == label)
        data = X_features[mask]
        count = len(data)
        
        if label == -1:
            # 外れ値 (-1)
            plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.3, marker="x", c='black', label=f'Outlier (-1) N={count}')
        else:
            # 正常なクラスター (0, 1, ...)
            norm_color = label / max_label
            plt.scatter(data[:, 0], data[:, 1], s=5, alpha=0.7, color=cmap(norm_color), label=f'Cluster {label} N={count}')
            
    plt.title(f'All Clusters - Task: {task_name} N={core_count+outlier_count}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 保存
    plot_path_2 = os.path.join(output_dir, f"{safe_task_name}_2_all_clusters.png")
    plt.savefig(plot_path_2)
    plt.close() # メモリ解放


def create_cleaned_dataset(npz_path, output_path, viz_dir):
    """
    HDBSCANを使用してタスクごとに外れ値を除去し、
    クリーンなデータのみを含む新しいNPZファイルを作成します。
    ★可視化機能が追加されています。
    """
    
    print(f"--- [Step 1] 元のNPZファイルをロード中: {npz_path} ---")
    try:
        data = np.load(npz_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {npz_path}", file=sys.stderr)
        return
        
    # ★可視化ディレクトリを作成
    os.makedirs(viz_dir, exist_ok=True)
    print(f"★可視化グラフは {viz_dir} にタスクごとに保存されます。")

    all_keys = list(data.keys())
    clean_data_dict = {key: [] for key in all_keys} 
    
    all_task_names = np.unique(data['task_name'])
    total_removed = 0
    total_initial = 0

    print(f"--- [Step 2] 全 {len(all_task_names)} タスクの外れ値除去を開始 ---")
    
    for task_name in all_task_names:
        mask = data['task_name'] == task_name
        
        # --- HDBSCAN用の特徴量を準備 ---
        robot_state = data['robot_state'][mask]
        goal_pos = data['soft_predicted_position'][mask]
        labels = data['energy'][mask].reshape(-1, 1)

        if len(labels) == 0:
            print(f"  [Task: {task_name}] ... スキップ (データなし)")
            continue

        current_pos = robot_state[:, 14:17]
        movement_distance = np.linalg.norm(goal_pos - current_pos, axis=1).reshape(-1, 1)
        X_features = np.concatenate([movement_distance, labels], axis=1) # ★標準化前のデータを保持
        
        initial_count = len(X_features)
        total_initial += initial_count
        
        if initial_count < (ABSOLUTE_MIN_SIZE * 2):
            print(f"  [Task: {task_name}] ... データ数 ({initial_count}) が少なすぎるため、全て保持します。")
            core_mask = np.full(initial_count, True) # 全てTrue (全て保持)
            cluster_labels = np.zeros(initial_count) # 全て 'Cluster 0' 扱いにする
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_features)

            min_size = max(ABSOLUTE_MIN_SIZE, int(initial_count * PERCENTAGE_RATE))
            
            clusterer = hdbscan.HDBSCAN(
                min_samples=ABSOLUTE_MIN_SIZE, # ★min_samples を ABSOLUTE_MIN_SIZE と一致させる
                min_cluster_size=min_size,
                allow_single_cluster=True,
            )
            cluster_labels = clusterer.fit_predict(X_scaled)
            core_mask = (cluster_labels != -1)
        
        num_removed = initial_count - np.sum(core_mask)
        total_removed += num_removed
        print(f"  [Task: {task_name}] ... 元データ: {initial_count}, 除去: {num_removed} ({num_removed/initial_count:.2%})")

        # --- ★可視化関数を呼び出す ---
        if initial_count > 0:
            plot_clusters(X_features, cluster_labels, core_mask, task_name, viz_dir)
        # -----------------------------

        # --- 「Core」データのみを辞書に追加 ---
        for key in all_keys:
            task_data_for_key = data[key][mask]
            clean_task_data = task_data_for_key[core_mask]
            clean_data_dict[key].append(clean_task_data)

    print("--- [Step 3] 全タスクの処理完了。新しいNPZファイルを作成中... ---")

    for key in all_keys:
        if clean_data_dict[key]:
            clean_data_dict[key] = np.concatenate(clean_data_dict[key])
        else:
            clean_data_dict[key] = np.array([])

    np.savez_compressed(output_path, **clean_data_dict)
    
    print("\n--- 完了 ---")
    print(f"元の総データ数: {total_initial}")
    print(f"除去された外れ値: {total_removed}")
    print(f"クリーンな総データ数: {total_initial - total_removed}")
    print(f"新しいファイルが保存されました: {output_path}")

if __name__ == '__main__':
    try:
        import matplotlib
    except ImportError:
        print("エラー: 'matplotlib' ライブラリが見つかりません。", file=sys.stderr)
        print("ターミナルで 'pip install matplotlib' を実行してインストールしてください。", file=sys.stderr)
        sys.exit(1)
        
    try:
        import hdbscan
    except ImportError:
        print("エラー: 'hdbscan' ライブラリが見つかりません。", file=sys.stderr)
        print("ターミナルで 'pip install hdbscan' を実行してインストールしてください。", file=sys.stderr)
        sys.exit(1)

    output_basename = os.path.splitext(os.path.basename(OUTPUT_NPZ_PATH))[0]
    final_viz_dir = os.path.join(VISUALIZATION_DIR, output_basename)
        
    create_cleaned_dataset(NPZ_FILE_PATH, OUTPUT_NPZ_PATH, final_viz_dir)