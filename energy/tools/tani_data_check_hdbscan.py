import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
import sys
import os

# --- パラメータ ---
NPZ_FILE_PATH = 'fused711.npz'
OUTPUT_NPZ_PATH = 'fused711_hdbscan-0.005.npz' # 保存する新しいファイル名

# HDBSCANのパラメータ (タスクごとのデータ数に対応)
ABSOLUTE_MIN_SIZE = 10    # これより小さい集団はクラスタと認めない
PERCENTAGE_RATE = 0.005
# --------------------

def create_cleaned_dataset(npz_path, output_path):
    """
    HDBSCANを使用してタスクごとに外れ値を除去し、
    クリーンなデータのみを含む新しいNPZファイルを作成します。
    """
    
    print(f"--- [Step 1] 元のNPZファイルをロード中: {npz_path} ---")
    try:
        data = np.load(npz_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {npz_path}", file=sys.stderr)
        return

    all_keys = list(data.keys())
    # 新しいNPZファイルに書き込むための辞書を初期化
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

        # 現在位置 (14:17)
        current_pos = robot_state[:, 14:17]
        # 移動距離
        movement_distance = np.linalg.norm(goal_pos - current_pos, axis=1).reshape(-1, 1)
        
        # クラスタリング対象 (移動距離, エネルギー)
        X_features = np.concatenate([movement_distance, labels], axis=1)
        
        initial_count = len(X_features)
        total_initial += initial_count
        
        # データが少なすぎるタスクは、安全のため全て保持する
        if initial_count < (ABSOLUTE_MIN_SIZE * 2):
            print(f"  [Task: {task_name}] ... データ数 ({initial_count}) が少なすぎるため、全て保持します。")
            core_mask = np.full(initial_count, True) # 全てTrue (全て保持)
        else:
            # --- 標準化 & HDBSCAN実行 ---
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_features)

            # データ数に応じて min_cluster_size を決定
            min_size = max(ABSOLUTE_MIN_SIZE, int(initial_count * PERCENTAGE_RATE))
            
            clusterer = hdbscan.HDBSCAN(
                min_samples=ABSOLUTE_MIN_SIZE,
                min_cluster_size=min_size,
                allow_single_cluster=True,
            )
            cluster_labels = clusterer.fit_predict(X_scaled)
            
            # -1 (ノイズ) 以外を「Core (正常)」データとする
            core_mask = (cluster_labels != -1)
        
        num_removed = initial_count - np.sum(core_mask)
        total_removed += num_removed
        print(f"  [Task: {task_name}] ... 元データ: {initial_count}, 除去: {num_removed} ({num_removed/initial_count:.2%})")

        # --- 「Core」データのみを辞書に追加 ---
        for key in all_keys:
            task_data_for_key = data[key][mask] # このタスクの全データを取得
            clean_task_data = task_data_for_key[core_mask] # Coreデータのみを抽出
            clean_data_dict[key].append(clean_task_data) # リストに追加

    print("--- [Step 3] 全タスクの処理完了。新しいNPZファイルを作成中... ---")

    # 各キーのリストを
    # (例: [task1_data, task2_data, ...])
    # 1つの大きな配列に結合
    for key in all_keys:
        if clean_data_dict[key]:
            clean_data_dict[key] = np.concatenate(clean_data_dict[key])
        else:
            clean_data_dict[key] = np.array([]) # データが空の場合

    # 新しいNPZファイルとして圧縮保存
    np.savez_compressed(output_path, **clean_data_dict)
    
    print("\n--- 完了 ---")
    print(f"元の総データ数: {total_initial}")
    print(f"除去された外れ値: {total_removed}")
    print(f"クリーンな総データ数: {total_initial - total_removed}")
    print(f"新しいファイルが保存されました: {output_path}")

if __name__ == '__main__':
    # hdbscan ライブラリが必要なことを確認
    try:
        import hdbscan
    except ImportError:
        print("エラー: 'hdbscan' ライブラリが見つかりません。", file=sys.stderr)
        print("ターミナルで 'pip install hdbscan' を実行してインストールしてください。", file=sys.stderr)
        sys.exit(1)
        
    create_cleaned_dataset(NPZ_FILE_PATH, OUTPUT_NPZ_PATH)