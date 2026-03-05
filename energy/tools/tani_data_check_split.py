import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

def visualize_outliers(npz_file_path: str, val_split: float = 0.2, random_seed: int = 42, output_dir: str = 'split_visuals'):
    """
    データセットをロードし、学習時と同じロジックで分割を行い、
    訓練データと検証データのエネルギー分布（ヒストグラム）を可視化して保存します。
    """
    
    # 保存用ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- グラフの保存先: {output_dir} ---")

    try:
        with np.load(npz_file_path) as data:
            all_task_names = np.unique(data['task_name'])
            print(f"--- データ分析開始: {npz_file_path} ---")
            print(f"--- 分割シード: {random_seed}, 検証データ比率: {val_split} ---\n")

            for task_name in all_task_names:
                mask = data['task_name'] == task_name
                
                # --- 学習コードと同じデータ抽出ロジック ---
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
                steps = data['step'][mask].astype(np.float32)
                
                if len(labels) == 0:
                    print(f"--- [Task: {task_name}] ... スキップ (データなし) ---")
                    continue

                # --- 学習コードと全く同じ分割を実行 ---
                _, _, train_labels, val_labels, _, _ = train_test_split(
                    inputs,
                    labels,
                    steps,
                    test_size=val_split,
                    random_state=random_seed, # シードを固定
                    shuffle=True
                )
                
                if len(train_labels) == 0 or len(val_labels) == 0:
                    print(f"--- [Task: {task_name}] ... スキップ (TrainまたはValが空) ---")
                    continue
                    
                print(f"--- [Task: {task_name}] ... グラフ生成中 ---")

                # --- ビン（棒）の数を計算 ---
                max_val = max(np.max(train_labels), np.max(val_labels))
                min_val = min(np.min(train_labels), np.min(val_labels))
                # ビンの幅が大きくなりすぎないように、最大でも100ビンにする
                bins = min(int(max_val - min_val), 100)
                if bins < 10: bins = 10

                # --- 1. 全体図のプロット ---
                plt.figure(figsize=(12, 7))
                plt.hist(train_labels, bins=bins, alpha=0.7, label=f'Train (N={len(train_labels)})', density=True, color='blue')
                plt.hist(val_labels, bins=bins, alpha=0.7, label=f'Val (N={len(val_labels)})', density=True, color='red')
                plt.title(f'Energy Distribution (Train vs Val) - Task: {task_name}')
                plt.xlabel('Energy')
                plt.ylabel('Density (Normalized)')
                plt.legend()
                
                output_filename = os.path.join(output_dir, f"task_{task_name}_hist.png")
                plt.savefig(output_filename)
                plt.close() # メモリ解放

                # --- 2. 95パーセンタイルまでの拡大図 ---
                p95 = np.percentile(labels, 95) # 全体の95%点
                
                plt.figure(figsize=(12, 7))
                plt.hist(train_labels, bins=bins, alpha=0.7, label=f'Train (N={len(train_labels)})', density=True, color='blue')
                plt.hist(val_labels, bins=bins, alpha=0.7, label=f'Val (N={len(val_labels)})', density=True, color='red')
                
                plt.title(f'Energy Distribution (Zoomed to 95%) - Task: {task_name}')
                plt.xlabel('Energy')
                plt.ylabel('Density (Normalized)')
                plt.legend()
                plt.xlim(min_val, p95) # X軸を95パーセンタイルまででクリップ
                
                output_filename_zoomed = os.path.join(output_dir, f"task_{task_name}_hist_zoomed95.png")
                plt.savefig(output_filename_zoomed)
                plt.close()

            print(f"\n--- 全タスクの可視化完了。{output_dir} を確認してください。---")

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {npz_file_path}", file=sys.stderr)
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)

if __name__ == '__main__':
    NPZ_FILE_PATH = 'fused711.npz' # 学習コードと同じパス
    visualize_outliers(NPZ_FILE_PATH)