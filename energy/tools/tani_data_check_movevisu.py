import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

def visualize_movement_outliers(npz_file_path: str, val_split: float = 0.2, random_seed: int = 42, output_dir: str = 'split_visuals'):
    """
    データセットをロードし、学習時と同じロジックで分割を行い、
    「移動距離 vs エネルギー」の散布図と、「単位エネルギー」のヒストグラムを作成します。
    """
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- グラフの保存先: {output_dir} ---")

    try:
        with np.load(npz_file_path) as data:
            all_task_names = np.unique(data['task_name'])
            print(f"--- データ分析開始: {npz_file_path} ---")

            for task_name in all_task_names:
                mask = data['task_name'] == task_name
                
                # --- 学習コードと同じデータ抽出ロジック ---
                labels = data['energy'][mask].reshape(-1, 1)
                steps = data['step'][mask].astype(np.float32)
                robot_state = data['robot_state'][mask]
                
                # ▼▼▼▼▼ (修正点 1) 学習コードとinputsを揃えるため、predicted_poseを正しく定義 ▼▼▼▼▼
                goal_pos = data['soft_predicted_position'][mask]
                goal_rot = data['soft_predicted_rotation_quat'][mask]
                predicted_pose = np.concatenate([goal_pos, goal_rot], axis=1)
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

                grip = data['soft_predicted_gripper_state'][mask].reshape(-1, 1).astype(np.float32)
                coll = data['soft_predicted_collision_state'][mask].reshape(-1, 1).astype(np.float32)
                fused_feat = data['rvt_fused_feat'][mask]
                
                # 学習コードと同一の inputs を作成（splitのランダム性を一致させるため）
                inputs = np.concatenate([robot_state, predicted_pose, grip, coll, fused_feat], axis=1)
                
                if len(labels) == 0:
                    print(f"--- [Task: {task_name}] ... スキップ (データなし) ---")
                    continue

                # ▼▼▼▼▼ (修正点 2) ご指摘の通り、正しいインデックス [14:17] を使用 ▼▼▼▼▼
                # robot_state の [14:17] が「現在の」X,Y,Z位置
                current_pos = robot_state[:, 14:17]
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                
                if current_pos.shape[1] != 3 or goal_pos.shape[1] != 3:
                    print(f"--- [Task: {task_name}] ... スキップ (位置データの次元が不正) ---")
                    continue
                    
                # 「移動距離」を計算 (ユークリッド距離)
                movement_distance = np.linalg.norm(goal_pos - current_pos, axis=1).reshape(-1, 1)
                
                # --- 学習コードと全く同じ分割を実行 ---
                # movement_distance も一緒に分割する
                _, _, train_labels, val_labels, _, _, train_dist, val_dist = train_test_split(
                    inputs,
                    labels,
                    steps,
                    movement_distance, # <--- これも分割
                    test_size=val_split,
                    random_state=random_seed, # シードを固定
                    shuffle=True
                )
                
                if len(train_labels) == 0 or len(val_labels) == 0:
                    print(f"--- [Task: {task_name}] ... スキップ (TrainまたはValが空) ---")
                    continue
                    
                print(f"--- [Task: {task_name}] ... グラフ生成中 ---")

                # --- 1. 散布図 (Energy vs Movement Distance) ---
                plt.figure(figsize=(12, 7))
                plt.scatter(train_dist, train_labels, alpha=0.3, label=f'Train (N={len(train_labels)})', color='blue', s=10)
                plt.scatter(val_dist, val_labels, alpha=0.3, label=f'Val (N={len(val_labels)})', color='red', s=10)
                plt.title(f'Energy vs Movement Distance - Task: {task_name}')
                plt.xlabel('Movement Distance (Goal - Current)')
                plt.ylabel('Energy')
                plt.legend()
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                
                output_filename = os.path.join(output_dir, f"task_{task_name}_scatter_dist_vs_energy.png")
                plt.savefig(output_filename)
                plt.close()

                # --- 2. 「単位エネルギー」Log(1 + E / Dist) のヒストグラム ---
                epsilon = 1e-8 # ゼロ除算を避ける
                train_specific_energy = train_labels.flatten() / (train_dist.flatten() + epsilon)
                val_specific_energy = val_labels.flatten() / (val_dist.flatten() + epsilon)

                train_log_spec_E = np.log1p(train_specific_energy)
                val_log_spec_E = np.log1p(val_specific_energy)
                
                combined_log_spec_E = np.concatenate([train_log_spec_E, val_log_spec_E])
                max_val = np.percentile(combined_log_spec_E, 99)
                bins = np.linspace(0, max_val, 100) 

                plt.figure(figsize=(12, 7))
                plt.hist(train_log_spec_E, bins=bins, alpha=0.7, label=f'Train (N={len(train_labels)})', density=True, color='blue')
                plt.hist(val_log_spec_E, bins=bins, alpha=0.7, label=f'Val (N={len(val_labels)})', density=True, color='red')
                plt.title(f'Log(Specific Energy) Distribution - Task: {task_name}')
                plt.xlabel('Log(1 + Energy / Distance)')
                plt.ylabel('Density (Normalized)')
                plt.legend()
                
                output_filename_spec_E = os.path.join(output_dir, f"task_{task_name}_hist_specific_energy.png")
                plt.savefig(output_filename_spec_E)
                plt.close()

            print(f"\n--- 全タスクの可視化完了。{output_dir} を確認してください。---")

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {npz_file_path}", file=sys.stderr)
    except KeyError as e:
        print(f"エラー: NPZファイルに必要なキーがありません: {e}", file=sys.stderr)
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)

if __name__ == '__main__':
    NPZ_FILE_PATH = 'fused711.npz'
    visualize_movement_outliers(NPZ_FILE_PATH)