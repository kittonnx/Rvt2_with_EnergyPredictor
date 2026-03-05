import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_single_plot_comparison(input_npz_path):
    """
    1つのプロットエリア内に、全データと高コストデータのみの
    箱ひげ図を並べて比較・可視化する。タスクの順序は固定。
    """
    # --- 1. データの読み込みと指標計算 ---
    print(f"--- 1. データの読み込み: {input_npz_path} ---")
    data = np.load(input_npz_path)
    
    robot_state = data['robot_state']
    predicted_position = data['predicted_position']
    energy = data['energy']
    task_name = data['task_name']
    
    current_positions = robot_state[:, :3] 
    target_positions = predicted_position[:, :3]
    move_distance = np.linalg.norm(target_positions - current_positions, axis=1)
    epsilon = 1e-6
    energy_per_distance = energy.flatten() / (move_distance + epsilon)

    df = pd.DataFrame({
        'task_name': task_name,
        'energy_per_distance': energy_per_distance
    })

    # --- 2. 高コストデータ（上位25%）のみを抽出 ---
    print("--- 2. 高コストデータ（上位25%）を抽出中... ---")
    
    high_cost_df = df.groupby('task_name').apply(
        lambda x: x[x['energy_per_distance'] > x['energy_per_distance'].quantile(0.75)]
    ).reset_index(drop=True)


    # --- 3. 比較グラフを作成 ---
    print("--- 3. 比較グラフを生成中... ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(28, 10))
    fig.suptitle("Comparison of Data Distributions", fontsize=20)

    # ▼▼▼ 変更点：プロットの順序を固定するためのリストを作成 ▼▼▼
    # アルファベット順でソートすることで、常に同じ順序を保証
    task_order = sorted(df['task_name'].unique())

    # --- 左のグラフ (全データ) ---
    sns.boxplot(ax=axes[0], x='task_name', y='energy_per_distance', data=df, order=task_order) # <-- orderを指定
    axes[0].set_title("Full Data Distribution", fontsize=16)
    axes[0].set_xlabel("Task Name", fontsize=12)
    axes[0].set_ylabel("Energy / Distance", fontsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    axes[0].grid(True)

    # --- 右のグラフ (高コストデータのみ) ---
    sns.boxplot(ax=axes[1], x='task_name', y='energy_per_distance', data=high_cost_df, order=task_order) # <-- orderを指定
    axes[1].set_title("High-Cost Group Only (Top 25% of each Task)", fontsize=16)
    axes[1].set_xlabel("Task Name", fontsize=12)
    axes[1].set_ylabel("Energy / Distance (Zoomed)", fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("--- 4. グラフを表示します ---")
    plt.show()


if __name__ == '__main__':
    INPUT_NPZ = 'task7.npz'
    visualize_single_plot_comparison(INPUT_NPZ)