import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
import sys
import os
import re # ファイル名サニタイズ用
import matplotlib.pyplot as plt


# ★可視化グラフを保存するディレクトリ
VISUALIZATION_DIR = 'hdbscan_plots_step' 

# HDBSCANのパラメータ (タスクごとのデータ数に対応)
MIN_SAMPLE = 10
ABSOLUTE_MIN_SIZE = 10    # これより小さい集団はクラスタと認めない
PERCENTAGE_RATE = 0.001
EPLISON = 0.7
# --------------------

# --- パラメータ ---
NPZ_FILE_PATH = 'fused7111216.npz'
OUTPUT_NPZ_PATH = f'fused7111216_hdbscan-0.001-sample-{MIN_SAMPLE}-MIN-{ABSOLUTE_MIN_SIZE}-step-eplison-{EPLISON}-under10-coloer.npz' # 保存する新しいファイル名

def sanitize_filename(name):
    """ファイル名として使えない文字を除去または置換する"""
    return re.sub(r'[\\/*?:"<>|]',"_", name)

def plot_task_outliers(X_features_2d_all, core_mask_all, task_name, output_dir):
    """
    「タスク全体」の外れ値/正常値プロットを1枚描画する
    (中身は「ステップごと」のクラスタリング結果の集計)
    
    Args:
        X_features_2d_all (np.array): タスク全体の2D特徴量
        core_mask_all (np.array): タスク全体のCore/Outlierマスク
        task_name (str): タスク名
        output_dir (str): 保存先
    """
    safe_task_name = sanitize_filename(task_name)
    xlabel = 'Movement Distance (m)'
    ylabel = 'Energy'
    total_count = len(X_features_2d_all)

    plt.figure(figsize=(10, 8))
    
    # 正常値 (Core)
    core_data = X_features_2d_all[core_mask_all]
    core_count = len(core_data)
    if core_data.size > 0:
        plt.scatter(core_data[:, 0], core_data[:, 1], s=5, alpha=0.5, label=f'Core (Kept) N={core_count}')
    
    # 外れ値 (Outlier)
    outlier_data = X_features_2d_all[~core_mask_all]
    outlier_count = len(outlier_data)
    if outlier_data.size > 0:
        plt.scatter(outlier_data[:, 0], outlier_data[:, 1], s=5, alpha=0.5, marker="x", label=f'Outlier (Removed) N={outlier_count}', c='black')

    plt.title(f'[Task-Wide] Outlier vs Core (from Per-Step clustering) - Task: {task_name} N={total_count}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # ★ファイル名を要望1に合わせて変更
    plot_path_1 = os.path.join(output_dir, f"{safe_task_name}_1_TaskWide_Outliers.png")
    plt.savefig(plot_path_1)
    plt.close()

def plot_step_facets(X_features_3d_all, cluster_labels_all, task_name, output_dir):
    """
    ステップごとの散布図（修正版）
    点を極小・透明にすることで、データが多くても密度がある程度わかるようにする
    """
    safe_task_name = sanitize_filename(task_name)
    xlabel = 'Movement Distance (m)'
    ylabel = 'Energy'
    
    steps = X_features_3d_all[:, 2]
    unique_steps = np.unique(steps).astype(int)
    n_steps = len(unique_steps)
    
    if n_steps == 0:
        return
    
    ncols = min(n_steps, 5)
    nrows = int(np.ceil(n_steps / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 5), squeeze=False)
    axes_flat = axes.flatten()
    
    for i, step in enumerate(unique_steps):
        ax = axes_flat[i]
        
        step_mask = (steps == step)
        step_features_3d = X_features_3d_all[step_mask]
        step_labels = cluster_labels_all[step_mask]
        
        step_unique_labels = np.unique(step_labels)
        cmap = plt.colormaps['hsv']
        max_label = step_labels.max()
        if max_label <= 0: max_label = 1
        
        for label in step_unique_labels:
            label_mask = (step_labels == label)
            data_2d = step_features_3d[label_mask][:, :2]
            count = len(data_2d)
            
            if label == -1:
                # 外れ値
                color = 'black'
                label_text = f'Outlier ({count})'
                marker = 'x'
                s_size = 2.0   # 外れ値は少し大きくてもよい
                alpha_val = 0.3
            else:
                # 正常値（クラスタ）
                color = cmap(label / max_label)
                label_text = f'C{label} ({count})'
                marker = 'o'
                # ★修正: 点を極小にして透明度を下げる
                s_size = 1   
                alpha_val = 0.3 
            
            ax.scatter(
                data_2d[:, 0], 
                data_2d[:, 1], 
                s=s_size, 
                alpha=alpha_val, 
                color=color, 
                label=label_text,
                marker=marker,
                edgecolors='none' # 枠線を消して描画を軽くする
            )
            
        ax.set_title(f'Step {step} (N={len(step_features_3d)})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # 凡例は要素が多いと邪魔になるため、文字サイズを小さく、マーカーを大きく表示
        lgnd = ax.legend(fontsize='x-small', markerscale=5)
        # 凡例の点の透明度は不透明に戻して見やすくする
        for lh in lgnd.legend_handles: 
            lh.set_alpha(1)

        ax.grid(True, linestyle='--', alpha=0.5)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
        
    fig.suptitle(f'Per-Step Cluster Analysis (Scatter) - Task: {task_name}', fontsize=16, y=1.03)
    fig.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{safe_task_name}_2_PerStep_Facets.png")
    plt.savefig(plot_path)
    plt.close()

from matplotlib.colors import LogNorm

def plot_step_heatmaps(X_features_3d_all, task_name, output_dir):
    """
    ステップごとの密度ヒートマップ（新規追加）
    散布図では潰れて見えない「どこに集中しているか」を可視化する
    """
    safe_task_name = sanitize_filename(task_name)
    xlabel = 'Movement Distance (m)'
    ylabel = 'Energy'
    
    steps = X_features_3d_all[:, 2]
    unique_steps = np.unique(steps).astype(int)
    n_steps = len(unique_steps)
    
    if n_steps == 0:
        return
    
    ncols = min(n_steps, 5)
    nrows = int(np.ceil(n_steps / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 5), squeeze=False)
    axes_flat = axes.flatten()
    
    for i, step in enumerate(unique_steps):
        ax = axes_flat[i]
        
        step_mask = (steps == step)
        step_features_3d = X_features_3d_all[step_mask]
        data_2d = step_features_3d[:, :2] # 距離とエネルギー
        
        if len(data_2d) == 0:
            continue
            
        # 2Dヒストグラム (ヒートマップ)
        # bins: グリッドの細かさ (50x50 ~ 100x100 くらいが適当)
        # norm: LogNorm() で対数スケールにすると、少ない外れ値と超密集地帯を同時に見やすい
        h = ax.hist2d(
            data_2d[:, 0], 
            data_2d[:, 1], 
            bins=80, 
            cmap='inferno', 
            norm=LogNorm(),
            cmin=1 # 1個以上ある場所だけ塗る
        )
        
        # 各サブプロットにカラーバーをつけると見づらくなるので、ここでは省略するか、
        # 必要なら fig.colorbar(h[3], ax=ax) を追加
        
        ax.set_title(f'Step {step} Density (N={len(data_2d)})')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
        
    fig.suptitle(f'Per-Step Density (Heatmap) - Task: {task_name}', fontsize=16, y=1.03)
    fig.tight_layout()
    
    # 別ファイルとして保存
    plot_path = os.path.join(output_dir, f"{safe_task_name}_3_PerStep_Heatmaps.png")
    plt.savefig(plot_path)
    plt.close()

def create_cleaned_dataset(npz_path, output_path, viz_dir):
    """
    ★ロジック: 「ステップごと」にクラスタリングを実行。
    ★可視化: ステップごとの結果を集約して「タスク全体」のプロットも描画する。
    """
    
    print(f"--- [Step 1] 元のNPZファイルをロード中: {npz_path} ---")
    try:
        data = np.load(npz_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {npz_path}", file=sys.stderr)
        return
        
    if 'step' not in data:
        print(f"エラー: NPZファイルに 'step' キーが含まれていません。処理を中断します。", file=sys.stderr)
        return

    os.makedirs(viz_dir, exist_ok=True)
    print(f"★可視化グラフは {viz_dir} にタスクごとに保存されます。")

    all_keys = list(data.keys())
    clean_data_dict = {key: [] for key in all_keys} 
    
    all_task_names = np.unique(data['task_name'])
    total_removed = 0
    total_initial = 0

    print(f"--- [Step 2] 全タスクの「ステップごと」の外れ値除去を開始 ---")
    
    for task_name in all_task_names:
        
        task_mask = data['task_name'] == task_name
        task_data_dict = {key: data[key][task_mask] for key in all_keys}
        
        robot_state = task_data_dict['robot_state']
        goal_pos = task_data_dict['soft_predicted_position']
        labels = task_data_dict['energy'].reshape(-1, 1)
        steps = task_data_dict['step'].reshape(-1, 1)

        if len(labels) == 0:
            print(f"  [Task: {task_name}] ... スキップ (データなし)")
            continue
            
        current_pos = robot_state[:, 14:17]
        movement_distance = np.linalg.norm(goal_pos - current_pos, axis=1).reshape(-1, 1)

        X_features_3d = np.concatenate([movement_distance, labels, steps], axis=1)

        # --- ★可視化のために、ステップごとの結果を一時保存するリスト ---
        agg_features_3d_list = [] # (N, 3) 
        agg_cluster_labels_list = [] # (N,)
        agg_core_mask_list = [] # (N,)
        # -----------------------------------------------------------
        
        # unique_steps_in_task = np.unique(steps).astype(int)

        unique_steps_in_task, counts = np.unique(steps, return_counts=True)
        unique_steps_in_task = unique_steps_in_task.astype(int)
        threshold_n = counts[0]*0.1
        
        for step in unique_steps_in_task:
            step_mask = (X_features_3d[:, 2] == step)
            X_features_2d = X_features_3d[step_mask][:, :2]
            
            initial_count = len(X_features_2d)
            total_initial += initial_count

            step_task_name = f"{task_name}_Step_{step}"

            if initial_count < threshold_n:
                print(f"  [Task: {step_task_name}] ... データ数 ({initial_count}) が閾値 ({threshold_n:.1f}) 未満のため、全除去。")
                core_mask = np.full(initial_count, False) 
                cluster_labels = np.full(initial_count, -1)
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_features_2d)

                min_size = max(ABSOLUTE_MIN_SIZE, int(initial_count * PERCENTAGE_RATE))
                
                clusterer = hdbscan.HDBSCAN(
                    min_samples=MIN_SAMPLE,
                    min_cluster_size=min_size,
                    allow_single_cluster=True,
                    cluster_selection_epsilon=EPLISON
                )
                cluster_labels = clusterer.fit_predict(X_scaled)
                core_mask = (cluster_labels != -1)
            
            num_removed = initial_count - np.sum(core_mask)
            total_removed += num_removed
            
            if initial_count > 0:
                 # ★ print 文をタスク名からステップタスク名に変更
                 print(f"  [Task: {step_task_name}] ... 元データ: {initial_count}, 除去: {num_removed} ({num_removed/initial_count:.2%})")
            
            # --- ★ 可視化関数 (plot_clusters) の呼び出しを削除 ---
            # if initial_count > 0:
            #    plot_clusters(X_features_2d, cluster_labels, core_mask, step_task_name, viz_dir)

            # ★可視化のために結果を保存
            if initial_count > 0:
                agg_features_3d_list.append(X_features_3d[step_mask])
                agg_cluster_labels_list.append(cluster_labels)
                agg_core_mask_list.append(core_mask)

            # --- 「Core」データのみを辞書に追加 ---
            for key in all_keys:
                task_data_for_key = task_data_dict[key]
                step_data_for_key = task_data_for_key[step_mask]
                clean_step_data = step_data_for_key[core_mask] 
                clean_data_dict[key].append(clean_step_data)
        
        # --- ★ステップのループ終了後、タスク単位で可視化を実行 ---
        if len(agg_features_3d_list) > 0:
            # 1. 全ステップの結果をタスク全体に集約
            X_features_3d_all = np.concatenate(agg_features_3d_list)
            cluster_labels_all = np.concatenate(agg_cluster_labels_list)
            core_mask_all = np.concatenate(agg_core_mask_list)
            
            # 2. 可視化関数を呼び出す
            
            # 要望1: タスク全体の外れ値/正常値プロット
            X_features_2d_all = X_features_3d_all[:, :2]
            plot_task_outliers(X_features_2d_all, core_mask_all, task_name, viz_dir)
            
            # 要望3: ステップごとのファセットプロット
            plot_step_facets(X_features_3d_all, cluster_labels_all, task_name, viz_dir)

            # ★新規追加: ステップごとのヒートマップ (密度確認用)
            # plot_step_heatmaps(X_features_3d_all, task_name, viz_dir)
        # -------------------------------------------------

    print("--- [Step 3] 全タスク/全ステップの処理完了。新しいNPZファイルを作成中... ---")

    for key in all_keys:
        if clean_data_dict[key]:
            clean_data_dict[key] = np.concatenate(clean_data_dict[key])
        else:
            clean_data_dict[key] = np.array([])

    # np.savez_compressed(output_path, **clean_data_dict)
    
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