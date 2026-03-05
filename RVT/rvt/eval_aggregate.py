import subprocess
import multiprocessing
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import json

# ----------------------------------------
# ▼▼▼ ユーザー設定 ▼▼▼
# ----------------------------------------

MODEL_FOLDER = "runs/rvt2" 

# MODEL_NAME = "tf-freeze-80.pth"
# BASE_LOG_NAME = "TF-freeze-80_aggregate"

# MODEL_NAME = "tf_notfuse_9.pth"
# BASE_LOG_NAME = "tf_notfuse_9_aggregate"

# MODEL_NAME = "tf_nohdbscan_test110.pth"
# BASE_LOG_NAME = "tf_nohdbscan_test110_aggregate"
# MODEL_NAME = "addexp-10-7-not.pth"
# BASE_LOG_NAME = "addexp-10-7-not_aggregate_seed322"
# MODEL_NAME = "addexp-15-5-177-yet.pth"
# BASE_LOG_NAME = "addexp-15-5-177-yet_aggregate_seed1234"
# MODEL_NAME = "addexp-10-7-2-123.pth"
# BASE_LOG_NAME = "addexp-10-7-2-123_aggregate_seed1234"
# MODEL_NAME = "addexp-17-7.pth"
# BASE_LOG_NAME = "addexp-17-7_aggregate_seed3333"
# MODEL_NAME = "addexp-10-5-134.pth"
# BASE_LOG_NAME = "addexp-10-5-134_aggregate_seed1234"
# MODEL_NAME = "addexp-15-7.pth"
# BASE_LOG_NAME = "addexp-15-7_aggregate_seed1234"

# MODEL_NAME = "tf_notfuse_8.pth"
# BASE_LOG_NAME = "tf_notfuse_8_aggregate_part3"

# MODEL_NAME = "model_99.pth"
# BASE_LOG_NAME = "rvt_normal_aggregate_part6"
# MODEL_NAME = "model_99.pth"
# BASE_LOG_NAME = "rvt_normal_aggregate_seed5555_1"


MODEL_NAME = "finetune_best.pth"
BASE_LOG_NAME = "finetune_best_100"





FIXED_SEED_BASE = 100
# FIXED_SEED_BASE = 1234

TASKS = "all"
EVAL_EPISODES = 25
DATA_PATH = "./data/test"

# ★ 並列数を増やすなら、実行回数(NUM_RUNS)も増やさないと意味がない場合が多いです
# 例: 4GPU * 2プロセス = 8並列 なので、最低8回は回さないと空きが出ます
NUM_RUNS = 4

# AVAILABLE_GPUS = [0]
# AVAILABLE_GPUS = [4,5,6,7]
# AVAILABLE_GPUS = [5,6,7,8]
AVAILABLE_GPUS = [0,1,2,3]


# AVAILABLE_GPUS = [8,9]
# AVAILABLE_GPUS = [6,7]
# AVAILABLE_GPUS = [4,5]
# AVAILABLE_GPUS = [2,3]
# AVAILABLE_GPUS = [0,1]




# ★★★ 新規設定: 1GPUあたりに走らせるプロセス数 ★★★
PROCS_PER_GPU = 1


OTHER_ARGS = [
    "--headless",
    "--save-video"
]

# ----------------------------------------
# ▲▲▲ ユーザー設定 ▲▲▲
# ----------------------------------------

# ★ 修正: GPU数 × 1GPUあたりのプロセス数
NUM_WORKERS = len(AVAILABLE_GPUS) * PROCS_PER_GPU

if len(AVAILABLE_GPUS) == 0:
    print("エラー: AVAILABLE_GPUS に使用するGPU IDを最低1つ指定してください。")
    sys.exit(1)

def run_single_eval(run_index):
    """
    eval.py を1回実行する関数
    """
    try:
        sub_dir_name = f"run_{run_index + 1}"
        relative_log_path = os.path.join(BASE_LOG_NAME, sub_dir_name)
        
        physical_gpu_id = AVAILABLE_GPUS[run_index % len(AVAILABLE_GPUS)]
        
        # seed 固定化
        current_seed = FIXED_SEED_BASE + run_index

        model_log_dir_name = MODEL_NAME.replace(".pth", "")
        log_dir = os.path.join(MODEL_FOLDER, "eval", relative_log_path, model_log_dir_name)
        csv_path = os.path.join(log_dir, "tani_log.csv")

        print(f"[Run {run_index+1}/{NUM_RUNS}] 開始. GPU: {physical_gpu_id}, Seed: {current_seed}, Log: {relative_log_path}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

        command = [
            "xvfb-run", "--auto-servernum",
            sys.executable, "eval.py",
            "--eval-datafolder", DATA_PATH,
            "--model-folder", MODEL_FOLDER,
            "--model-name", MODEL_NAME,
            "--log-name", relative_log_path,
            "--device", "0",
            "--tasks", *TASKS.split(),
            "--eval-episodes", str(EVAL_EPISODES),
            "--seed", str(current_seed),
            *OTHER_ARGS
        ]

        subprocess.run(command, env=env, check=True, capture_output=True, text=True, encoding='utf-8')
        
        print(f"[Run {run_index+1}/{NUM_RUNS}] 正常終了.")
        return csv_path 
        
    except subprocess.CalledProcessError as e:
        print(f"--- [Run {run_index+1}/{NUM_RUNS}] エラー ---")
        print(e.stdout)
        print(e.stderr)
        print(f"------------------------------------")
        return None
    except Exception as e:
        print(f"[Run {run_index+1}/{NUM_RUNS}] 予期せぬエラー: {e}")
        return None

def analyze_log(log_path):
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"警告: 集計スキップ (ログファイルが見つかりません: {log_path})", file=sys.stderr)
        return None
    
    results = {}
    for task_name, group in df.groupby('task_name'):
        success_rate = (group['reward'] == 100.0).mean()
        avg_energy_all = group['energy'].mean()
        success_cases = group[group['reward'] == 100.0]
        avg_energy_success = success_cases['energy'].mean() if not success_cases.empty else np.nan
            
        results[task_name] = {
            'success_rate': success_rate,
            'avg_energy_all': avg_energy_all,
            'avg_energy_success': avg_energy_success
        }
    return results

def aggregate_results(log_paths):
    stats = defaultdict(lambda: defaultdict(list)) 
    valid_run_count = 0

    for log_path in log_paths:
        if log_path is None: continue 
        run_metrics = analyze_log(log_path)
        if run_metrics is None: continue
            
        valid_run_count += 1
        for task, metrics in run_metrics.items():
            stats[task]['success_rate'].append(metrics['success_rate'])
            stats[task]['avg_energy_all'].append(metrics['avg_energy_all'])
            if not np.isnan(metrics['avg_energy_success']):
                stats[task]['avg_energy_success'].append(metrics['avg_energy_success'])

    final_results = {}
    for task, metrics_data in stats.items():
        final_results[task] = {}
        for metric_name, values in metrics_data.items():
            if not values:
                mean, std = np.nan, np.nan
            else:
                mean = np.mean(values)
                std = np.std(values)
            
            final_results[task][f"{metric_name}_mean"] = mean
            final_results[task][f"{metric_name}_std"] = std
    
    return final_results

def main():
    print(f"評価を {NUM_RUNS} 回実行します。")
    print(f"使用GPU: {AVAILABLE_GPUS}")
    print(f"プロセス数/GPU: {PROCS_PER_GPU}")
    print(f"総並列ワーカー数: {NUM_WORKERS}")
    
    # 警告ロジックも更新
    total_slots = len(AVAILABLE_GPUS) * PROCS_PER_GPU
    if NUM_WORKERS > total_slots:
        print("警告: ワーカー数が想定スロット数を超えています。")

    print("並列実行を開始...")
    with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
        log_paths = pool.map(run_single_eval, range(NUM_RUNS))

    print("\nすべての評価実行が完了しました。")
    print("結果を集計中...")
    aggregated_metrics = aggregate_results(log_paths)
    
    output_dir = os.path.join(MODEL_FOLDER, "eval", BASE_LOG_NAME)
    os.makedirs(output_dir, exist_ok=True)

    csv_data = []
    for task_name, metrics in aggregated_metrics.items():
        row = {"task_name": task_name}
        row.update(metrics)
        csv_data.append(row)
    
    json_path = os.path.join(output_dir, "aggregated_result.json") # パスを定義

    if csv_data:
        df_res = pd.DataFrame(csv_data)
        csv_path = os.path.join(output_dir, "aggregated_result.csv")
        df_res.to_csv(csv_path, index=False)
        
        # JSONも保存
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_metrics, f, indent=2, ensure_ascii=False)

        print(f"\n結果を保存しました:\n  CSV: {csv_path}\n  JSON: {json_path}")
    else:
        print("\n保存する結果データがありませんでした。")

    print("\n--- 総合結果 (簡易表示) ---")
    print(json.dumps(aggregated_metrics, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) 
    main()