#!/bin/bash


#!/bin/bash

# --- 設定項目 ---

DATASET_NAME="task_15"

# データセットの保存先パス (ご自身の環境に合わせて変更してください)
SAVE_PATH="/misc/dl001/dataset/tani/${DATASET_NAME}"
# ログファイルの保存先ディレクトリ
LOG_DIR="./rlbench_logs/${DATASET_NAME}"
# 各タスクで収集するエピソード数
EPISODES_PER_TASK=500
# その他のオプション
IMAGE_SIZE="128,128"
RENDERER="opengl"

# 実行対象のタスクリスト (18個)
TASKS=(
    close_jar
    insert_onto_square_peg
    light_bulb_in
    meat_off_grill
    open_drawer
    place_cups
    place_shape_in_shape_sorter
    place_wine_at_rack_location
    push_buttons
    put_groceries_in_cupboard
    put_item_in_drawer
    put_money_in_safe
    reach_and_drag
    slide_block_to_color_target
    stack_blocks
    stack_cups
    sweep_to_dustpan_of_size
    turn_tap
)

# --- 実行処理 ---
# ログ用ディレクトリがなければ作成
mkdir -p "$LOG_DIR"

echo "RLBenchのデータ収集を開始します..."
echo "各タスクは個別のプロセスとしてバックグラウンドで実行されます。"
echo "ログは '${LOG_DIR}' ディレクトリに保存されます。"
echo "--------------------------------------------------"

# 各タスクをループしてバックグラウンドで実行
for TASK_NAME in "${TASKS[@]}"; do
    # 各タスク用のログファイル名を定義
    LOG_FILE="$LOG_DIR/log_${TASK_NAME}.txt"
    
    echo "タスク '${TASK_NAME}' を開始... ログファイル: ${LOG_FILE}"
    
    # nohupを使ってコマンドを実行し、ログをリダイレクトしてバックグラウンドへ
    # --processes=1 とし、各スクリプトはシングルプロセスで動作させる
    nohup xvfb-run --auto-servernum python tools/dataset_generator.py \
        --save_path="$SAVE_PATH" \
        --tasks="$TASK_NAME" \
        --episodes_per_task="$EPISODES_PER_TASK" \
        --image_size="$IMAGE_SIZE" \
        --renderer="$RENDERER" \
        --processes=1 > "$LOG_FILE" 2>&1 &
done

echo "--------------------------------------------------"
echo "全てのタスクの起動を試みました。"
echo "実行中のジョブは 'jobs' コマンドで確認できます。"
echo "各プロセスの実行状況やエラーは、'${LOG_DIR}' 内のログファイルを参照してください。"
echo "プロセスの詳細な状態は 'ps aux | grep dataset_generator.py' などで確認できます。"
