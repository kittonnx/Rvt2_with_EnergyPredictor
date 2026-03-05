# copy_outlier_videos_direct.py

import os
import shutil
import pandas as pd
import io

def copy_specific_videos_directly():
    """
    指定されたタスク名とエピソード番号を「そのまま」使い、
    対応する動画ファイルを新しいフォルダにコピーする。
    """
    # --- ▼▼▼ 設定：あなたの環境に合わせて3箇所を修正してください ▼▼▼ ---

    # 1. 調査したいタスク名とエピソード番号のリスト
    OUTLIER_DATA = """
task_name,episode
turn_tap,280
light_bulb_in,406
light_bulb_in,229
light_bulb_in,256
place_shape_in_shape_sorter,151
light_bulb_in,414
place_cups,4
light_bulb_in,258
place_cups,136
place_cups,132
place_cups,25
light_bulb_in,268
light_bulb_in,277
place_cups,32
light_bulb_in,325
light_bulb_in,67
light_bulb_in,278
light_bulb_in,301
light_bulb_in,389
slide_block_to_color_target,372
slide_block_to_color_target,154
light_bulb_in,227
light_bulb_in,126
place_cups,100
light_bulb_in,422
light_bulb_in,42
place_cups,122
place_cups,76
light_bulb_in,267
light_bulb_in,208
light_bulb_in,122
place_cups,17
place_cups,79
place_cups,35
light_bulb_in,199
place_shape_in_shape_sorter,158
place_cups,131
place_cups,31
light_bulb_in,156
place_cups,36
place_cups,24
place_cups,85
slide_block_to_color_target,333
place_cups,23
place_cups,127
place_cups,103
place_cups,41
place_cups,134
place_cups,22
place_shape_in_shape_sorter,151
place_shape_in_shape_sorter,61
light_bulb_in,342
place_cups,92
light_bulb_in,245
light_bulb_in,79
place_cups,94
place_cups,124
put_groceries_in_cupboard,26
light_bulb_in,439
light_bulb_in,119
put_groceries_in_cupboard,104
light_bulb_in,88
place_cups,18
place_cups,120
light_bulb_in,155
place_cups,111
place_cups,145
place_shape_in_shape_sorter,69
light_bulb_in,110
place_cups,86
place_cups,143
insert_onto_square_peg,153
light_bulb_in,142
light_bulb_in,222
slide_block_to_color_target,134
light_bulb_in,221
place_cups,51
place_cups,131
place_cups,107
insert_onto_square_peg,84
turn_tap,190
place_cups,112
place_cups,75
place_cups,142
slide_block_to_color_target,242
place_cups,132
place_cups,77
light_bulb_in,242
light_bulb_in,178
place_cups,54
light_bulb_in,108
place_cups,95
place_cups,87
place_cups,89
place_cups,115
place_cups,80
put_groceries_in_cupboard,164
light_bulb_in,303
light_bulb_in,22
place_cups,66
place_cups,47
place_cups,70
place_cups,130
place_cups,140
place_cups,45
place_cups,91
place_shape_in_shape_sorter,168
place_cups,146
place_cups,29
place_cups,138
insert_onto_square_peg,7
put_groceries_in_cupboard,41
place_shape_in_shape_sorter,107
place_cups,153
place_shape_in_shape_sorter,184
place_cups,146
place_cups,74
light_bulb_in,413
place_cups,20
place_cups,50
place_cups,94
place_cups,15
place_cups,61
put_groceries_in_cupboard,3
place_cups,106
put_groceries_in_cupboard,192
insert_onto_square_peg,146
light_bulb_in,443
place_cups,49
place_cups,115
place_cups,21
place_cups,150
place_cups,126
place_shape_in_shape_sorter,134
meat_off_grill,294
turn_tap,401
place_cups,105
place_cups,114
place_cups,63
place_cups,55
place_cups,83
    """

    # 2. ★★★ 元となる動画ファイルが保存されているフォルダのパス ★★★
    SOURCE_VIDEO_DIR = "/misc/dl00/tani/energy_rvt/RVT/rvt/runs/rvt2/eval/name_task7/model_99/videos" # 例: "/home/user/rlbench_data/videos"

    # 3. ★★★ 抽出した動画を保存する新しいフォルダの名前 ★★★
    DESTINATION_DIR = "./copy_videos"

    # --- ▲▲▲ 設定はここまで ▲▲▲ ---
    # --- 2. 調査対象のリストをDataFrameに読み込み ---
    outlier_df = pd.read_csv(io.StringIO(OUTLIER_DATA))
    outlier_df['episode'] = pd.to_numeric(outlier_df['episode'])

    # --- 3. ファイルのコピー処理 ---
    print(f"\n'{DESTINATION_DIR}' フォルダに動画をコピーします...")
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    
    copied_count = 0
    not_found_count = 0
    
    for index, row in outlier_df.iterrows():
        task_name = row['task_name']
        episode_num = row['episode'] # ★元のエピソード番号をそのまま使用
        
        # ファイル名がNaNなどの場合はスキップ
        if pd.isna(episode_num):
            continue
            
        episode_num = int(episode_num)
        
        # ★ファイル名を元のエピソード番号で構築
        source_filename = f"{task_name}_success_{episode_num-1}.mp4"
        source_path = os.path.join(SOURCE_VIDEO_DIR, source_filename)
        destination_path = os.path.join(DESTINATION_DIR, source_filename)
        
        if os.path.exists(source_path):
            print(f"コピー中: {source_filename}")
            shutil.copy2(source_path, destination_path)
            copied_count += 1
        else:
            print(f"警告: ファイルが見つかりません: {source_path}")
            not_found_count += 1
            
    print("\n--- 処理完了 ---")
    print(f"コピーされたファイル数: {copied_count}")
    print(f"見つからなかったファイル数: {not_found_count}")
    print(f"動画は '{DESTINATION_DIR}' フォルダに保存されました。")


if __name__ == "__main__":
    copy_specific_videos_directly()