# find_video_episode_number.py

import numpy as np
import pandas as pd
import os

def find_video_episode_numbers():
    """
    .npzファイルから全成功エピソードのリストを作成し、
    外れ値CSVの各データが、動画ファイルとして何番目に相当するかを特定する。
    """
    # --- 設定 ---
    # ★★★ あなたの.npzファイル名に書き換えてください ★★★
    NPZ_FILE_PATH = "task7.npz"
    
    # ★★★ 前のステップで生成した外れ値CSVファイル名 ★★★
    OUTLIER_CSV_PATH = "high_energy_outliers.csv"
    
    # ★★★ 最終的な出力ファイル名 ★★★
    OUTPUT_CSV_PATH = "outliers_with_video_episodes.csv"

    # --- 1. 必要なファイルの存在確認 ---
    if not os.path.exists(NPZ_FILE_PATH):
        print(f"エラー: データファイル '{NPZ_FILE_PATH}' が見つかりません。")
        return
    if not os.path.exists(OUTLIER_CSV_PATH):
        print(f"エラー: 外れ値ファイル '{OUTLIER_CSV_PATH}' が見つかりません。")
        return

    # --- 2. NPZファイルから「動画番号の対応表」を作成 ---
    print(f"'{NPZ_FILE_PATH}' を読み込み、動画番号の対応表を作成しています...")
    data_npz = np.load(NPZ_FILE_PATH)
    
    # タスク名と元のエピソード番号だけを持つDataFrameを作成
    df_all_episodes = pd.DataFrame({
        'task_name': data_npz['task_name'],
        'episode': data_npz['episode']
    })

    # 重複を削除し、ユニークなエピソードのリストを取得
    df_unique_episodes = df_all_episodes.drop_duplicates().copy()

    # タスク名、元のエピソード番号の順でソート
    df_unique_episodes = df_unique_episodes.sort_values(by=['task_name', 'episode'])

    # タスクごとにグループ化し、1から始まる連番（動画番号）を振る
    # cumcount()は0から始まるので、+1する
    df_unique_episodes['video_episode_number'] = df_unique_episodes.groupby('task_name').cumcount() + 1
    
    print("対応表が完成しました。")
    # print(df_unique_episodes.head()) # 対応表の先頭をプレビュー

    # --- 3. 外れ値CSVを読み込み、対応表をマージ ---
    print(f"'{OUTLIER_CSV_PATH}' を読み込んでいます...")
    df_outliers = pd.read_csv(OUTLIER_CSV_PATH)
    
    print("外れ値データに動画番号を付与しています...")
    # 'task_name'と'episode'をキーにして、外れ値データに対応表を結合(マージ)する
    df_merged = pd.merge(
        df_outliers,
        df_unique_episodes,
        on=['task_name', 'episode'],
        how='left' # outliersデータは全て残す
    )
    
    # --- 4. 結果を新しいCSVファイルとして保存 ---
    try:
        # video_episode_number列をenergy列の隣に移動させて見やすくする
        cols = df_merged.columns.tolist()
        cols.insert(cols.index('energy') + 1, cols.pop(cols.index('video_episode_number')))
        df_merged = df_merged[cols]

        df_merged.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"動画番号を付与した結果を '{OUTPUT_CSV_PATH}' として保存しました。")
    except Exception as e:
        print(f"CSVファイルの保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    find_video_episode_numbers()