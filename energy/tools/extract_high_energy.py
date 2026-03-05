# extract_outliers.py (修正版)

import numpy as np
import pandas as pd
import os

def extract_high_energy_data():
    """
    NPZファイルからデータを読み込み、指定した閾値を超えるエネルギーを持つ
    データポイントを抽出し、CSVファイルとして保存する。
    """
    # --- 設定 ---
    NPZ_FILE_PATH = "task7.npz"
    ENERGY_THRESHOLD = 200.0
    OUTPUT_CSV_PATH = "high_energy_outliers.csv"

    # --- 1. データの読み込み ---
    print(f"'{NPZ_FILE_PATH}' からデータを読み込んでいます...")
    if not os.path.exists(NPZ_FILE_PATH):
        print(f"エラー: データファイル '{NPZ_FILE_PATH}' が見つかりません。")
        return
        
    data = np.load(NPZ_FILE_PATH)
    
    # --- 2. 外れ値の抽出 ---
    energy_values = data['energy']
    outlier_indices = np.where(energy_values > ENERGY_THRESHOLD)[0]
    num_outliers = len(outlier_indices)
    total_data = len(energy_values)
    
    if num_outliers == 0:
        print(f"{ENERGY_THRESHOLD}Jを超えるデータは見つかりませんでした。")
        return
        
    print(f"全 {total_data} 件中、{ENERGY_THRESHOLD}Jを超えるデータが {num_outliers} 件見つかりました。")

    # --- 3. 外れ値データをDataFrameに格納 ---
    outlier_data = {}
    
    # --- ▼▼▼ここから変更▼▼▼ ---
    # .npzファイルに含まれる全てのキーをループ
    for key in data.files:
        # 該当するインデックスのデータだけを抽出
        column_data = data[key][outlier_indices]
        
        # データが2次元以上（例: robot_state）の場合、各列を分解して追加
        if column_data.ndim > 1:
            for i in range(column_data.shape[1]):
                # 新しい列名を作成 (例: robot_state_0, robot_state_1, ...)
                new_key = f"{key}_{i}"
                outlier_data[new_key] = column_data[:, i]
        else:
            # 1次元のデータはそのまま追加
            outlier_data[key] = column_data
    # --- ▲▲▲ここまで変更▲▲▲ ---

    df_outliers = pd.DataFrame(outlier_data)
    df_outliers = df_outliers.sort_values(by='energy', ascending=False)

    # --- 4. CSVファイルとして保存 ---
    try:
        df_outliers.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"外れ値データを '{OUTPUT_CSV_PATH}' として保存しました。")
    except Exception as e:
        print(f"CSVファイルの保存中にエラーが発生しました: {e}")

if __name__ == "__main__":
    extract_high_energy_data()