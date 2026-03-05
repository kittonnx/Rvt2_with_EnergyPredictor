import numpy as np
import pandas as pd
import os # osのインポートは必要なので残します

def export_filtered_data_to_csv(input_path: str, output_path: str, threshold: float):
    """
    .npzファイルからエネルギーが閾値以上のデータを抽出し、CSVファイルに保存する。
    """
    print(f"'{input_path}' からデータを読み込んでいます...")
    try:
        data = np.load(input_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{input_path}' が見つかりません。")
        return

    if 'energy' not in data:
        print(f"エラー: ファイル内に 'energy' というキーが見つかりません。")
        return

    # Step 1: NPZの全データをpandasのDataFrameに変換
    df = pd.DataFrame()
    for key in data.files:
        array = data[key]
        if array.ndim == 2 and array.shape[1] > 1:
            for i in range(array.shape[1]):
                df[f'{key}_{i}'] = array[:, i]
        else:
            df[key] = array.flatten()

    num_original = len(df)
    print(f"元のサンプル数: {num_original}")

    # Step 2: DataFrame上でフィルタリング
    filtered_df = df[df['energy'] >= threshold].copy()
    
    num_filtered = len(filtered_df)

    if num_filtered == 0:
        print(f"エネルギーが{threshold}以上のデータは見つかりませんでした。")
        return

    print(f"エネルギーが{threshold}以上のサンプル数: {num_filtered}")

    # Step 3: フィルタリング後のDataFrameをCSVに保存
    filtered_df.to_csv(output_path, index=False)
    print(f"フィルタリングされたデータが '{output_path}' に保存されました。")


# --- メインの実行ブロック ---
if __name__ == '__main__':
    # --- ここで値を直接編集してください ---
    INPUT_FILE_PATH = 'eneall.npz'
    OUTPUT_FILE_PATH = 'filtered_high_energy.csv'
    ENERGY_THRESHOLD = 500.0
    # -----------------------------------
    
    # 上で設定した固定値を使って関数を呼び出し
    export_filtered_data_to_csv(
        input_path=INPUT_FILE_PATH, 
        output_path=OUTPUT_FILE_PATH, 
        threshold=ENERGY_THRESHOLD
    )