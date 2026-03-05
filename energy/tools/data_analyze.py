import numpy as np
import pandas as pd
import os

def convert_npz_to_csv(input_path: str, output_path: str):
    """
    .npzファイル内の全てのデータをCSVファイルに変換して保存する。
    """
    print(f"'{input_path}' からデータを読み込んでいます...")
    try:
        data = np.load(input_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{input_path}' が見つかりません。")
        return

    print("データをDataFrameに変換しています...")
    df = pd.DataFrame()
    
    # .npzファイル内の各配列をループ処理
    for key in data.files:
        array = data[key]
        
        # キーが 'rvt_lang_feat' で、3次元配列の場合の特別処理
        if 'lang_feat' in key and array.ndim == 3:
            print(f"キー '{key}' (shape: {array.shape}) の系列方向(axis=1)の平均を計算しています...")
            # 系列方向(axis=1)で平均を取り、(N, 128) のような2次元配列に要約
            summary_array = np.mean(array, axis=1)
            
            # 要約した2次元配列を 'key_0', 'key_1', ... のように別々の列として追加
            for i in range(summary_array.shape[1]):
                df[f'{key}_mean_{i}'] = summary_array[:, i]

        # 配列が2次元で、複数の列を持つ場合 (例: (N, 21))
        elif array.ndim == 2 and array.shape[1] > 1:
            # 各次元を 'key_0', 'key_1', ... のように別々の列として追加
            for i in range(array.shape[1]):
                df[f'{key}_{i}'] = array[:, i]
                
        # 配列が1次元の場合 (例: (N,) or (N, 1))
        else:
            # そのまま1つの列として追加
            df[key] = array.flatten()

    # DataFrameをCSVファイルに保存
    # index=False は、DataFrameのインデックス(0, 1, 2...)をファイルに書き出さない設定
    df.to_csv(output_path, index=False)
    print(f"データが '{output_path}' に正常に保存されました。")


# --- メインの実行ブロック ---
if __name__ == '__main__':
    # --- ここでファイル名を直接編集してください ---
    # INPUT_FILE_PATH = "../rvt/runs/rvt2/eval/testguy/model_99/task7.npz"  
    INPUT_FILE_PATH = "./fused7.npz"  
    OUTPUT_FILE_PATH = 'output_all_data.csv'
    # ------------------------------------------
    
    convert_npz_to_csv(
        input_path=INPUT_FILE_PATH, 
        output_path=OUTPUT_FILE_PATH
    )