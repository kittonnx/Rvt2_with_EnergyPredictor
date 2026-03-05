import numpy as np

def inspect_npz_file(file_path: str):
    """
    .npzファイルを開き、格納されている各配列の
    キー、次元数(shape)、データ型(dtype)を表示する。
    """
    print(f"--- '{file_path}' の内容 ---")
    try:
        # allow_pickle=Trueは文字列などが格納されている場合に必要
        with np.load(file_path, allow_pickle=True) as data:
            if not data.files:
                print("ファイル内にデータが見つかりませんでした。")
                return

            print(f"{'キー名':<30} | {'次元数 (shape)':<20} | {'データ型 (dtype)'}")
            print("-" * 70)
            
            for key in data.files:
                array = data[key]
                # 整形して表示
                print(f"{key:<30} | {str(array.shape):<20} | {array.dtype}")

    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        print("--- 確認終了 ---")


# --- メインの実行ブロック ---
if __name__ == '__main__':
    # --- ここに確認したい.npzファイルのパスを記入してください ---
    INPUT_FILE_PATH = "./energy_dataset.npz"
    # ---------------------------------------------------------
    
    inspect_npz_file(INPUT_FILE_PATH)