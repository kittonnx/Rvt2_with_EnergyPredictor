import pandas as pd
import io
import sys

CSV_FILE_PATH = '../rvt/runs/rvt2/eval/finetune_notsche_lambda_1.0_early_5e-4/model_156/tani_log.csv'

# 結果をファイルに保存したい場合は、ここにファイル名を指定してください。
# 保存しない場合は None のままにしてください。
OUTPUT_FILE_PATH = None # 例: 'analysis_output.csv'
# ----------------

def analyze_task_results(file_path, output_file=None):
    """
    指定されたCSVファイルを読み込み、タスクごとの統計情報を計算し、
    Excelに貼り付けやすいカンマ区切り形式で表示またはファイルに保存する。
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。", file=sys.stderr)
        print("スクリプト内の CSV_FILE_PATH を正しいファイル名に変更してください。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中に問題が発生しました。ファイルが破損していませんか？", file=sys.stderr)
        print(f"詳細: {e}", file=sys.stderr)
        sys.exit(1)

    # 成功フラグを作成 (rewardが100.0ならTrue)
    df['success'] = (df['reward'] == 100.0)

    # タスクごとにグループ化
    grouped = df.groupby('task_name')

    # 各統計量を計算
    success_rate = grouped['success'].mean() * 100
    avg_energy_all = grouped['energy'].mean()
    
    # 成功エピソードのみの平均エネルギー
    # 成功エピソードがない場合はNaNになる
    avg_energy_success_only = df[df['success']].groupby('task_name')['energy'].mean()

    # 結果を一つのDataFrameにまとめる
    results_df = pd.DataFrame({
        'Success Rate (%)': success_rate,
        'Avg Energy (All Episodes)': avg_energy_all,
        'Avg Energy (Success Only)': avg_energy_success_only
    })

    # NaN (成功0回) の場合の表示を 'N/A' に変更
    # Excelに貼り付けることを考慮し、文字列ではなく空欄（NaN）のままにするか、
    # 特定の文字列（例: 'N/A'）にするかは検討が必要。今回は「N/A (Success=0)」に
    # NaNをそのまま出力したい場合はこの行をコメントアウトしてください
    results_df['Avg Energy (Success Only)'] = results_df['Avg Energy (Success Only)'].fillna('N/A (Success=0)')
    
    # 浮動小数点数の表示精度を設定
    pd.options.display.float_format = '{:.2f}'.format

    # 結果の出力
    if output_file:
        try:
            results_df.to_csv(output_file, index=True) # index=Trueでtask_nameも出力
            print(f"分析結果を '{output_file}' に保存しました。", file=sys.stderr)
        except Exception as e:
            print(f"エラー: 結果をファイル '{output_file}' に保存できませんでした。", file=sys.stderr)
            print(f"詳細: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # 標準出力にカンマ区切りで表示（Excelに貼り付けやすい形式）
        # to_csv の StringIO を使うことで、PandasのCSV出力機能を活用しつつ標準出力に流す
        output_buffer = io.StringIO()
        results_df.to_csv(output_buffer, index=True) # index=Trueでtask_name（インデックス）も出力
        print(output_buffer.getvalue())

if __name__ == '__main__':
    analyze_task_results(CSV_FILE_PATH, OUTPUT_FILE_PATH)