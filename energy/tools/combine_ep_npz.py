import numpy as np
import argparse
from collections import defaultdict

def definitive_combine_and_sort_FIXED(input_files, output_path):
    """
    並べ替えロジックの致命的な誤りを修正した、最終確定版プログラム。
    """
    # ステップ1: タスクごとにデータを蓄積し、エピソードの最大値を管理する
    print("--- ステップ1: ファイルを一つずつ処理し、タスク毎にデータを振り分け、エピソードを再採番 ---")
    
    tasks_data_pool = defaultdict(lambda: {'data': defaultdict(list), 'last_episode': 0})

    for file_path in sorted(input_files):
        try:
            print(f"  - 処理中: {file_path}")
            with np.load(file_path, allow_pickle=True) as data:
                tasks_in_this_file = np.unique(data['task_name'])

                for task in tasks_in_this_file:
                    offset = tasks_data_pool[task]['last_episode']
                    mask = (data['task_name'] == task)
                    current_chunk = {key: arr[mask] for key, arr in data.items()}
                    
                    original_episodes = current_chunk['episode']
                    new_episodes = original_episodes + offset
                    current_chunk['episode'] = new_episodes
                    
                    if new_episodes.size > 0:
                        tasks_data_pool[task]['last_episode'] = np.max(new_episodes)
                    
                    for key, arr in current_chunk.items():
                        tasks_data_pool[task]['data'][key].append(arr)

        except Exception as e:
            print(f"警告: '{file_path}' の処理中にエラー: {e}。スキップします。")

    if not tasks_data_pool:
        print("エラー: 有効なデータが読み込めませんでした。")
        return

    # ステップ2: 全てのタスクのデータを一つに結合する
    print("\n--- ステップ2: 全タスクのデータを一つの巨大なデータセットに結合 ---")
    
    final_combined_data = defaultdict(list)
    for task_name in sorted(tasks_data_pool.keys()):
        task_content = tasks_data_pool[task_name]
        for key, list_of_arrays in task_content['data'].items():
            concatenated_array = np.concatenate(list_of_arrays, axis=0)
            final_combined_data[key].append(concatenated_array)

    for key, list_of_arrays in final_combined_data.items():
        final_combined_data[key] = np.concatenate(list_of_arrays, axis=0)

    # ステップ3: 「タスク名、エピソード番号、ステップ」で全体を並べ替える
    print("\n--- ステップ3: 全データを最終的に並べ替え ---")

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    #  致命的な誤りを修正: np.lexsortは優先度が【低い】キーから順に指定する
    #  優先順位: ①タスク名 > ②エピソード番号 > ③ステップ番号
    #  したがって、指定する順序は (ステップ、エピソード、タスク名) となる
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    sort_keys = (
        final_combined_data['step'],
        final_combined_data['episode'],
        final_combined_data['task_name']
    )
    sorted_indices = np.lexsort(sort_keys)
    
    for key in final_combined_data:
        final_combined_data[key] = final_combined_data[key][sorted_indices]

    print("並べ替え完了。")

    # ステップ4: 最終的な.npzファイルとして保存する
    print(f"\n--- ステップ4: 最終データを '{output_path}' に保存 ---")
    np.savez_compressed(output_path, **final_combined_data)
    
    print("\n✅ 全ての処理が完了しました。")


def main():
    parser = argparse.ArgumentParser(description="【ソート修正版】NPZファイルを仕様通りに結合・再採番・ソートします。")
    parser.add_argument("input_files", nargs='+', help=".npzファイルのパス（複数指定可）。")
    parser.add_argument("-o", "--output", default="dataset_combined_sorted.npz", help="出力ファイル名。")
    args = parser.parse_args()
    definitive_combine_and_sort_FIXED(args.input_files, args.output)

if __name__ == "__main__":
    main()