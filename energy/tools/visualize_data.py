# analyze_data.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_data():
    """
    NPZファイルからデータを読み込み、エネルギー分布を分析・可視化する。
    """
    # --- 設定 ---
    # ★★★ あなたの.npzファイル名に書き換えてください ★★★
    NPZ_FILE_PATH = "low_200_energy.npz" 

    # --- 1. データの読み込み ---
    print(f"'{NPZ_FILE_PATH}' からデータを読み込んでいます...")
    if not os.path.exists(NPZ_FILE_PATH):
        print(f"エラー: データファイル '{NPZ_FILE_PATH}' が見つかりません。")
        return
        
    data = np.load(NPZ_FILE_PATH)
    
    # エネルギーとタスク名をPandas DataFrameに格納
    df = pd.DataFrame({
        'energy': data['energy'],
        'task_name': data['task_name']
    })
    print(f"合計 {len(df)} 件のデータを読み込みました。")
    print("-" * 30)

    # --- 2. データセット全体の分析 ---
    print("【1. データセット全体のエネルギー統計】")
    # .describe()で主要な統計量（平均、中央値、標準偏差など）を表示
    print(df['energy'].describe())
    print("-" * 30)

    # 全体のエネルギー分布をヒストグラムで可視化
    plt.figure(figsize=(15, 6))
    sns.histplot(df['energy'], kde=True, bins=100)
    plt.title('Overall Energy Distribution (Histogram)')
    plt.xlabel('Energy (J)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

    # --- 3. タスクごとの分析 ---
    print("\n【2. タスクごとのエネルギー統計】")
    # task_nameでグループ化し、各タスクの統計量を表示
    pd.set_option('display.max_rows', 50) # 表示行数を設定
    print(df.groupby('task_name')['energy'].describe())
    print("-" * 30)

    # タスクごとのエネルギー分布をボックスプロットで可視化
    plt.figure(figsize=(18, 8))
    sns.boxplot(x='task_name', y='energy', data=df)
    plt.title('Energy Distribution per Task (Box Plot)')
    plt.xlabel('Task Name')
    plt.ylabel('Energy (J)')
    plt.xticks(rotation=45, ha='right') # タスク名が重ならないように回転
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_data()