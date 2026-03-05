import os
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import json
import argparse
import glob

# --- 1. Dataset and DataModule (変更なし) ---
class TorqueDataset(Dataset):
    def __init__(self, features, labels, steps, lang_ids, num_lang_classes):
        self.features = features
        self.labels = labels
        self.steps = steps
        self.lang_ids = lang_ids
        self.num_lang_classes = num_lang_classes

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        lang_id = torch.tensor(self.lang_ids[idx], dtype=torch.long)
        lang_one_hot = F.one_hot(lang_id, num_classes=self.num_lang_classes).float()
        step = torch.tensor(self.steps[idx], dtype=torch.float32).unsqueeze(0)
        return (x, lang_one_hot, step), y

class EnergyDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data: np.lib.npyio.NpzFile, 
                 task_name: str, 
                 global_lang_to_id: dict, 
                 global_total_lang_classes: int, 
                 batch_size: int = 1024, 
                 val_split: float = 0.2):
        super().__init__()
        self.data = data 
        self.task_name = task_name
        self.global_lang_to_id = global_lang_to_id
        self.global_total_lang_classes = global_total_lang_classes
        self.batch_size = batch_size
        self.val_split = val_split

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        mask = self.data['task_name'] == self.task_name
        
        robot_state = self.data['robot_state'][mask]
        predicted_pose = np.concatenate([
            self.data['soft_predicted_position'][mask],
            self.data['soft_predicted_rotation_quat'][mask]
        ], axis=1)
        grip = self.data['soft_predicted_gripper_state'][mask].reshape(-1, 1).astype(np.float32)
        coll = self.data['soft_predicted_collision_state'][mask].reshape(-1, 1).astype(np.float32)
        
        # ★ (外観特徴ありの定義に戻す) ★
        proprio_feat = self.data['rvt_proprio_feat'][mask]
        if proprio_feat.ndim == 3: 
            proprio_feat = proprio_feat.squeeze(1)

        inputs = np.concatenate([
            robot_state, predicted_pose, grip, coll, proprio_feat
        ], axis=1)

        labels = self.data['energy'][mask].reshape(-1, 1)
        lang_goals = self.data['lang_goal'][mask]
        lang_ids = np.array([self.global_lang_to_id[name] for name in lang_goals])
        steps = self.data['step'][mask].astype(np.float32)
        steps = (steps - 1) / (25 - 1)
        
        train_inputs, val_inputs, train_labels, val_labels, train_steps, val_steps,train_lang_ids, val_lang_ids = train_test_split(
            inputs, labels, steps, lang_ids,
            test_size=self.val_split,
            random_state=42, # 学習時と完全一致させる
            stratify=lang_ids,
            shuffle=True
        )
        
        self.train_dataset = TorqueDataset(train_inputs, train_labels, train_steps, train_lang_ids, self.global_total_lang_classes)
        self.val_dataset = TorqueDataset(val_inputs, val_labels, val_steps, val_lang_ids, self.global_total_lang_classes)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

# --- 2. LightningModule (外観特徴ありの定義に戻す) ---
class EnergyPredictorPL(pl.LightningModule):
    def __init__(self, 
                 num_lang_classes: int, 
                 learning_rate: float = 1e-2,
                 dropout: float = 0.5):
        super().__init__()
        self.save_hyperparameters()

        # ★ (外観特徴ありの定義に戻す) ★
        BASE_FEATURES_DIM = 30
        PROPRIO_FEAT_DIM = 64 # 外観特徴の次元
        STEP_DIM = 1
        total_input_dim = BASE_FEATURES_DIM + PROPRIO_FEAT_DIM + num_lang_classes + STEP_DIM
        # (例: 30 + 64 + 217 + 1 = 312次元)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(total_input_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        features, lang_one_hot, step = x
        em_features = torch.cat([features, lang_one_hot, step], dim=1)
        return self.model(em_features)

# --- 3. 分析ロジック (結果を return するように変更) ---
def analyze_task(ckpt_path, task_name, data_module, id_to_lang, data_split, num_lang_classes):
    print(f"\n--- {task_name} ({data_split}データ) の分析開始 ---")
    print(f"使用CKPT: {ckpt_path}")

    # 1. データローダーを選択
    if data_split == 'train':
        dataloader_to_check = data_module.train_dataloader()
    else:
        dataloader_to_check = data_module.val_dataloader()

    # 2. モデルをロード
    try:
        model = EnergyPredictorPL.load_from_checkpoint(
            ckpt_path,
            num_lang_classes=num_lang_classes 
        )
    except Exception as e:
        print(f"エラー: {task_name} のCKPTロードに失敗しました。スキップします。")
        print(f"詳細: {e}")
        return None # <-- ★ (今回の変更点 1) Noneを返す ★

    # 3. 分析実行
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    results = []
    print("予測を実行中...")
    with torch.no_grad():
        for batch in dataloader_to_check:
            x, y = batch
            features, lang_one_hot, step = x
            
            device = model.device
            x_on_device = (features.to(device), lang_one_hot.to(device), step.to(device))
            y_on_device = y.to(device)

            y_hat = model(x_on_device)
            errors = torch.square(y_hat - y_on_device)
            lang_ids = torch.argmax(lang_one_hot, dim=1)

            errors_cpu = errors.cpu()
            y_hat_cpu = y_hat.cpu()
            y_cpu = y_on_device.cpu()
            lang_ids_cpu = lang_ids.cpu()
            step_cpu = step.cpu()

            for i in range(errors_cpu.shape[0]):
                # ★ (今回の変更点 2) 結果に lang_goal_str も追加 ★
                lang_id_item = lang_ids_cpu[i].item()
                lang_goal_str = id_to_lang.get(lang_id_item, "ID不明")
                results.append({
                    'mse': errors_cpu[i].item(),
                    'predicted_energy': y_hat_cpu[i].item(),
                    'actual_energy': y_cpu[i].item(),
                    'lang_id': lang_id_item,
                    'lang_goal': lang_goal_str, # <-- 追加
                    'step_normalized': step_cpu[i].item() 
                })

    # 4. ソートして表示
    sorted_results = sorted(results, key=lambda d: d['mse'], reverse=True)
    top_20_results = sorted_results[:20] # トップ20件を保持

    print(f"\n--- {task_name} ({data_split}データ): MSEワースト 20 サンプル ---")
    print(f"{'MSE':>12} | {'予測値':>12} | {'実測値':>12} | {'Step (正規化)':>15} | 言語指示 (lang_goal)")
    print("-" * 85)
    
    for res in top_20_results:
        original_step = (res['step_normalized'] * (25 - 1)) + 1
        print(f"{res['mse']:12.4f} | {res['predicted_energy']:12.4f} | {res['actual_energy']:12.4f} | {res['step_normalized']:15.4f} ({original_step:2.0f}) | {res['lang_goal']}")
    
    print("-" * 85)
    
    # ★ (今回の変更点 3) トップ20件の結果を返す ★
    return top_20_results 


# --- 4. main関数 (全タスク実行 ＋ JSON保存) ---
def main(args):
    print("--- 全タスク自動分析プログラム開始 ---")
    
    # 1. グローバルマップをロード
    print(f"読み込み中: {args.map_file}")
    with open(args.map_file, 'r', encoding='utf-8') as f:
        all_task_mappings = json.load(f)
    
    global_total_lang_classes = all_task_mappings['total_lang_classes']
    global_lang_to_id = all_task_mappings['lang_to_id']
    id_to_lang = {v: k for k, v in global_lang_to_id.items()}

    # 2. NPZデータをロード
    print(f"読み込み中: {args.npz_file}")
    with np.load(args.npz_file) as data:
        
        # 3. チェックポイントフォルダをスキャン
        # ★ (今回の変更点 4) 検索パターンを 'appearance-model-*.ckpt' に変更 ★
        ckpt_search_path = os.path.join(args.checkpoints_dir, "appearance-model-*.ckpt")
        all_ckpt_files = glob.glob(ckpt_search_path)
        
        if not all_ckpt_files:
            print(f"エラー: '{ckpt_search_path}' に一致するCKPTファイルが見つかりません。")
            return
            
        print(f"発見したCKPTファイル ( {len(all_ckpt_files)} 件):")

        # ★ (今回の変更点 5) 最終結果を保存する辞書を初期化 ★
        all_tasks_results = {}
        
        # 4. 見つかったCKPTごとにループ
        for ckpt_path in all_ckpt_files:
            filename = os.path.basename(ckpt_path)
            try:
                # ★ (今回の変更点 6) タスク名抽出ロジックを変更 ★
                # "appearance-model-" (17文字) の後から、次の "-" までがタスク名
                task_name_part = filename[17:] 
                task_name = task_name_part.split('-')[0] 
            except IndexError:
                print(f"警告: '{filename}' からタスク名を抽出できませんでした。スキップします。")
                continue
            
            # 5. DataModuleを準備
            data_module = EnergyDataModule(
                data=data, 
                task_name=task_name, 
                global_lang_to_id=global_lang_to_id,
                global_total_lang_classes=global_total_lang_classes,
                batch_size=args.batch_size, 
                val_split=0.2
            )
            data_module.setup()

            # 6. 分析関数を呼び出し、結果を受け取る
            worst_samples = analyze_task(
                ckpt_path=ckpt_path,
                task_name=task_name,
                data_module=data_module,
                id_to_lang=id_to_lang,
                data_split=args.data_split,
                num_lang_classes=global_total_lang_classes
            )
            
            # ★ (今回の変更点 7) 結果を辞書に保存 ★
            if worst_samples is not None:
                all_tasks_results[task_name] = worst_samples

    # --- 7. ★ (今回の変更点 8) 全タスクの結果を単一のJSONファイルに保存 ★ ---
    output_filename = os.path.join(args.checkpoints_dir, f"worst_mse_analysis_{args.data_split}.json")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_tasks_results, f, ensure_ascii=False, indent=4)
        print(f"\n{'='*50}")
        print(f"✅ 全タスクの分析結果を保存しました: {output_filename}")
        print(f"{'='*50}")
    except Exception as e:
        print(f"\nエラー: 最終JSONファイルの保存に失敗しました。")
        print(f"詳細: {e}")
    # ----------------------------------------------------------------

    print("\n--- 全タスクの分析が完了しました ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="【全タスク自動】学習済みCKPTをロードして、MSEワーストデータを分析します。")
    
    parser.add_argument("--data_split", type=str, default="val", 
                        choices=['val', 'train'],
                        help="分析するデータセット ( 'val' または 'train' )")
    
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="CKPTファイルが保存されているディレクトリ")

    parser.add_argument("--npz_file", type=str, default="soft_lang711.npz", 
                        help="データソースのNPZファイル")
    
    parser.add_argument("--map_file", type=str, default="checkpoints/lang_map_all_tasks.json",
                        help="グローバル言語マップのJSONファイル")
    
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="分析時のバッチサイズ（大きいほど速い）")
    
    args = parser.parse_args()
    
    main(args)