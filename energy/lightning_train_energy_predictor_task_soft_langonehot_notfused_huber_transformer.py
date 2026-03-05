import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from schedulefree import RAdamScheduleFree
from sklearn.model_selection import train_test_split
import yaml

# --- 1. Dataset and DataModule (変更なし) ---
class TorqueDataset(Dataset):
    def __init__(self, features, labels, steps, energy_mean, energy_std):
        self.features = features
        self.labels = labels
        self.steps = steps
        self.energy_mean = energy_mean
        self.energy_std = energy_std

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor((self.labels[idx] - self.energy_mean) / self.energy_std, dtype=torch.float32)
        step = torch.tensor(self.steps[idx], dtype=torch.float32).unsqueeze(0)
        return (x, step), y

class EnergyDataModule(pl.LightningDataModule):
    def __init__(self, data, task_name: str, batch_size: int = 1024, val_split: float = 0.2):
        super().__init__()
        self.data = data
        self.task_name = task_name
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage=None):
        data = self.data
        mask = data['task_name'] == self.task_name
        
        # --- 重要: 入力データの構築 ---
        # 1. Robot State (22次元と仮定: Joint pos/vel, Gripper pose等)
        robot_state = data['robot_state'][mask] 
        
        # 2. Predicted Action (9次元: Pos(3) + Rot(4) + Grip(1) + Coll(1))
        predicted_pose = np.concatenate([
            data['soft_predicted_position'][mask],
            data['soft_predicted_rotation_quat'][mask]
        ], axis=1)
        grip = data['soft_predicted_gripper_state'][mask].reshape(-1, 1).astype(np.float32)
        coll = data['soft_predicted_collision_state'][mask].reshape(-1, 1).astype(np.float32)
        
        # 結合: State(22) + Action(9) = Total(31)
        inputs = np.concatenate([robot_state, predicted_pose, grip, coll], axis=1)
        
        labels = data['energy'][mask].reshape(-1, 1)

        steps = data['step'][mask].astype(np.float32)
        steps = (steps - 1) / (25 - 1) # 正規化
        
        train_inputs, val_inputs, train_labels, val_labels, train_steps, val_steps = train_test_split(
            inputs, labels, steps, test_size=self.val_split, random_state=42, shuffle=True
        )

        self.energy_mean = np.mean(train_labels)
        self.energy_std = np.std(train_labels) + 1e-8
        self.energy_min = np.min(train_labels)
        self.energy_max = np.max(train_labels)

        self.train_dataset = TorqueDataset(train_inputs, train_labels, train_steps, self.energy_mean, self.energy_std)
        self.val_dataset = TorqueDataset(val_inputs, val_labels, val_steps, self.energy_mean, self.energy_std)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

# --- 2. Transformer Model (ここを変更) ---
class EnergyPredictorTransformer(pl.LightningModule):
    def __init__(
        self, 
        state_dim: int = 22,   # ロボット状態の次元
        action_dim: int = 9,   # アクションの次元
        d_model: int = 128,    # 内部の埋め込み次元
        nhead: int = 4,        # ヘッド数
        num_layers: int = 3,   # Transformer層の数
        dim_feedforward: int = 256,
        dropout: float = 0.1, 
        learning_rate: float = 1e-3,
        energy_mean: float = 0.0, 
        energy_std: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.energy_mean = energy_mean
        self.energy_std = energy_std
        
        # --- Tokenizer: 数値を「意味ベクトル」に変換 ---
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.step_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # CLSトークン: 全体の情報を集約するための学習可能なパラメータ
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=False,
            norm_first=True, # Pre-LN (安定学習のため)
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Head ---
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.criterion = nn.HuberLoss()

    def forward(self, x):

        features, step_val = x
        batch_size = features.size(0)
        
        current_state = features[:, :self.hparams.state_dim]
        next_action   = features[:, self.hparams.state_dim:]
        
        # Embed -> (Batch, 1, Dim)
        t_state  = self.state_embed(current_state).unsqueeze(1)
        t_action = self.action_embed(next_action).unsqueeze(1)
        t_step   = self.step_embed(step_val).unsqueeze(1)
        t_cls    = self.cls_token.expand(batch_size, -1, -1)
        
        # Sequence化 (Batch, Seq, Feature)
        src = torch.cat((t_cls, t_state, t_action, t_step), dim=1)
        
        # 修正: (Batch, Seq, Dim) -> (Seq, Batch, Dim) に入れ替え
        # これによりメモリ配置が変わり、クラッシュを回避できます
        src = src.permute(1, 0, 2).contiguous()
        
        # Transformerに入力 (Seq, Batch, Dim)
        out = self.transformer(src)
        
        # 出力も (Seq, Batch, Dim) なので、先頭の [CLS] トークン (index 0) を取得
        # out[0] は (Batch, Dim) になります
        cls_output = out[0]
        
        return self.head(cls_output)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_huber_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        y_unscaled = (y * self.energy_std) + self.energy_mean
        y_hat_unscaled = (y_hat * self.energy_std) + self.energy_mean
        self.log('val_mae', F.l1_loss(y_hat_unscaled, y_unscaled), on_epoch=True)
        return loss

    def configure_optimizers(self):
        # TransformerにはScheduleFreeが非常に適しています
        return {'optimizer': RAdamScheduleFree(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)}
    
    def on_train_start(self):
        self.trainer.optimizers[0].train()

# --- 3. Main Function ---
def main():
    # --- 設定 ---
    # NPZ_FILE_PATH = 'fused7111216.npz'
    NPZ_FILE_PATH = 'fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz'
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    EPOCHS = 2000 # Transformerは収束が少し早い場合が多いですが念のため多めに
    DROPOUT = 0.1 # TransformerはMLPよりDropout低めが一般的
    LAYERS = 3
    
    # 構造の定義 (記録用)
    architecture = 'Transformer-d128-nhead4-lyr3'
    # PROJECT_NAME = f'TF_NoFused_EnergyPredictor_nohdbscan'
    PROJECT_NAME = f'TF_NoFused_EnergyPredictor_for_test'
    # PROJECT_NAME = f'TF_NoFused_EnergyPredictor'
    GROUP_NAME = f'{architecture}_lr{LEARNING_RATE}_dr{DROPOUT}_{NPZ_FILE_PATH}_numlayer-{LAYERS}'

    map_dir = 'checkpoints'
    os.makedirs(map_dir, exist_ok=True)
    
    # 次元数の定義 (重要！)
    STATE_DIM = 21 # データセットに合わせて調整してください (robot_stateの列数)
    ACTION_DIM = 9 # predicted_pose(7) + grip(1) + coll(1) = 9
    INPUT_DIM = STATE_DIM + ACTION_DIM

    task_configs = {}

    with np.load(NPZ_FILE_PATH) as data:
        all_task_names = np.unique(data['task_name'])
        print(f"--- Transformerモデルで合計 {len(all_task_names)} タスクの学習を開始します ---")
        
        # データセット内の実際の次元数を確認（念のため）
        actual_state_dim = data['robot_state'].shape[1]
        if actual_state_dim != STATE_DIM:
            print(f"Warning: 指定されたSTATE_DIM({STATE_DIM})と実際のデータの次元({actual_state_dim})が異なります。実際の値を使用します。")
            STATE_DIM = actual_state_dim

        for task_name_np in all_task_names:
            task_name = str(task_name_np)
            print(f"=== Task: {task_name} ===")
            
            config = {
                'data_name': NPZ_FILE_PATH,
                'learning_rate': LEARNING_RATE,
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'architecture': architecture,
                'dropout': DROPOUT,
                'task_name': task_name,
                'state_dim': STATE_DIM,
                'action_dim': ACTION_DIM
            }
            
            wandb_logger = WandbLogger(project=PROJECT_NAME, group=GROUP_NAME, config=config, name=task_name)
            
            checkpoint_dir = os.path.join(map_dir, PROJECT_NAME, GROUP_NAME)
            checkpoint_callback = ModelCheckpoint(
                monitor='val_huber_loss',
                dirpath=checkpoint_dir,
                filename=f'{task_name}-{{epoch:02d}}-{{val_huber_loss:.4f}}-numlayer-{LAYERS}',
                save_top_k=1,
                mode='min'
            )
            early_stop_callback = EarlyStopping(monitor='val_huber_loss', patience=300, verbose=True, mode='min')

            # DataModuleの準備
            data_module = EnergyDataModule(data=data, task_name=task_name, batch_size=BATCH_SIZE)
            data_module.setup()

            # Transformerモデルの構築
            model = EnergyPredictorTransformer(
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                d_model=128,
                nhead=4,
                num_layers=LAYERS,
                dropout=DROPOUT,
                learning_rate=LEARNING_RATE,
                energy_mean=data_module.energy_mean,
                energy_std=data_module.energy_std
            )

            trainer = pl.Trainer(
                max_epochs=EPOCHS,
                logger=wandb_logger,
                accelerator='auto',
                callbacks=[early_stop_callback, checkpoint_callback],
                log_every_n_steps=10
            )

            trainer.fit(model, datamodule=data_module)
            wandb_logger.experiment.finish()

            # コンフィグ保存用の情報収集
            best_path = checkpoint_callback.best_model_path
            task_configs[task_name] = {
                'enabled': True,
                'lambda': 1.0,
                'ckpt_path': best_path,
                'normalization': {
                    'mean': float(data_module.energy_mean),
                    'std': float(data_module.energy_std),
                    'min': float(data_module.energy_min),
                    'max': float(data_module.energy_max)
                },
                'model_params': { # ロード時に必要になる情報
                    'state_dim': STATE_DIM,
                    'action_dim': ACTION_DIM
                }
            }

    # 最終的なYAML保存
    energy_config = {
        'model_type': 'Transformer', # 識別用
        'model_shared_config': {
            'd_model': 128,
            'dropout': DROPOUT
        },
        'task_configs': task_configs
    }
    
    yaml_output_path = os.path.join(map_dir, PROJECT_NAME, GROUP_NAME, 'energy_config.yaml')
    with open(yaml_output_path, 'w') as f:
        yaml.dump(energy_config, f, sort_keys=False, indent=2, default_flow_style=False)
    
    print(f"全学習完了。設定ファイルを保存しました: {yaml_output_path}")

if __name__ == '__main__':
    main()