# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import pprint

import clip
import torch
import torchvision
import numpy as np
import torch.nn as nn
import bitsandbytes as bnb
import yaml

from scipy.spatial.transform import Rotation
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR

import rvt.utils.peract_utils as peract_utils
import rvt.mvt.utils as mvt_utils
import rvt.utils.rvt_utils as rvt_utils
import peract_colab.arm.utils as arm_utils

from rvt.mvt.augmentation import apply_se3_aug_con, aug_utils
from peract_colab.arm.optim.lamb import Lamb
from yarr.agents.agent import ActResult
from rvt.utils.dataset import _clip_encode_text
from rvt.utils.lr_sched_utils import GradualWarmupScheduler



def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        out[key] = np.mean(np.array(val))
    pprint.pprint(out)
    return out


import pytorch_lightning as pl

class EnergyPredictorTransformer(pl.LightningModule):
    def __init__(
        self, 
        state_dim: int = 21,
        action_dim: int = 9,
        fused_dim: int = 768,
        use_fused: bool = True,
        comp: bool = True,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1, 
    ):
        super().__init__()
        self.save_hyperparameters()
        
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
        if use_fused:
            if comp:
                self.fused_compressor = nn.Sequential(
                    nn.Linear(fused_dim, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    
                    nn.Linear(256, 16), # ここで16次元にする
                    nn.LayerNorm(16),
                    nn.GELU()
                )

                self.fused_embed = nn.Sequential(
                    nn.Linear(16, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU()
                )
            else:
                self.fused_embed = nn.Sequential(
                    nn.Linear(fused_dim, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU()
                )
        self.step_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=False, # バグ回避
            norm_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        if self.hparams.use_fused:
            features, step_val, fused_feat = x
        else:
            features, step_val = x
        batch_size = features.size(0)
        
        current_state = features[:, :self.hparams.state_dim]
        next_action   = features[:, self.hparams.state_dim:]
        
        t_state  = self.state_embed(current_state).unsqueeze(1)
        t_action = self.action_embed(next_action).unsqueeze(1)
        t_step   = self.step_embed(step_val).unsqueeze(1)
        t_cls    = self.cls_token.expand(batch_size, -1, -1)
        
        if self.hparams.use_fused:
            if self.hparams.comp:
                compressed = self.fused_compressor(fused_feat)
                t_fused = self.fused_embed(compressed).unsqueeze(1)
            else:
                t_fused = self.fused_embed(fused_feat).unsqueeze(1)

            src = torch.cat((t_cls, t_state, t_action, t_fused, t_step), dim=1)
        else:
            src = torch.cat((t_cls, t_state, t_action, t_step), dim=1)
        src = src.permute(1, 0, 2).contiguous() # バグ回避のための転置
        
        out = self.transformer(src)
        cls_output = out[0]
        return self.head(cls_output)


class RVTAgent:
    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int,
        stage_two: bool,
        add_lang: bool,
        amp: bool,
        bnb: bool,
        move_pc_in_bound: bool,
        lr: float = 0.0001,
        lr_cos_dec: bool = False,
        cos_dec_max_step: int = 60000,
        warmup_steps: int = 0,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = peract_utils.SCENE_BOUNDS,
        cameras: list = peract_utils.CAMERAS,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir="",
    ):
        """
        :param gt_hm_sigma: the std of the groundtruth hm, currently for for
            2d, if -1 then only single point is considered
        :type gt_hm_sigma: float
        :param rot_ver: version of the rotation prediction network
            Either:
                0: same as peract, independent discrete xyz predictions
                1: xyz prediction dependent on one another
        :param rot_x_y_aug: only applicable when rot_ver is 1, it specifies how
            much error we should add to groundtruth rotation while training
        :param log_dir: a folder location for saving some intermediate data
        """

        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.amp = amp
        self.bnb = bnb
        self.stage_two = stage_two
        self.add_lang = add_lang
        self.log_dir = log_dir
        self.warmup_steps = warmup_steps
        self.lr_cos_dec = lr_cos_dec
        self.cos_dec_max_step = cos_dec_max_step
        self.scene_bounds = scene_bounds
        self.cameras = cameras
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

        self.scaler = GradScaler(enabled=self.amp)

        self.last_action_dict = None
        self.last_soft_prediction = None
        self.last_fused_feat = None

        # self.energy_model_type = "mlp"
        self.energy_model_type = "transformer"
        self.use_fused = False
        self.detach_fused = False
        # self.use_fused = True
        # self.detach_fused = True

        self.comp = False
        # self.comp = True

        self.freeze_backbone = False
        # self.freeze_backbone = True


        # self.energy_config_path = '../energy/checkpoints/TF_Fused_EnergyPredictor/dr0.2_comp_768-256-16/energy_config_10.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_Fused_EnergyPredictor/dr0.2_comp_768-256-16/energy_config_60.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_Fused_EnergyPredictor/dr0.2_comp_768-256-16/energy_config_80.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_Fused_EnergyPredictor/dr0.2_comp_768-256-16/energy_config_100.yaml'

        # self.energy_config_path = '../energy/checkpoints/TF_Fused_EnergyPredictor/Transformer-d128-fusedA_lr0.001_dr0.1_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_20.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_Fused_EnergyPredictor/Transformer-d128-fusedA_lr0.001_dr0.1_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_10.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_Fused_EnergyPredictor/Transformer-d128-fusedA_lr0.001_dr0.1_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_80.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_Fused_EnergyPredictor/Transformer-d128-fusedA_lr0.001_dr0.1_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_100.yaml'
        
        # self.energy_config_path = '../energy/checkpoints/TF_NoFused_EnergyPredictor_nohdbscan/Transformer-d128-nhead4-lyr3_lr0.001_dr0.2_fused7111216.npz_numlayer-3/energy_config_10.yaml'

        # self.energy_config_path = '../energy/checkpoints/TF_NoFused_EnergyPredictor/Transformer-d128-nhead4-lyr3_lr0.001_dr0.2_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_lambda5.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_NoFused_EnergyPredictor/Transformer-d128-nhead4-lyr3_lr0.001_dr0.2_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_lambda10-7.yaml'
        self.energy_config_path = '../../energy/checkpoints/TF_NoFused_EnergyPredictor/Transformer-d128-nhead4-lyr3_lr0.001_dr0.2_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_lambda10.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_NoFused_EnergyPredictor/Transformer-d128-nhead4-lyr3_lr0.001_dr0.2_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_lambda15-5.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_NoFused_EnergyPredictor/Transformer-d128-nhead4-lyr3_lr0.001_dr0.2_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_lambda10-5.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_NoFused_EnergyPredictor/Transformer-d128-nhead4-lyr3_lr0.001_dr0.2_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_lambda10-3.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_NoFused_EnergyPredictor/Transformer-d128-nhead4-lyr3_lr0.001_dr0.2_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_lambda9.yaml'
        # self.energy_config_path = '../energy/checkpoints/TF_NoFused_EnergyPredictor/Transformer-d128-nhead4-lyr3_lr0.001_dr0.2_fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz_numlayer-3/energy_config_lambda7.yaml'

        # self.energy_config_path = '../energy/checkpoints/notfused_EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-256-128-64_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda3.yaml'
        # self.energy_config_path = '../energy/checkpoints/notfused_EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-256-128-64_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda2.yaml'
        # self.energy_config_path = '../energy/checkpoints/notfused_EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-256-128-64_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config.yaml'
        # self.energy_config_path = '../energy/checkpoints/notfused_EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-256-128-64_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda5.yaml'
        # self.energy_config_path = '../energy/checkpoints/notfused_EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-256-128-64_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda10.yaml'
        # self.energy_config_path = '../energy/checkpoints/notfused_EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-256-128-64_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda20.yaml'
        # self.energy_config_path = '../energy/checkpoints/notfused_EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-256-128-64_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda30.yaml'
        # self.energy_config_path = '../energy/checkpoints/notfused_EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-256-128-64_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda40.yaml'
        # self.energy_config_path = '../energy/checkpoints/notfused_EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-256-128-64_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda05.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused711_hdbscan-0.001-sample-5-MIN-20-step-under10.npz/energy_config_13.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_13.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda-2.0.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda-2.5.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_13task_lambda-2.0.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_13task_lambda-1.0.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda-1.5.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda-0.1.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda-0.25.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config_lambda-0.5.yaml'
        # self.energy_config_path = '../energy/checkpoints/EnergyPredictor_kai_ultra_super_new_super_tani/arch-MLP-512-256-128_lr-0.001_drop-0.3-fused7111216_hdbscan-0.001-sample-20-MIN-20-step-eplison-0.5-under10.npz/energy_config.yaml'
        self.energy_in_dim = 31
        # self.energy_in_dim = 31+768
        self.energy_dropout = 0.3
        self.energy_task_configs = {}
        self.energy_nets = nn.ModuleDict()

        # try:
        with open(self.energy_config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.energy_config = config
        
        if 'model_shared' in config:
            self.energy_in_dim = config['model_shared'].get('in_dim', self.energy_in_dim)
            self.energy_dropout = config['model_shared'].get('dropout', self.energy_dropout)

        if 'task_configs' in config:
            self.energy_task_configs = config.get('task_configs', {})

        # except FileNotFoundError:
        #     print(f"Warning: Energy config file not found: {self.energy_config_path}")
        # except Exception as e:
        #     print(f"Error loading energy config: {e}")

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        trainable_params = []
        if self.freeze_backbone and training:
            print("--- Config: Virtual Freezing (Optimizer Filtering) ---")
            
            for name, param in self._network.named_parameters():
                # 名前で判定（UpSample層と出力層は学習対象、それ以外は無視）
                # 'up0' = ヒートマップ生成部, 'feat_fc' = 回転・状態予測部
                if "up0" in name or "feat_fc" in name:
                    trainable_params.append(param) # ★リストに追加（学習する）
            
        else:
            # フリーズしない場合は全員追加
            trainable_params = [p for p in self._network.parameters() if p.requires_grad]
        
        if self._optimizer_type == "lamb":
            if self.bnb:
                print("Using 8-Bit Optimizer")
                self._optimizer = bnb.optim.LAMB(
                    trainable_params,
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                )
            else:
                # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
                self._optimizer = Lamb(
                    trainable_params,
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                trainable_params,
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception("Unknown optimizer")

        if self.lr_cos_dec:
            after_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=self.cos_dec_max_step,
                eta_min=self._lr / 100,  # mininum lr
            )
        else:
            after_scheduler = None
        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=after_scheduler,
        )

        
        for task_name, config in self.energy_task_configs.items():

            if not config.get('enabled', True):
                continue
        
            ckpt_path = config.get('ckpt_path')
            model = self._load_energy_predictor(ckpt_path=ckpt_path, in_dim=self.energy_in_dim, dropout=self.energy_dropout)
            self.energy_nets[task_name] = model

        self.energy_nets.to(self._device)


    def _load_energy_predictor(self, ckpt_path: str, in_dim: int = 31, dropout: float = 0.3):
        
        if self.energy_model_type == "transformer":
            # Transformerの定義
            model = EnergyPredictorTransformer(
                state_dim=21, 
                action_dim=9,
                fused_dim = 768,
                use_fused=self.use_fused, 
                comp=self.comp,
                d_model=128,
                nhead=4,
                num_layers=3, 
                dropout=0.1, 
            ).to(self._device)
            print(f"Loading [Transformer] Energy Predictor from {ckpt_path}")

        elif self.energy_model_type == "mlp":
            model = nn.Sequential(
                nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 1)
            ).to(self._device)
            print(f"Loading [MLP] Energy Predictor from {ckpt_path}")
            
        else:
            raise ValueError(f"Unknown energy_model_type: {self.energy_model_type}")

        # 重みロード
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        
        # prefix調整
        if self.energy_model_type == "mlp":
            # MLPの場合、学習コードによっては "model." が付くことがあるので除去
            model_sd = {k.replace("model.", ""): v for k, v in sd.items() if k.startswith("model.")}
            if len(model_sd) == 0: model_sd = sd # そのままの場合
        else:
            # Transformer (LightningModule) はそのまま
            model_sd = sd

        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        if missing or unexpected:
            print("[EnergyPredictor] state_dict mismatch:", "missing:", missing, "unexpected:", unexpected)

        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model


    # def _load_energy_predictor(self, ckpt_path: str, in_dim: int = 31, dropout: float = 0.3):
    #     import torch.nn as nn, torch

    #     if self.energy_model_type == "transformer":

    #     else: 
    #         energy_net = nn.Sequential(
    #             nn.Linear(in_dim, 256), 
    #             nn.BatchNorm1d(256), 
    #             nn.ReLU(), 
    #             nn.Dropout(dropout),
                
    #             nn.Linear(256, 128),    
    #             nn.BatchNorm1d(128), 
    #             nn.ReLU(), 
    #             nn.Dropout(dropout),
                
    #             nn.Linear(128, 64),     
    #             nn.BatchNorm1d(64),  
    #             nn.ReLU(), 
    #             nn.Dropout(dropout),
                
    #             nn.Linear(64, 1)
    #         ).to(self._device)

    #     ckpt = torch.load(ckpt_path, map_location="cpu")
    #     sd = ckpt.get("state_dict", ckpt)
    #     model_sd = {k.replace("model.",""): v for k,v in sd.items() if k.startswith("model.")}
    #     missing, unexpected = energy_net.load_state_dict(model_sd, strict=False)
    #     if missing or unexpected:
    #         print("[EnergyPredictor] state_dict mismatch:", "missing:", missing, "unexpected:", unexpected)

    #     energy_net.eval()
    #     for p in energy_net.parameters():
    #         p.requires_grad = False
    #     return energy_net



    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    # copied from per-act and removed the translation part
    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans, and replace all other
            # q
            if self.stage_two:
                out = out["mvt2"]
                q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if self.rot_ver == 0:
            # (bs, 218)
            rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
            grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
            # (bs, 2)
            collision_q = out["feat"].view(bs, -1)[
                :, self.num_all_rot + 2 : self.num_all_rot + 4
            ]
        elif self.rot_ver == 1:
            rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]),
                              dim=-1).view(bs, -1)
            grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
            collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]
        else:
            assert False

        y_q = None

        return q_trans, rot_q, grip_q, collision_q, y_q, pts

    def update(
        self,
        step: int,
        replay_sample: dict,
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
    ) -> dict:
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
        assert replay_sample["lang_goal_embs"].shape[1:] == (1, 77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (
            1,
            self._net_mod.proprio_dim,
        )

        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
            :, -1
        ].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][
            :, -1
        ].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        action_grip = action_rot_grip[:, -1]  # (b,)
        lang_goal_embs = replay_sample["lang_goal_embs"][:, -1].float()
        tasks = replay_sample["tasks"]

        proprio = arm_utils.stack_on_channel(replay_sample["low_dim_state"])  # (b, 4)
        return_out = {}

        obs, pcd = peract_utils._preprocess_inputs(replay_sample, self.cameras)

        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)

            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        with autocast(enabled=self.amp):
            (
                action_rot_x_one_hot,
                action_rot_y_one_hot,
                action_rot_z_one_hot,
                action_grip_one_hot,  # (bs, 2)
                action_collision_one_hot,  # (bs, 2)
            ) = self._get_one_hot_expert_actions(
                bs, action_rot, action_grip, action_ignore_collisions, device=self._device
            )

            if self.rot_ver == 1:
                rot_x_y = torch.cat(
                    [
                        action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                        action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                    ],
                    dim=-1,
                )
                if self.rot_x_y_aug != 0:
                    # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                    rot_x_y += torch.randint(
                        -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                    ).to(rot_x_y.device)
                    rot_x_y %= self._num_rotation_classes

            out = self._network(
                pc=pc,
                img_feat=img_feat,
                proprio=proprio,
                lang_emb=lang_goal_embs,
                img_aug=img_aug,
                wpt_local=wpt_local if self._network.training else None,
                rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            )

            q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
                out, dims=(bs, nc, h, w)
            )

            action_trans = self.get_action_trans(
                wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
            )

        loss_log = {}
        if backprop:
            with autocast(enabled=self.amp):
                # cross-entropy loss
                trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
                rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                grip_loss = 0.0
                collision_loss = 0.0
                if self.add_rgc_loss:
                    rot_loss_x = self._cross_entropy_loss(
                        rot_q[
                            :,
                            0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                        ],
                        action_rot_x_one_hot.argmax(-1),
                    ).mean()

                    rot_loss_y = self._cross_entropy_loss(
                        rot_q[
                            :,
                            1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                        ],
                        action_rot_y_one_hot.argmax(-1),
                    ).mean()

                    rot_loss_z = self._cross_entropy_loss(
                        rot_q[
                            :,
                            2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                        ],
                        action_rot_z_one_hot.argmax(-1),
                    ).mean()

                    grip_loss = self._cross_entropy_loss(
                        grip_q,
                        action_grip_one_hot.argmax(-1),
                    ).mean()

                    collision_loss = self._cross_entropy_loss(
                        collision_q, action_collision_one_hot.argmax(-1)
                    ).mean()

                total_loss = (
                    trans_loss
                    + rot_loss_x
                    + rot_loss_y
                    + rot_loss_z
                    + grip_loss
                    + collision_loss
                )


                pred_wpt, pred_rot_quat, pred_grip, pred_coll = self._soft_pred_feats(out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info, tau=1.0)
                state_vec = replay_sample["energy_robot_state"][:, -1, :].float().to(self._device)
                step_val = replay_sample["keypoint_idx"].float().to(self._device) + 1.0
                step_val_norm = (step_val - 1.0) / 24.0
                action_feat = torch.cat([
                    pred_wpt,       # (B, 3)
                    pred_rot_quat,  # (B, 4)
                    pred_grip,      # (B, 1)
                    pred_coll,      # (B, 1)
                ], dim=-1) # (B, 9)

                total_energy_loss_sum = torch.tensor(0.0, device=self._device, dtype=state_vec.dtype)
                bs = state_vec.shape[0]

                fused_feat = None
                if self.use_fused:
                    if self.detach_fused:
                        fused_feat = out['mvt2']['fused_feat_vector'].detach()
                    else:
                        fused_feat = out['mvt2']['fused_feat_vector']


                for task_name, model in self.energy_nets.items():
                    task_config = self.energy_task_configs[task_name]
                    mask = torch.tensor([t == task_name for t in tasks], device=self._device)
                    if mask.sum() == 0:
                        continue
                        
                    idx = mask.nonzero(as_tuple=True)[0]
                    norm_config = task_config['normalization']
                    energy_mean = norm_config['mean']
                    energy_std = norm_config['std']
                    energy_min = norm_config['min']
                    energy_max = norm_config['max']

                    energy_lambda = task_config['lambda'] 

                    if self.energy_model_type == "transformer":
                        feat_tf = torch.cat([state_vec[idx], action_feat[idx]], dim=-1) 
                        step_tf = step_val_norm.unsqueeze(-1)[idx]

                        if self.use_fused:
                            e_hat = model((feat_tf, step_tf, fused_feat[idx])).view(-1)
                        else:
                            e_hat = model((feat_tf, step_tf)).view(-1)

                    elif self.energy_model_type == "mlp":
                        feat_mlp = torch.cat([
                            state_vec[idx], 
                            action_feat[idx], 
                            step_val.unsqueeze(-1)[idx] # <--- 生のステップ
                        ], dim=-1)
                        e_hat = model(feat_mlp).view(-1)
                    
                    else:
                        raise ValueError("Invalid energy_model_type")

                    e_hat_raw = e_hat.mul(energy_std).add(energy_mean)

                    e_hat_clamped = torch.clamp(e_hat_raw, min=energy_min, max=energy_max)

                    e_norm = (e_hat_clamped - energy_min) / (energy_max - energy_min + 1e-8)

                    task_loss_sum = (energy_lambda * e_norm).sum()
                    total_energy_loss_sum += task_loss_sum

                energy_loss = total_energy_loss_sum / float(bs)
                total_loss = total_loss + energy_loss


            self._optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self._optimizer)
            self.scaler.update()
            self._lr_sched.step()

            loss_log = {
                "energy_loss": energy_loss.item(),
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        if eval_log:
            with torch.no_grad():
                wpt = torch.cat([x.unsqueeze(0) for x in wpt])
                pred_wpt, pred_rot_quat, _, _ = self.get_pred(
                    out,
                    rot_q,
                    grip_q,
                    collision_q,
                    y_q,
                    rev_trans,
                    dyn_cam_info=dyn_cam_info,
                )

                return_log = manage_eval_log(
                    self=self,
                    tasks=tasks,
                    wpt=wpt,
                    pred_wpt=pred_wpt,
                    action_rot=action_rot,
                    pred_rot_quat=pred_rot_quat,
                    action_grip_one_hot=action_grip_one_hot,
                    grip_q=grip_q,
                    action_collision_one_hot=action_collision_one_hot,
                    collision_q=collision_q,
                    reset_log=reset_log,
                )

                return_out.update(return_log)

        return return_out

    @torch.no_grad()
    def act(
        self, step: int, observation: dict, deterministic=True, pred_distri=False
    ) -> ActResult:
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_goal_tokens[0])
            lang_goal_embs = lang_goal_embs.float()
        else:
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )

        proprio = arm_utils.stack_on_channel(observation["low_dim_state"])

        obs, pcd = peract_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )

        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )

        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_goal_embs,
            img_aug=0,  # no img augmentation while acting
        )

        self.last_fused_feat = out["mvt2"]["fused_feat_vector"].detach()

        _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
            out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
        )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )

        self.last_action_dict = {
            'position': pred_wpt[0].cpu().numpy(),
            'rotation_quat': pred_rot_quat[0],
            'gripper_state': pred_grip[0].cpu().numpy(),
            'collision_state': pred_coll[0].cpu().numpy(),
        }

        # ソフトな予測結果を計算し、保存する
        s_wpt, s_rot, s_grip, s_coll = self._soft_pred_feats(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )
        self.last_soft_prediction = {
            'position': s_wpt[0].cpu().numpy(),
            'rotation_quat': s_rot[0].cpu().numpy(),
            'gripper_state': s_grip[0].cpu().numpy(),
            'collision_state': s_coll[0].cpu().numpy(),
        }

        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
            )
        )
        if pred_distri:
            x_distri = rot_grip_q[
                0,
                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
            ]
            y_distri = rot_grip_q[
                0,
                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
            ]
            z_distri = rot_grip_q[
                0,
                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
            ]
            return ActResult(continuous_action), (
                x_distri.cpu().numpy(),
                y_distri.cpu().numpy(),
                z_distri.cpu().numpy(),
            )
        else:
            return ActResult(continuous_action)
        

    def _soft_pred_feats(self, out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info, tau=1.0):
        # ---- waypoint ----
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(out, mvt1_or_mvt2, dyn_cam_info, y_q)
        pred_wpt = torch.stack(
            [_rev_trans(_pred_wpt_local) for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans)],
            dim=0
        )  # (B,3)

        # ---- rotation: full-soft (no ST), extrinsic 'xyz' matched to: euler = id*res - 180
        B = rot_q.shape[0]
        C = self._num_rotation_classes
        res = float(self._rotation_resolution)

        # 角度グリッドの sin/cos をキャッシュ（C や res が変われば再生成）
        cache_key = (int(C), float(res))
        if getattr(self, "_rot_cache_key", None) != cache_key:
            grid_deg = torch.linspace(-180.0, -180.0 + res * (C - 1), C, dtype=torch.float32)  # CPU, fp32
            grid_rad = torch.deg2rad(grid_deg)
            self._sin_grid_host = torch.sin(grid_rad)  # CPU保持
            self._cos_grid_host = torch.cos(grid_rad)
            self._rot_cache_key = cache_key

        sin_g = self._sin_grid_host.to(device=rot_q.device, dtype=rot_q.dtype)  # (C,)
        cos_g = self._cos_grid_host.to(device=rot_q.device, dtype=rot_q.dtype)  # (C,)

        # logits -> prob（温度 tau）。混合精度は維持
        rx_logit = rot_q[:, 0*C:1*C] / float(tau)  # (B,C)
        ry_logit = rot_q[:, 1*C:2*C] / float(tau)
        rz_logit = rot_q[:, 2*C:3*C] / float(tau)
        px = torch.softmax(rx_logit, dim=-1)       # (B,C)
        py = torch.softmax(ry_logit, dim=-1)
        pz = torch.softmax(rz_logit, dim=-1)

        # 円周平均角: GEMV で集約し、atan2 直前のみ FP32 に
        sx = torch.matmul(px, sin_g).float(); cxv = torch.matmul(px, cos_g).float()
        sy = torch.matmul(py, sin_g).float(); cyv = torch.matmul(py, cos_g).float()
        sz = torch.matmul(pz, sin_g).float(); czv = torch.matmul(pz, cos_g).float()
        rx = torch.atan2(sx, cxv)  # (B,)
        ry = torch.atan2(sy, cyv)
        rz = torch.atan2(sz, czv)

        # 外因XYZ: q = qz ⊗ qy ⊗ qx（xyzw）
        hx, hy, hz = rx * 0.5, ry * 0.5, rz * 0.5
        cx, cy, cz = torch.cos(hx), torch.cos(hy), torch.cos(hz)
        sxh, syh, szh = torch.sin(hx), torch.sin(hy), torch.sin(hz)

        # 閉形式（qz*qy*qx）
        x = sxh * cy * cz - cx * syh * szh
        y = cx * syh * cz + sxh * cy * szh
        z = cx * cy * szh - sxh * syh * cz
        w = cx * cy * cz + sxh * syh * szh
        q = torch.stack([x, y, z, w], dim=-1)  # (B,4)

        # 正規化と符号規約（w>=0）
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        q = torch.where(q[:, 3:4] < 0, -q, q)
        pred_rot_quat = q.to(rot_q.dtype)  # xyzw

        # ---- gripper / collision: 全面ソフト
        pred_grip = torch.softmax(grip_q, dim=-1)[:, 1:2].to(rot_q.dtype)       # (B,1)
        pred_coll = torch.softmax(collision_q, dim=-1)[:, 1:2].to(rot_q.dtype)  # (B,1)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll




    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(
            out, mvt1_or_mvt2, dyn_cam_info, y_q
        )

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=False,
                dyn_cam_info=dyn_cam_info,
                out=out,
            )
            assert wpt_img2.shape[1] == 1

            # (bs, 1, 2 * num_img, 2)
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans

    def reset(self):
        pass

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
