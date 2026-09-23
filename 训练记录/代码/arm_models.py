#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实验 C 两臂的统一推理封装（评测与训练共用同一套前向口径）。

对应《实施方案》§6 实验 C；索引 docs/decision_log.md 的 D-27 / D-28 / P-32 / P-33 / P-35。

为什么单独一个文件
  train_tse.py 只需要「训练时怎么前向」。评测还多三件事：

    1. **整条不裁剪**推理：波形长度 T 时编码器出 Lf = T//16 - 1 帧，而空间特征文件
       是 T//16 帧 —— 尾部多 1 帧必须丢掉（P-33）。差一帧不会报错，只会让标签静默错位，
       所以这里做成显式校验而不是"差不多"。
    2. 从 checkpoint 恢复两臂，并且**只允许 bottleneck 那一层形状不同**：其余任何
       不一致（缺键 / 多键 / 形状不符）当场 raise。用 strict=False 会让 512 宽度的权重
       被静默丢弃，两臂就变成"自己跟自己比"。
    3. 记录「这一路用的是哪份权重」，写进 per_sample.csv 备查。

数值口径
  TF32 在 conv_tasnet_spatial 导入时已默认关闭（P-35）。实验 C 的公平性论证依赖
  「第 0 步两臂逐元素恒等」，开着 TF32 时 GPU 上 max|A-B| 会到 255。
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from conv_tasnet_spatial import (  # noqa: E402
    KERNEL, MAX_EXTRA_FRAMES, N_SPATIAL, SR, STRIDE,
    build_arm, build_arm_b,
)

ARMS = ("A", "B")


def n_frames_of(n_samples):
    """长度 n_samples 的片段对应的编码器帧数（P-33）。"""
    return (int(n_samples) - KERNEL) // STRIDE + 1


def spatial_path_of(root, subset, sample_id):
    """空间特征落盘路径（D-20）。"""
    return os.path.join(root, "space", subset, sample_id + ".space.npy")


def new_cols_of(model):
    """返回 (权重张量, 新增列起始下标)。只有臂 B 有新增列。"""
    conv = model.masker.bottleneck.conv
    return conv.weight, int(conv.in_channels) - N_SPATIAL


def load_arm(arm, ckpt=None, device="cuda"):
    """按臂名构造模型。ckpt=None 表示用官方预训练权重（等价于训练第 0 步）。

    Returns:
        (model, info)  info 记录权重来源，供 per_sample.csv 与答辩追溯。
    """
    arm = str(arm).upper()
    if arm == "A":
        model = build_arm("A")
    elif arm == "B":
        model = build_arm_b()
    else:
        raise ValueError("未知臂 %r（只支持 A / B）" % arm)

    info = {"arm": arm, "ckpt": None, "ckpt_arm": None, "epoch": None,
            "loaded_from": getattr(model, "loaded_from", None)}
    if ckpt:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "model" in blob:
            sd = blob["model"]
            info["epoch"] = blob.get("epoch")
            info["ckpt_arm"] = (blob.get("args") or {}).get("arm")
        else:
            sd = blob
        cur = model.state_dict()
        miss = [k for k in cur if k not in sd]
        extra = [k for k in sd if k not in cur]
        bad = [(k, tuple(sd[k].shape), tuple(cur[k].shape))
               for k in sd if k in cur and tuple(sd[k].shape) != tuple(cur[k].shape)]
        if miss or extra or bad:
            raise RuntimeError(
                "checkpoint 与臂 %s 不匹配：缺 %d 键 / 多 %d 键 / 形状不符 %d 处；"
                "前几处形状冲突 %s"
                % (arm, len(miss), len(extra), len(bad), bad[:2]))
        model.load_state_dict(sd)
        info["ckpt"] = os.path.abspath(ckpt)

    dev = str(device)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[arm_models] CUDA 不可用，回退 CPU")
        dev = "cpu"
    model.to(torch.device(dev)).eval()
    info["device"] = dev
    info["n_params"] = sum(p.numel() for p in model.parameters())
    return model, info


@torch.no_grad()
def infer(model, arm, mix, spatial=None, device=None):
    """整条（或任意长度）推理。

    Args:
        mix: [T] float32，单通道波形（本项目取 ch1 = linear3 中心麦）
        spatial: [N_SPATIAL, N] float32（当前 N_SPATIAL = 6），臂 B 必需；N 应等于 T//16（尾部多 1 帧会被丢掉）
    Returns:
        [n_src, T] float32 numpy
    """
    arm = str(arm).upper()
    x = np.asarray(mix, dtype=np.float32).ravel()
    T = int(x.size)
    if T <= KERNEL:
        raise ValueError("输入太短：T=%d（编码器 kernel=%d）" % (T, KERNEL))
    dev = device or next(model.parameters()).device
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).to(dev)

    if arm == "B":
        if spatial is None:
            raise ValueError("臂 B 必须有空间特征（消融请显式传全零）")
        sp = np.asarray(spatial, dtype=np.float32)
        if sp.ndim != 2 or sp.shape[0] != N_SPATIAL:
            raise ValueError("空间特征形状应为 [%d, N]，实际 %s" % (N_SPATIAL, tuple(sp.shape)))
        Lf = n_frames_of(T)
        n = int(sp.shape[1])
        if n < Lf or n - Lf > MAX_EXTRA_FRAMES:
            raise ValueError("空间特征帧数 %d 与波形长度 %d 不匹配：应为 %d 帧，最多多 %d 帧"
                             % (n, T, Lf, MAX_EXTRA_FRAMES))
        sp_t = torch.from_numpy(np.ascontiguousarray(sp[:, :Lf])).unsqueeze(0).to(dev)
        est = model(t, sp_t)
    else:
        est = model(t)

    return est[0].detach().cpu().numpy().astype(np.float32)