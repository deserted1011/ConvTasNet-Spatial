#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arm_models.py —— 臂 B 的加载与推理封装（本包对外的稳定接口）。

被 separate_B.py / load_and_infer.py 调用。这一层只做三件事：

  1. **找配方与权重**：默认同目录的 config.json + 臂B_模型.pt（也可以用 --config / --ckpt 指到别处）
  2. **严格加载**：缺键 / 多键 / 形状不符 -> 当场 raise。用 strict=False 会让 518 宽的那一层被
     静默丢掉，模型照跑不误、结果却退化，且**不会报任何错** —— 这是本项目最忌讳的失败模式。
  3. **推理**：把空间特征的帧数对齐到编码器帧数（尾部多出的 1 帧丢掉），再跑前向。

用法
    from arm_models import load_arm, infer, n_frames_of
    model, info = load_arm("B", device="cpu")
    est = infer(model, "B", mix[T], spatial[6, N])      # -> [2, T]
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from conv_tasnet_spatial import (  # noqa: E402
    DEFAULT_CONFIG, KERNEL, MAX_EXTRA_FRAMES, N_SPATIAL, SR, STRIDE,
    load_model, n_frames_of,
)

__all__ = ["load_arm", "infer", "n_frames_of",
           "N_SPATIAL", "MAX_EXTRA_FRAMES", "KERNEL", "STRIDE", "SR"]


def load_arm(arm="B", ckpt=None, device="cpu", config=None):
    """载入目标模型（臂 B）。

    Args:
        arm: 只接受 "B"（本交付包只含目标模型；臂 A 在参赛材料包里，且需要另一份配方）
        ckpt: 权重文件；None = 同目录的 臂B_模型.pt
        device: "cpu" / "cuda"（以 cuda 开头但显卡不可用时会自动回退并打印提示）
        config: 结构配方；None = 同目录的 config.json
    Returns:
        (model, info)；info 里记录了权重来源、参数量、设备、训练轮次，便于复现与追溯。
    """
    arm = str(arm).upper()
    if arm != "B":
        raise ValueError(
            "本交付包只含臂 B（空间特征早融合）。臂 A 的权重在参赛材料包里，"
            "且它的 bottleneck 输入是 512、需要另一份 config.json，不能拿这份权重混用。")
    model, raw = load_model(config=config or DEFAULT_CONFIG, weights=ckpt, device=device)
    info = {
        "arm": "B",
        "ckpt": raw["weights"],
        "loaded_from": raw["weights"],
        "config": raw["config"],
        "epoch": raw["epoch"],
        "device": raw["device"],
        "n_params": raw["num_params"],
        "n_keys": raw["n_keys"],
        "n_spatial": raw["n_spatial"],
    }
    return model, info


@torch.no_grad()
def infer(model, arm, mix, spatial=None, device=None):
    """整条（或任意长度）推理。

    Args:
        model: load_arm() 出来的模型
        arm: "B"
        mix: [T] float32（或 [B, T]），单通道波形 —— 本项目取 linear3 的 ch1（中心麦）
        spatial: [6, N] float32（或 [B, 6, N]），N 应等于 T // 16（尾部多出的帧会被丢掉）
        device: 不给就用模型所在设备
    Returns:
        [n_src, T]（或 [B, n_src, T]）float32 numpy
    """
    arm = str(arm).upper()
    if arm != "B":
        raise ValueError("本交付包只含臂 B")
    x = np.asarray(mix, dtype=np.float32)
    two_d = x.ndim == 2
    if x.ndim == 1:
        x2 = x[None, :]
    elif two_d:
        x2 = x
    else:
        raise ValueError("波形维度应为 1 或 2，实际 %d" % x.ndim)
    T = int(x2.shape[-1])
    if T <= KERNEL:
        raise ValueError("输入太短：T=%d（编码器核长 %d）" % (T, KERNEL))

    dev = device or next(model.parameters()).device
    t = torch.from_numpy(np.ascontiguousarray(x2)).to(dev)

    if spatial is None:
        raise ValueError("臂 B 必须有空间特征（要做消融请显式传全零，见 load_and_infer.py）")
    sp = np.asarray(spatial, dtype=np.float32)
    if sp.ndim == 2:
        sp = sp[None, ...]
    if sp.ndim != 3 or sp.shape[1] != int(model.n_spatial):
        raise ValueError("空间特征形状应为 [%d, N]（或 [B, %d, N]），实际 %s"
                         % (int(model.n_spatial), int(model.n_spatial), tuple(sp.shape)))
    Lf = n_frames_of(T)
    n = int(sp.shape[-1])
    if n < Lf or n - Lf > MAX_EXTRA_FRAMES:
        raise ValueError("空间特征帧数 %d 与波形长度 %d 不匹配：应为 %d 帧，最多多 %d 帧"
                         % (n, T, Lf, MAX_EXTRA_FRAMES))
    sp_t = torch.from_numpy(np.ascontiguousarray(sp[:, :, :Lf])).to(dev)
    est = model(t, sp_t)
    est = est.detach().cpu().numpy().astype(np.float32)
    return est if two_d else est[0]
