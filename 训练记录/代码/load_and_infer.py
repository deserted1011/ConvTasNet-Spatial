#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小示例：加载本模型权重，对一条多通道 wav 出 [2, T]。

    python 代码/load_and_infer.py --mix <多通道 wav> --ref-channel 1 --out out.wav
    python 代码/load_and_infer.py --mix <wav> --spatial <xxx.space.npy> --out out.wav

注意：**不给 --spatial 就用全零特征**，相当于把空间这一路摘掉（消融口径），
用来对比空间特征的实际贡献；不代表「没有空间特征也能跑」。
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from model_io import infer, load_model, n_frames_of  # noqa: E402
from conv_tasnet_spatial import N_SPATIAL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True, help="多通道 wav（16 kHz）")
    ap.add_argument("--ref-channel", type=int, default=1, help="参考通道，默认 1（linear3 中心麦）")
    ap.add_argument("--spatial", default=None, help="[6, N] float32 npy（1000 Hz、已 z-score）；不给 = 全零")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "ckpt_ep30.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out.wav")
    args = ap.parse_args()

    with sf.SoundFile(args.mix) as f:
        sr = f.samplerate
        x = f.read(dtype="float32", always_2d=True)[:, args.ref_channel]
    if sr != 16000:
        raise SystemExit("采样率应为 16000 Hz，实际 %d" % sr)

    model, info = load_model(ckpt=args.ckpt, device=args.device)
    print("权重 %s / epoch %s / 设备 %s / 参数量 %.4f M"
          % (info["ckpt"] or info["loaded_from"], info["epoch"], info["device"], info["n_params"] / 1e6))

    Lf = n_frames_of(len(x))
    if args.spatial:
        sp = np.load(args.spatial).astype(np.float32)
        if sp.ndim != 2 or sp.shape[0] != N_SPATIAL:
            raise SystemExit("空间特征形状应为 [%d, N]，实际 %s" % (N_SPATIAL, tuple(sp.shape)))
    else:
        sp = np.zeros((N_SPATIAL, Lf), np.float32)
        print("[提示] 未给空间特征，用全零：空间这一路被摘掉（消融口径），用于对比它的实际贡献")

    y = infer(model, x, sp, device=info["device"])
    sf.write(args.out, y.T, sr, subtype="FLOAT")
    print("输入 %d 样本 -> 输出 %s -> %s" % (len(x), tuple(y.shape), args.out))


if __name__ == "__main__":
    main()
