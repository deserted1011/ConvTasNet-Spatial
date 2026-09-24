#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""load_and_infer.py —— 最小示例：50 行看懂「模型怎么加载、输入怎么喂」。

    python load_and_infer.py 三麦录音.wav --out out.wav
    python load_and_infer.py 三麦录音.wav --device cpu --out out.wav

它做的事（等价于 separate.py 的 1~4 步，但只跑一条、输出 float32 波形，便于二次处理）：
    读多通道 wav -> 取 ch1 当波形 -> 算 6 通道空间特征并 z-score -> 加载模型 -> 前向 -> 写 wav

`--zero-spatial` 是**消融开关**：把空间特征全置 0 再喂进去 —— 相当于把空间这一路摘掉，
用来对比空间特征到底改变了多少输出（这也是「空间特征确实接进了网络、而且是唯一入口」的
一个侧证）。要注意它**不是**「没有空间特征也能跑」的意思，也不等于模型退化成官方底座：
权重已经微调过，第 1~512 列本身也变了。
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spatial_feat as sfeat                      # noqa: E402
from model_loader import infer, load_model, n_frames_of   # noqa: E402
from separate import READ_WAV, WRITE_WAV, resample  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="最小推理示例")
    ap.add_argument("input", help="多通道 wav（>= 3 通道；16 kHz 最佳，其它采样率会自动重采样）")
    ap.add_argument("--out", default="out.wav", help="输出 wav（float32，完整动态范围，不做峰值归一）")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--ckpt", default=os.path.join(HERE, "convtasnet_spatial.pt"))
    ap.add_argument("--scaler", default=os.path.join(HERE, "space_feat_stats.json"))
    ap.add_argument("--device", default=None, help="cpu / cuda（默认有显卡用 cuda）")
    ap.add_argument("--ref-channel", type=int, default=1, help="参考麦通道号，默认 1（linear3 中心麦）")
    ap.add_argument("--zero-spatial", action="store_true",
                    help="消融：空间特征全置 0（摘掉空间这一路，用于对比它的实际贡献）")
    args = ap.parse_args()

    x, sr = READ_WAV(args.input)
    if x.shape[0] < 3:
        raise SystemExit("输入只有 %d 个通道：本模型需要 >= 3（ch0/ch1/ch2 = 2 cm 间距线性三麦）" % x.shape[0])
    x, sr = resample(x, sr)

    model, info = load_model(ckpt=args.ckpt, device=args.device or ("cuda" if _cuda() else "cpu"),
                           config=args.config)
    print("权重 %s（epoch=%s）｜配置 %s" % (info["ckpt"], info["epoch"], os.path.basename(info["config"])))
    print("设备 %s｜参数量 %d（%.4f M）｜空间特征通道 %d"
          % (info["device"], info["n_params"], info["n_params"] / 1e6, info["n_spatial"]))

    wav = np.ascontiguousarray(x[args.ref_channel], dtype=np.float32)
    if args.zero_spatial:
        sp = np.zeros((info["n_spatial"], wav.size // sfeat.HOP), np.float32)
        print("[消融] 空间特征全置 0：空间这一路被摘掉，输出可用于对比它的实际贡献")
    else:
        raw = sfeat.extract(x, sr=sr)
        sp = sfeat.apply_scaler(raw, args.scaler)
        print("空间特征 %s（原始量纲 ITD 中位数 %+.3f us / ILD 中位数 %+.3f dB）"
              % (tuple(sp.shape), np.median(raw[0]) * 1e6, np.median(raw[1])))
    print("波形取自 ch%d，%d 样点（%.2f 秒），编码器帧数 Lf = %d"
          % (args.ref_channel, wav.size, wav.size / sr, n_frames_of(wav.size)))

    y = infer(model, wav, sp, device=info["device"])
    _d = os.path.dirname(os.path.abspath(args.out))
    if _d:
        os.makedirs(_d, exist_ok=True)          # --out 指到不存在的目录时自动建，别在最后一步崩
    WRITE_WAV(args.out, y, sr)
    print("输出 %s -> %s（%d 路，每路 %d 样点）"
          % (tuple(y.shape), args.out, y.shape[0], y.shape[1]))
    print("提醒：两路编号是任意的，先听再选；本模型不做目标人拒识。")
    return 0


def _cuda():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    sys.exit(main())
