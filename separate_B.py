#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""separate_B.py —— 臂 B 分离主脚本（这个包里**唯一**需要你跑的入口）

把一段**远场混合录音**分成两路。用法与臂 A 版一致，唯一区别：输入必须带
**>= 3 个麦克风通道**（ch0/ch1/ch2 = 2 cm 间距的线性三麦，ch1 是参考麦）。

    python separate_B.py 三麦录音.wav
    python separate_B.py 三麦录音.wav --outdir 结果 --device cpu
    python separate_B.py 长录音.wav --segment-seconds 15 --overlap-seconds 1

输出（默认与输入同目录 / `--outdir`）：
    <输入名>_spk1.wav     16 kHz / PCM16 / 峰值归一 0.9
    <输入名>_spk2.wav     同上
    <输入名>_debug.png     ITD/ILD 逐帧曲线 + 几何上限参考线（没有 matplotlib 时退化为 .csv）
    <输入名>_debug.csv    逐帧空间特征（原始量纲 + z 版），两张图任何一个都缺不了它

⚠️ 输出是「两路」，不是「目标那一路」：模型是双说话人分离器，`_spk1` / `_spk2`
   的编号是任意的，**先听再选**。要"只留某一个人的声音"得再做一步声纹比对（不在本包内）。
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE                                  # 本脚本与 config.json / 权重 / 标尺同目录（交付包根）
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spatial_feat as sfeat                # noqa: E402


# ---------------------------------------------------------------- wav 读写（soundfile 优先）
def _io_backend():
    try:
        import soundfile as sf

        def read(p):
            x, sr = sf.read(p, dtype="float32", always_2d=True)
            return np.ascontiguousarray(x.T), int(sr)

        def write(p, y, sr):
            sf.write(p, np.clip(y, -1.0, 1.0).T, sr, subtype="PCM_16")

        return read, write, "soundfile"
    except ImportError:
        from scipy.io import wavfile

        def read(p):
            sr, x = wavfile.read(p)
            x = np.asarray(x)
            if x.dtype.kind == "i":
                x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
            if x.ndim == 1:
                x = x[:, None]
            return np.ascontiguousarray(x.T.astype(np.float32)), int(sr)

        def write(p, y, sr):
            # scipy 要 [样点, 声道]，本项目内部一律 [声道, 样点] —— 二维时必须转置，
            # 否则样点数会被当成声道数写进 WAV 头（长音频直接 struct.error）
            y = np.asarray(y)
            if y.ndim == 2:
                y = y.T
            wavfile.write(p, sr, (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16))

        return read, write, "scipy"


READ_WAV, WRITE_WAV, IO_BACKEND = _io_backend()


def resample(x, sr, target=sfeat.FS):
    """[C, T] 任意采样率 -> 16 kHz。输入本来是 16 kHz 时原样返回。"""
    if int(sr) == target:
        return x, int(sr)
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(target))
        y = resample_poly(x, target // g, int(sr) // g, axis=1)
        return np.ascontiguousarray(y, dtype=np.float32), target
    except ImportError:
        raise SystemExit("输入是 %d Hz，重采样需要 scipy（pip install scipy）；"
                         "或先把音频转成 16000 Hz" % sr)


def peak_norm(y, peak=0.9):
    m = float(np.max(np.abs(y))) if np.size(y) else 0.0
    return y if m <= 0 else (y * (peak / m)).astype(np.float32)


def write_debug(outdir, stem, raw, z, stats, diag):
    """逐帧空间特征落 CSV（永远写），能画就再落一张 PNG。"""
    csv_p = os.path.join(outdir, stem + "_debug.csv")
    with open(csv_p, "w", encoding="utf-8", newline="") as fh:
        head = ["frame"] + ["%s_raw" % n for n in sfeat.FEAT_NAMES] + \
               ["%s_z" % n for n in sfeat.FEAT_NAMES]
        fh.write(",".join(head) + "\n")
        for t in range(raw.shape[1]):
            fh.write(",".join(["%d" % t] + ["%.9g" % v for v in raw[:, t]] +
                              ["%.6f" % v for v in z[:, t]]) + "\n")
    nf = max(int(raw.shape[1]), 1)
    over = diag.get("over", [0, 0, 0])
    edge = diag.get("edge", [0, 0, 0])
    print("      空间特征质检：夹取前越界帧 %s（占 %.4f%%），触及搜索窗边界（估计器失效）%s（占 %.4f%%）"
          % (list(over), 100.0 * sum(over) / (nf * len(sfeat.PAIRS)),
             list(edge), 100.0 * sum(edge) / (nf * len(sfeat.PAIRS))))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("      [提示] 没装 matplotlib，debug 图跳过（CSV 已写出：%s）" % os.path.basename(csv_p))
        return csv_p
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    t = np.arange(raw.shape[1]) / 1000.0
    for k, (i, j) in enumerate(sfeat.PAIRS):
        ax[0].plot(t, raw[2 * k] * 1e6, lw=0.7, label="ITD_%d%d" % (i, j))
        ax[0].axhline(sfeat.GEOM_MAX_ITD_S[k] * 1e6, ls="--", lw=0.6, color="grey")
        ax[0].axhline(-sfeat.GEOM_MAX_ITD_S[k] * 1e6, ls="--", lw=0.6, color="grey")
        ax[1].plot(t, raw[2 * k + 1], lw=0.7, label="ILD_%d%d" % (i, j))
    ax[0].set_ylabel("ITD (us)"); ax[1].set_ylabel("ILD (dB)"); ax[1].set_xlabel("time (s)")
    ax[0].set_title("spatial features | %s | frames=%d | clipped-to-geom=%d (%.4f%%)"
                    % (stem, raw.shape[1], sum(over), 100.0 * sum(over) / nf))
    for a in ax:
        a.legend(loc="upper right", fontsize=7); a.grid(alpha=0.25)
    png_p = os.path.join(outdir, stem + "_debug.png")
    fig.tight_layout(); fig.savefig(png_p, dpi=130); plt.close(fig)
    return png_p


def segment_ranges(T, seg, ov):
    """把 [0, T) 切成若干段；起点必须落在 16 的整数倍（帧对齐）。seg<=0 表示不切。"""
    if seg <= 0 or T <= seg:
        return [(0, T)]
    step = max(seg - ov, sfeat.HOP)
    out = []
    s = 0
    while s < T:
        e = min(s + seg, T)
        out.append((s, e))
        if e >= T:
            break
        s += step
    return out


def main():
    ap = argparse.ArgumentParser(description="臂 B（空间特征早融合）分离脚本")
    ap.add_argument("input", help="输入音频路径，**>= 3 通道**，16 kHz 最佳（其他采样率会自动重采样）")
    ap.add_argument("--outdir", default=None, help="输出目录（默认与输入同目录）")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"), help="结构配方")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "臂B_模型.pt"), help="臂 B 权重")
    ap.add_argument("--scaler", default=os.path.join(HERE, "space_feat_stats.json"),
                    help="标准化标尺（训练时那份 train-fit；换一份就等于换一套尺度）")
    ap.add_argument("--device", default=None, help="cpu / cuda（默认有显卡用 cuda）")
    ap.add_argument("--ref-channel", type=int, default=1, help="参考麦通道号，默认 1（linear3 中心麦）")
    ap.add_argument("--segment-seconds", type=float, default=20.0, help="分块长度（秒）；0 = 不切块")
    ap.add_argument("--overlap-seconds", type=float, default=0.5, help="块间重叠（秒）")
    args = ap.parse_args()

    t0 = time.time()
    if not os.path.isfile(args.input):
        raise SystemExit("找不到输入文件：%s" % args.input)
    if not os.path.isfile(args.config):
        raise SystemExit("找不到结构配方：%s" % args.config)
    if not os.path.isfile(args.scaler):
        raise SystemExit("找不到标准化标尺：%s" % args.scaler)
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)

    from arm_models import infer, load_arm, n_frames_of   # 延迟导入：先报清楚参数错误
    import torch

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, info = load_arm("B", ckpt=args.ckpt, device=dev)
    print("[1/4] 设备：%s（torch %s，io=%s）" % (info["device"], torch.__version__, IO_BACKEND))
    print("[2/4] 模型就绪：%d 参数（%.4f M），权重来自 %s（训练第 %s 轮，见 config.json）"
          % (info["n_params"], info["n_params"] / 1e6, os.path.basename(args.ckpt),
             cfg.get("training_provenance", {}).get("epochs", "?")))

    x, sr = READ_WAV(args.input)
    if x.shape[0] < 3:
        raise SystemExit("输入只有 %d 个通道：臂 B 需要 >= 3（ch0/ch1/ch2 = linear3）" % x.shape[0])
    if args.ref_channel >= x.shape[0]:
        raise SystemExit("--ref-channel %d 超出通道数 %d" % (args.ref_channel, x.shape[0]))
    x, sr = resample(x, sr)
    wav = np.ascontiguousarray(x[args.ref_channel], dtype=np.float32)     # 波形取 ch1（参考麦）
    print("[3/4] 读到 %s：%d 通道 / %d Hz / %d 样本（%.2f 秒）-> 取 ch%d 为波形，ch0-2 算空间特征"
          % (os.path.basename(args.input), x.shape[0], sr, wav.size, wav.size / sr, args.ref_channel))

    diag = {}
    raw = sfeat.extract(x, sr=sr, diag=diag)          # [6, T//16] 原始量纲（ITD 已按几何上限夹取）
    z = sfeat.apply_scaler(raw, args.scaler)
    if raw.shape[1] != wav.size // sfeat.HOP:
        raise SystemExit("特征帧数 %d != T // %d = %d" % (raw.shape[1], sfeat.HOP, wav.size // sfeat.HOP))

    seg = int(round(args.segment_seconds * sr / sfeat.HOP)) * sfeat.HOP if args.segment_seconds > 0 else 0
    ov = int(round(args.overlap_seconds * sr / sfeat.HOP)) * sfeat.HOP
    ranges = segment_ranges(wav.size, seg, ov)
    out = np.zeros((2, wav.size), np.float32)
    wsum = np.zeros(wav.size, np.float32)
    if len(ranges) > 1:
        print("      长音频分块：%d 段（每段 %.1f s / 重叠 %.2f s）"
              % (len(ranges), seg / sr, ov / sr))
    for s, e in ranges:
        Lf = n_frames_of(e - s)
        sp = z[:, s // sfeat.HOP: s // sfeat.HOP + Lf + 1]
        y = infer(model, "B", wav[s:e], sp, device=info["device"])
        w = np.ones(e - s, np.float32)
        if s > 0:
            n = min(ov, e - s); w[:n] = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
        if e < wav.size:
            n = min(ov, e - s)
            w[-n:] = np.minimum(w[-n:], np.linspace(1.0, 0.0, n, endpoint=False, dtype=np.float32))
        out[:, s:e] += y * w
        wsum[s:e] += w
    out /= np.maximum(wsum, 1e-8)
    if not np.isfinite(out).all():
        raise SystemExit("输出含非有限值，拒绝写盘（先查输入音频）")

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    paths = []
    for i in range(out.shape[0]):
        p = os.path.join(outdir, "%s_spk%d.wav" % (stem, i + 1))
        WRITE_WAV(p, peak_norm(out[i]), sr)
        paths.append(p)
    dbg = write_debug(outdir, stem, raw, z, args.scaler, diag)
    print("[4/4] 分离完成：输出 %d 路，每路 %d 点，用时 %.1f s" % (out.shape[0], out.shape[1], time.time() - t0))
    for p in paths:
        print("      -> %s" % p)
    print("      -> %s" % dbg)
    print("      提醒：spk1/spk2 编号是任意的，先听再选；本模型不做目标人拒识。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
