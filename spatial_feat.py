#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""spatial_feat.py —— 六通道空间特征提取器（ITD / ILD），单文件、不依赖 torch。

这是本模型的**输入侧**：网络吃的是 [6, L] 的空间特征，不是原始多通道波形。
口径与训练时**逐位一致**（常量、窗、限带、搜索窗、夹取顺序全部照搬训练侧的数据生成实现）。

    几何：linear3，ch0 / ch1 / ch2 间距 2 cm，**ch1 = 参考麦**，16 kHz
    六通道顺序（写死，错了不报错）：
      0 itd_01_s   1 ild_01_db   2 itd_02_s   3 ild_02_db   4 itd_12_s   5 ild_12_db
    ITD = t(ch0) - t(ch1)（> 0 表示声源偏向 ch1 一侧），单位秒，限带 2-4 kHz
    ILD = 10*log10(E_i / E_j)，全带，单位 dB
    帧率 1000 Hz：n_frames = T // 16；分析窗 512 样本、hop 16 样本，窗心对准编码器帧心
    ITD 落盘前按各自几何上限夹取（(0,1)(1,2) 58.31 us / (0,2) 116.62 us）

用法
    python spatial_feat.py --selftest                      # 合成信号自校验，不需要任何数据
    python spatial_feat.py --wav 样本.wav                  # 看一条 wav 的特征形状与量级
    python spatial_feat.py --wav 样本.wav --save-npy out.npy --stats space_feat_stats.json

被 separate.py 调用，也可单独当库用：
    from spatial_feat import extract, apply_scaler
    feat = extract(multi_ch, sr=16000)      # [6, L] float32，原始量纲（已夹取）
    feat_z = apply_scaler(feat, "space_feat_stats.json")   # z-score，喂给网络的就是这份
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------- 口径常量（与训练侧一致）
FS = 16000
HOP = 16
WIN = 512
N_FFT = 2 * WIN
BAND_HZ = (2000.0, 4000.0)
EDGE_HZ = 250.0
PAIRS = ((0, 1), (0, 2), (1, 2))
MIC_SPACING_M = 0.020
C_SOUND = 343.0
LAG_MARGIN = 3.0
GEOM_MAX_ITD_S = tuple(abs(j - i) * MIC_SPACING_M / C_SOUND for i, j in PAIRS)
MAX_LAGS = tuple(int(math.ceil(LAG_MARGIN * g * FS)) for g in GEOM_MAX_ITD_S)
MAX_LAG_SAMPLES = MAX_LAGS[0]
EDGE_ITD_S = tuple((ml - 0.1) / FS for ml in MAX_LAGS)
N_SPATIAL = 2 * len(PAIRS)
EPS = 1e-12
FEAT_NAMES = tuple("%s_%d%d_%s" % (nm, i, j, u)
                   for (i, j) in PAIRS for nm, u in (("itd", "s"), ("ild", "db")))


# ---------------------------------------------------------------- 核心算法（照搬生成侧）
def band_mask(freqs, lo, hi, edge):
    """2-4 kHz raised-cosine 软掩码。硬截断会在时域引入振铃，污染相关峰。"""
    t = np.clip((freqs - (lo - edge)) / edge, 0.0, 1.0)
    u = np.clip(((hi + edge) - freqs) / edge, 0.0, 1.0)
    return (np.sin(0.5 * np.pi * t) ** 2) * (np.sin(0.5 * np.pi * u) ** 2)


def compute_raw(x0, x1, fs=FS, win=WIN, hop=HOP, n_fft=N_FFT, band=BAND_HZ,
                edge=EDGE_HZ, max_lag=MAX_LAG_SAMPLES, chunk=4096):
    """逐帧算一对通道的 ITD / ILD。返回 float32 [2, n_frames]，n_frames = n // hop。

    全程向量化（分块堆帧一次做 FFT），不逐帧 Python 循环。
    """
    n = int(min(len(x0), len(x1)))
    n_frames = n // hop
    out = np.zeros((2, max(n_frames, 0)), np.float32)
    if n_frames <= 0:
        return out
    x0 = np.asarray(x0[:n], dtype=np.float32)
    x1 = np.asarray(x1[:n], dtype=np.float32)

    pad_l = win // 2 - hop // 2      # 窗中心对准编码器帧中心
    pad_r = win
    w = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    mask = band_mask(freqs, band[0], band[1], edge).astype(np.float32)
    lags = np.fft.fftfreq(n_fft, d=1.0) * n_fft     # 相关函数的 lag 轴（单位：样本）
    search = np.abs(lags) <= max_lag                 # 只在几何可达的 lag 范围内找峰

    x0p = np.pad(x0, (pad_l, pad_r))
    x1p = np.pad(x1, (pad_l, pad_r))
    base = np.arange(win, dtype=np.int32)

    for s in range(0, n_frames, chunk):
        e = min(s + chunk, n_frames)
        idx = (np.arange(s, e, dtype=np.int32) * hop)[:, None] + base[None, :]
        A = x0p[idx] * w
        B = x1p[idx] * w
        G = np.fft.rfft(A, n=n_fft, axis=1) * np.conj(np.fft.rfft(B, n=n_fft, axis=1))
        G /= (np.abs(G) + EPS)        # PHAT 白化
        G *= mask                     # 限带
        cc = np.fft.irfft(G, n=n_fft, axis=1)
        cc[:, ~search] = -np.inf            # 搜索窗外的 lag 不参与选峰

        # 抛物线亚样本插值：左右邻居必须从**完整** lag 轴取，并按周期回绕。
        # 早期版本从截断后的搜索数组取邻居，峰值落在 lag=0 时左邻居被夹成自身 ->
        # 分母退化 -> 恒定输出 -0.5 样本（-31 us，占 2 cm 满量程一半），肉眼看不出来。
        k = np.argmax(cc, axis=1)
        rows = np.arange(cc.shape[0])
        y0 = cc[rows, (k - 1) % n_fft]
        y1 = cc[rows, k]
        y2 = cc[rows, (k + 1) % n_fft]
        den = y0 - 2.0 * y1 + y2
        safe = np.isfinite(den) & (np.abs(den) > EPS)
        delta = np.where(safe, 0.5 * (y0 - y2) / np.where(safe, den, 1.0), 0.0)
        np.clip(delta, -1.0, 1.0, out=delta)

        out[0, s:e] = ((lags[k] + delta) / fs).astype(np.float32)
        e0 = np.einsum("ij,ij->i", A, A)
        e1 = np.einsum("ij,ij->i", B, B)
        out[1, s:e] = (10.0 * np.log10((e0 + EPS) / (e1 + EPS))).astype(np.float32)
    return out


def extract(multi_ch, sr=FS, diag=None):
    """多通道波形 -> [6, n_frames] float32 原始量纲特征（ITD 已按几何上限夹取）。

    Args:
        multi_ch: [C, T] float32，C >= 3，取前 3 路（ch0/ch1/ch2 = linear3，ch1 为参考麦）
        sr: 采样率，**必须是 16000**（重采样请在调用前做，见 separate.py）
        diag: 传一个 dict 进来，函数顺手把两个质检计数写进去（给 separate.py 打日志/画图用）：
              diag["over"][k] = 第 k 对通道**夹取前** |ITD| 超过几何上限的帧数（物理上不可能 -> 选错峰）
              diag["edge"][k] = 第 k 对通道**夹取前** |ITD| 贴到搜索窗边界的帧数（估计器失效）
              计数必须在夹取**之前**统计 —— 夹取之后恒为 0，数字会失去含义
    Returns:
        [6, T // 16] float32，未 z-score；顺序见模块开头
    """
    x = np.asarray(multi_ch)
    if x.ndim != 2:
        raise ValueError("multi_ch 应为 [C, T] 二维数组，实际 shape=%s" % (tuple(x.shape),))
    if x.shape[0] < 3:
        raise ValueError("至少需要 3 个通道（ch0/ch1/ch2 = linear3），实际 %d 通道" % x.shape[0])
    if int(sr) != FS:
        raise ValueError("extract() 只接受 16 kHz，实际 %s；请先重采样" % (sr,))
    x = np.ascontiguousarray(x[:3], dtype=np.float32)   # [C, T]：行 = 通道，列 = 样点
    feat = np.concatenate([compute_raw(x[i], x[j], max_lag=ml)
                           for (i, j), ml in zip(PAIRS, MAX_LAGS)], axis=0)
    n_over, n_edge = [], []
    for k in range(len(PAIRS)):
        v = feat[2 * k].astype(np.float64)
        lim = GEOM_MAX_ITD_S[k]
        n_over.append(int(np.sum(np.abs(v) > lim)))              # 夹取前越界帧（诊断）
        n_edge.append(int(np.sum(np.abs(v) >= EDGE_ITD_S[k])))   # 夹取前贴到搜索窗边界
        np.clip(v, -lim, lim, out=v)
        feat[2 * k] = v.astype(np.float32)
    if diag is not None:
        diag["over"] = n_over
        diag["edge"] = n_edge
    return feat.astype(np.float32)


def apply_scaler(feat, stats_path):
    """按 stats 文件做 z-score（逐通道 (raw - mean) / std），顺序与训练侧完全一致。"""
    with open(stats_path, encoding="utf-8") as fh:
        st = json.load(fh)
    out = np.empty_like(np.asarray(feat, dtype=np.float32))
    for i, nm in enumerate(FEAT_NAMES):
        m = float(st["raw_mean"][nm])
        s = float(st["raw_std"][nm])
        if not np.isfinite(m) or not np.isfinite(s) or s == 0.0:
            raise ValueError("标尺 %s 里 %s 的 mean/std 非法：%r / %r" % (stats_path, nm, m, s))
        out[i] = ((feat[i].astype(np.float64) - m) / s).astype(np.float32)
    return out


# ---------------------------------------------------------------- 自校验（不需要任何数据）
def band_noise(n, fs, lo, hi, rng):
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    m = (f >= lo) & (f <= hi)
    X = np.zeros(len(f), dtype=np.complex128)
    X[m] = rng.normal(size=int(m.sum())) + 1j * rng.normal(size=int(m.sum()))
    x = np.fft.irfft(X, n=n)
    return x / (np.max(np.abs(x)) + EPS)


def frac_delay(x, delay_samples, fs):
    """频域相位斜坡实现任意分数样本时延：y[t] = x[t - delay]。"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * delay_samples / fs), n=len(x))


def selftest(tol_samples=0.05, tol_db=0.001, seed=0):
    """合成信号核验：已知时延 / 已知增益差 / 符号约定 / 长基线搜索窗。

    最关键的是 A：它不需要任何数据，专抓"插值在 lag=0 处退化、恒定偏 -0.5 样本"
    那个坑 —— -0.5 样本 = -31 us，而 2 cm 阵列满量程才 58 us。

    判据 tol_samples 取 0.05 样本，与生成侧（sim/spatial_feat.py）的自校验阈值一致：
    抛物线插值在 1/4 样本处有约 0.028 样本的系统偏差（实测最坏 0.0284），这是算法
    本身的偏差、两侧同源，不是实现错；追求更紧的阈值只会得到"永远 FAIL"。
    """
    rng = np.random.default_rng(seed)
    n = FS * 4
    x = band_noise(n, FS, 2000.0, 4000.0, rng)
    ok = True
    worst_itd, worst_ild = 0.0, 0.0
    print("自校验 A：已知分数时延 -> ITD（正值 = ch0 更晚）；判据 |误差| <= %.3f 样本" % tol_samples)
    for d in (0.0, 0.25, -0.25, 0.5, -0.5, 0.933, -0.933):
        est = float(np.median(compute_raw(frac_delay(x, d, FS), x)[0]))
        err = (est - d / FS) * FS
        worst_itd = max(worst_itd, abs(err))
        flag = "OK " if abs(err) <= tol_samples else "BAD"
        ok &= abs(err) <= tol_samples
        print("   真值 %+6.3f 样本 -> 估计 %+6.3f 样本  误差 %+6.4f 样本  %s" % (d, est * FS, err, flag))

    print("自校验 B：已知通道增益差 -> ILD；判据 |误差| <= %.3f dB" % tol_db)
    for g in (0.0, 3.0, -3.0, 6.0):
        gain = 10.0 ** (g / 20.0)
        est = float(np.median(compute_raw(gain * x, x)[1]))
        err = est - g
        worst_ild = max(worst_ild, abs(err))
        flag = "OK " if abs(err) <= tol_db else "BAD"
        ok &= abs(err) <= tol_db
        print("   真值 %+5.1f dB -> 估计 %+5.2f dB  误差 %+5.3f dB  %s" % (g, est, err, flag))

    print("自校验 C：符号约定（az=+90 度应得正 ITD）")
    a, off = 90.0, 0.02
    tau = (np.hypot(3.0 * np.cos(np.deg2rad(a)), 3.0 * np.sin(np.deg2rad(a)) + off) - 3.0) / 343.0
    est = float(np.median(compute_raw(frac_delay(x, tau * FS, FS), x)[0]))
    flag = "OK " if abs(est - tau) * FS <= tol_samples else "BAD"
    ok &= abs(est - tau) * FS <= tol_samples
    print("   几何真值 %+.3f 样本 (%+.2f us) -> 估计 %+.3f 样本  %s" % (tau * FS, tau * 1e6, est * FS, flag))

    print("自校验 D：长基线 (0,2) 4 cm —— 几何上限 %.3f 样本，核验 +-%d 搜索窗"
          % (GEOM_MAX_ITD_S[1] * FS, MAX_LAGS[1]))
    for d in (1.866, -1.866, 1.0, -1.0):
        est = float(np.median(compute_raw(frac_delay(x, d, FS), x, max_lag=MAX_LAGS[1])[0]))
        err = (est - d / FS) * FS
        worst_itd = max(worst_itd, abs(err))
        flag = "OK " if abs(err) <= tol_samples else "BAD"
        ok &= abs(err) <= tol_samples
        print("   真值 %+6.3f 样本 -> 估计 %+6.3f 样本  误差 %+6.4f 样本  %s" % (d, est * FS, err, flag))

    sh = np.zeros((3, n), np.float32)
    sh[0] = frac_delay(x, 0.5, FS)
    sh[1] = x
    sh[2] = frac_delay(x, -0.5, FS)
    f = extract(sh, sr=FS)
    itd_res = np.abs(f[2] - (f[0] + f[4]))      # t02 - t01 - t12
    ild_res = np.abs(f[3] - (f[1] + f[5]))      # ILD_02 - ILD_01 - ILD_12
    print("自校验 E：通道装配恒等式（ITD_02 = ITD_01 + ITD_12；ILD_02 = ILD_01 + ILD_12）")
    print("   ITD 残差 中位数 %.3e s，最大 %.3e s" % (np.median(itd_res), itd_res.max()))
    print("   ILD 残差 中位数 %.3e dB，最大 %.3e dB" % (np.median(ild_res), ild_res.max()))
    ok &= itd_res.max() < 1e-5 and ild_res.max() < 1e-4
    print("自校验 F：形状契约  n_frames == T // 16 -> %d == %d  %s"
          % (f.shape[1], n // HOP, "OK" if f.shape[1] == n // HOP else "BAD"))
    ok &= f.shape == (N_SPATIAL, n // HOP)
    print("自校验总体：%s（ITD 最大误差 %.4f 样本 / ILD 最大误差 %.4f dB；判据 %.3f 样本 / %.3f dB）"
          % ("PASS" if ok else "FAIL", worst_itd, worst_ild, tol_samples, tol_db))
    return ok


def _read_wav(path):
    """读多通道 wav -> (float32 [C, T], sr)。优先 soundfile，没有就用 scipy。"""
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return np.ascontiguousarray(x.T), int(sr)
    except ImportError:
        from scipy.io import wavfile
        sr, x = wavfile.read(path)
        x = np.asarray(x)
        if x.dtype.kind == "i":
            x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
        if x.ndim == 1:
            x = x[:, None]
        return np.ascontiguousarray(x.T.astype(np.float32)), int(sr)


def main():
    ap = argparse.ArgumentParser(description="六通道空间特征（ITD/ILD）提取器")
    ap.add_argument("--selftest", action="store_true", help="合成信号自校验（不需要数据）")
    ap.add_argument("--wav", default=None, help="看一条多通道 wav 的特征")
    ap.add_argument("--save-npy", default=None, help="把 [6, L] 特征存成 npy")
    ap.add_argument("--stats", default=None, help="给了就顺带做 z-score（--save-npy 存的是 z 版）")
    args = ap.parse_args()

    if args.selftest or not args.wav:
        return 0 if selftest() else 1

    x, sr = _read_wav(args.wav)
    print("读入 %s：%d 通道 / %d Hz / %d 样本（%.2f 秒）" % (args.wav, x.shape[0], sr, x.shape[1], x.shape[1] / sr))
    if sr != FS:
        raise SystemExit("采样率应为 %d Hz，实际 %d：请先重采样（separate.py 会自动做）" % (FS, sr))
    f = extract(x, sr=sr)
    print("特征 [%d, %d]，逐通道 raw 统计：" % f.shape)
    for i, nm in enumerate(FEAT_NAMES):
        print("   %-10s mean=%+.6e  std=%.6e  min=%+.3e  max=%+.3e"
              % (nm, f[i].mean(), f[i].std(), f[i].min(), f[i].max()))
    assert f.shape[0] == N_SPATIAL, "特征通道数错"
    assert f.shape[1] == x.shape[1] // HOP, "帧数应为 T // %d" % HOP
    if args.stats:
        f = apply_scaler(f, args.stats)
        print("已按 %s 做 z-score（这一份才是网络输入）" % os.path.basename(args.stats))
    if args.save_npy:
        np.save(args.save_npy, f)
        print("已写出 %s（%s）" % (args.save_npy, tuple(f.shape)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
