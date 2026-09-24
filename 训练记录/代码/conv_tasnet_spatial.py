#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conv-TasNet + 空间特征早融合（N_SPATIAL 路）—— 训练侧实现（基于 asteroid）

在官方 Conv-TasNet 底座上，把逐帧对齐的 N_SPATIAL 个空间特征在
masker.bottleneck 的 1x1 卷积处沿通道维拼接（512 -> 512+N_SPATIAL）。
N_SPATIAL=6 时是 linear3 三对的 (0,1)(0,2)(1,2) x (ITD/ILD)。

===========================================================================
拼接点为什么必须在 GlobLN 之后（这是本文件最重要的一条）
===========================================================================
asteroid 的 masker.bottleneck = Sequential(GlobLN(), Conv1d(512, 128, 1))

GlobLN 的 mean / var 是**跨通道**统计的。如果把空间通道拼在 GlobLN 之前：
    - 归一化统计量会从 512 通道变成 512+N_SPATIAL 通道；
    - 原有 512 通道的归一化输出也随之改变；
    => 起点权重下输出就已经和官方底座不同，"差异只能来自空间线索"这句话不成立。

拼在 GlobLN 之后 + 新增 N_SPATIAL 列权重置 0：
    - 前向：新列贡献恒为 0 => 起点权重下与官方底座**逐元素恒等**（与空间特征取值无关）；
    - 反向：新列的梯度 = dL/dy * 特征^T != 0 => 空间线索照样能学进去。
代价：空间特征不经过 GlobLN。这反而是对的 —— 训练集统计量本来
就把它 z-score 标准化过了，再走一次 GlobLN 等于被全局统计量二次归一。

===========================================================================
帧对齐
===========================================================================
编码器 stride=16 / kernel=32 / padding=0  =>  L = (T - 32) // 16 + 1
特征生成侧产出 n_frames = T // 16 = L + 1（尾部多 1 帧）
本模块取前 L 帧，多出的 <= MAX_EXTRA_FRAMES 帧丢弃，超过就报错（防接错位）。

时间约定：空间特征第 t 帧的窗中心 = 16t + 7.5 样本，编码器第 t 帧中心 = 16t + 16，
相差 0.53 ms < 1 帧（窗长 32 ms），可忽略。

用法
  python conv_tasnet_spatial.py --selftest          # 全部验收，不碰训练
  python conv_tasnet_spatial.py --check-data       # 用 data/sim 真实样本端到端
"""

import argparse
import copy
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k"
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_CKPT_ROOT = os.environ.get("PRETRAINED_ROOT", os.path.join(HERE, "pretrained"))
# 空间特征通道数。1 对 (0,1) 时是 2；六通道版用 linear3 的 3 对 -> 6。
# 改这一个数会同时影响：build_spatial_model 的 bottleneck 输入通道（512 -> 512+N_SPATIAL）
# 以及所有依赖它的形状校验。老的 2 通道 ckpt 在这之后**加载会直接报形状不匹配** ——
# 这是刻意的，防止拿 2 通道权重去跑 6 通道数据而静默出错。
N_SPATIAL = 6
MAX_EXTRA_FRAMES = 2
STRIDE = 16
KERNEL = 32
SR = 16000
DATA_ROOT = os.environ.get("SIM_ROOT", os.path.join(HERE, "data", "sim"))


# ------------------------------------------------------------------ 数值策略
def set_numeric_policy(tf32=False):
    """关掉 TF32（默认关）。

    「起点权重下与官方底座逐元素恒等」这条论证依赖数值一致。但 PyTorch 默认
    `cudnn.allow_tf32 = True`：bottleneck 的输入通道变成 512+N_SPATIAL 之后，
    cuDNN 会为这两层选到**不同精度**的 kernel，单层相对差 ~3e-7，经 24 层 TCN
    放大到 ~1.2e-3（GPU 实测 max|Δ| = 255，而 |y| ~ 2.2e5）。
    关掉 TF32 后 GPU 上恢复**精确 0**。
    """
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)


set_numeric_policy(False)


# ------------------------------------------------------------------ 模型

def local_ckpt_path(ckpt=CKPT, root=None):
    """本地离线权重路径（存在才返回）。"""
    root = root or LOCAL_CKPT_ROOT
    p = os.path.join(root, ckpt.split("/")[-1], "pytorch_model.bin")
    return p if os.path.exists(p) else None


def load_pretrained(ckpt=CKPT, prefer_local=True):
    """加载预训练权重，**优先本地**（离线、不依赖 HF 缓存与镜像状态）。

    本地权重文件里装的就是 conf dict（model_name / model_args / state_dict），
    所以 torch.load -> ConvTasNet.from_pretrained(conf) 完全不触发下载。
    注意用 model(wav) 前向，不要用 separate()（model_args.sample_rate 被写成 8000，
    separate 会报错或在 resample=True 下静默降采样）。
    """
    from asteroid.models import ConvTasNet
    local = local_ckpt_path(ckpt) if prefer_local else None
    if local:
        conf = torch.load(local, map_location="cpu", weights_only=False)
        m = ConvTasNet.from_pretrained(conf)
        m.loaded_from = local
    else:
        m = ConvTasNet.from_pretrained(ckpt)
        m.loaded_from = ckpt
    m.eval()
    return m


class FusedBottleneck(nn.Sequential):
    """GlobLN -> [沿通道维拼接 N_SPATIAL 个空间特征] -> 1x1 卷积（512+N -> 128）。

    继承 nn.Sequential 而不是 nn.Module：这样子模块名仍是 "0" / "1"，
    state_dict 的 key 与原始模型完全一致，只有 bottleneck.1.weight 的形状从
    512 变 512+N_SPATIAL —— "只有那一层变了"这件事可以直接用 key 比对来验收。
    """

    def __init__(self, norm, conv, n_spatial):
        super().__init__(norm, conv)
        self.n_spatial = int(n_spatial)

    def forward(self, x, spatial=None):
        h = super().forward(x) if spatial is None or not self.n_spatial else \
            self[1](torch.cat([self[0](x), spatial], dim=1))
        return h

    @property
    def norm(self):
        return self[0]

    @property
    def conv(self):
        return self[1]


def _fit_spatial(spatial, batch, n_frames):
    """把 [N_SPATIAL, N] / [B, N_SPATIAL, N] 的空间特征对齐到编码器的 n_frames 帧。"""
    if spatial is None:
        return None
    if spatial.dim() == 2:
        spatial = spatial.unsqueeze(0)
    if spatial.dim() != 3:
        raise ValueError("空间特征维度应为 2 或 3，实际 %d" % spatial.dim())
    if spatial.shape[1] != N_SPATIAL:
        raise ValueError("空间特征通道数应为 %d，实际 %d" % (N_SPATIAL, spatial.shape[1]))
    if spatial.shape[0] == 1 and batch > 1:
        spatial = spatial.expand(batch, -1, -1)
    if spatial.shape[0] != batch:
        raise ValueError("空间特征 batch=%d 与波形 batch=%d 不一致"
                         % (spatial.shape[0], batch))
    n = spatial.shape[-1]
    if n < n_frames:
        raise ValueError("空间特征帧数 %d < 编码器帧数 %d（波形与特征不匹配）" % (n, n_frames))
    if n - n_frames > MAX_EXTRA_FRAMES:
        raise ValueError("空间特征帧数 %d 比编码器帧数 %d 多 %d 帧，超过容忍上限 %d"
                         % (n, n_frames, n - n_frames, MAX_EXTRA_FRAMES))
    return spatial[..., :n_frames]


class SpatialTDConvNet(object):
    """只在 TDConvNet.forward 的基础上，把 spatial 透传给 bottleneck。

    TDConvNet 的原始 forward 源码见 asteroid 0.7.0 的 masker.tdconvnet；
    这里逐行照抄，唯一差别是 bottleneck 多收一个 spatial 参数。
    实例化方式见 build_spatial_model()：直接把已有 masker 实例的 __class__ 换成这个类，
    权重、子模块、超参一个都不动。
    """

    def forward(self, mixture_w, spatial=None):
        batch, _, n_frames = mixture_w.size()
        sp = _fit_spatial(spatial, batch, n_frames)
        output = self.bottleneck(mixture_w, sp)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        return self.output_act(score)


class ConvTasNetSpatial(object):
    """ConvTasNet.forward 的带 spatial 版本；其余方法（forward_encoder 等）原样继承。"""

    def forward_masker(self, tf_rep, spatial=None):
        return self.masker(tf_rep, spatial)

    def forward(self, wav, spatial=None):
        squeeze = wav.dim() == 1
        if squeeze:
            wav = wav.unsqueeze(0)
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        n_samples = wav.shape[-1]
        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep, spatial)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)
        if decoded.shape[-1] > n_samples:
            decoded = decoded[..., :n_samples]
        elif decoded.shape[-1] < n_samples:
            decoded = F.pad(decoded, (0, n_samples - decoded.shape[-1]))
        return decoded.squeeze(0) if squeeze else decoded


def build_spatial_model(base=None, n_spatial=N_SPATIAL):
    """在预训练底座上做**最小改动**：bottleneck 的 1x1 卷积 512 -> 512+N_SPATIAL，新列置 0。

    注意：改造是**原地**的（直接换掉 base.masker.bottleneck 并改 __class__）。
    所以传入 base 时先 deepcopy —— 否则传进来的官方底座会被就地改造，
    后面再想拿它做对照，那就已经不是原来那个模型了。
    """
    base = load_pretrained() if base is None else copy.deepcopy(base)
    old = base.masker.bottleneck
    norm, conv = old[0], old[1]
    if not isinstance(conv, nn.Conv1d) or tuple(conv.kernel_size) != (1,):
        raise TypeError("预期 bottleneck[1] 是 1x1 Conv1d，实际 %r" % (conv,))

    new_conv = nn.Conv1d(conv.in_channels + n_spatial, conv.out_channels,
                         conv.kernel_size, stride=conv.stride, padding=conv.padding,
                         dilation=conv.dilation, groups=conv.groups,
                         bias=conv.bias is not None)
    with torch.no_grad():
        new_conv.weight.zero_()                              # 新增 N_SPATIAL 列置 0
        new_conv.weight[:, :conv.in_channels].copy_(conv.weight)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    base.masker.bottleneck = FusedBottleneck(norm, new_conv, n_spatial)
    base.masker.__class__ = type("SpatialTDConvNet", (SpatialTDConvNet, type(base.masker)), {})
    base.__class__ = type("ConvTasNetSpatial", (ConvTasNetSpatial, type(base)), {})
    return base


# ------------------------------------------------------------------ 工具

def si_sdr(est, ref, eps=1e-8):
    """Scale-invariant SDR（零均值，dB）。est / ref: [..., T]。"""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(dim=-1, keepdim=True) / (ref.pow(2).sum(dim=-1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    return 10.0 * torch.log10((target.pow(2).sum(dim=-1) + eps)
                              / (noise.pow(2).sum(dim=-1) + eps))


def pit_si_sdr_loss(est, ref):
    """PIT SI-SDR 损失（负号，越小越好）。est: [B, n_src, T]；ref: [B, n_src, T]。"""
    b, n, _ = est.shape
    scores = torch.stack([si_sdr(est[:, i], ref[:, j])
                          for i in range(n) for j in range(n)], dim=1).view(b, n, n)
    rows = list(range(n))
    from itertools import permutations
    best = None
    for perm in permutations(range(n)):
        val = sum(scores[:, i, perm[i]] for i in range(n)) / n
        best = val if best is None else torch.maximum(best, val)
    return -best.mean()


def state_dict_shapes(model):
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def diff_shapes(da, db):
    keys = sorted(set(da) | set(db))
    return [(k, da.get(k), db.get(k)) for k in keys if da.get(k) != db.get(k)]


def load_real_sample(root=DATA_ROOT, subset="main"):
    """取一条真实样本：ch1 波形 + 对应的 .space.npy。"""
    wavs = sorted(glob.glob(os.path.join(root, subset, "*.wav")))
    if not wavs:
        raise SystemExit("找不到样本：%s" % os.path.join(root, subset, "*.wav"))
    import soundfile as sf
    wav_path = wavs[0]
    x, sr = sf.read(wav_path, always_2d=True)
    feat_path = os.path.join(root, "space", subset,
                             os.path.basename(wav_path)[:-4] + ".space.npy")
    sp = np.load(feat_path)
    return wav_path, torch.from_numpy(x.T.astype(np.float32)), sr, torch.from_numpy(sp)


# ------------------------------------------------------------------ 验收

def check_shapes():
    print("[1] state_dict 形状对比（改结构前必须确认只有那一层变）")
    a = load_pretrained()
    da = state_dict_shapes(a)
    n_a = sum(p.numel() for p in a.parameters())
    b = build_spatial_model(a)
    db = state_dict_shapes(b)
    n_b = sum(p.numel() for p in b.parameters())
    d = diff_shapes(da, db)
    for k, sa, sb in d:
        print("    变化: %-28s %s -> %s" % (k, sa, sb))
    expected = {k for k, _, _ in d}
    assert expected == {"masker.bottleneck.1.weight"}, "变化的层不止 bottleneck[1]：%s" % expected
    print("    参数量 %.0f -> %.0f  (+%d, +%.5f%%)"
          % (n_a, n_b, n_b - n_a, 100.0 * (n_b - n_a) / n_a))
    assert n_b - n_a == N_SPATIAL * 128, \
        "参数增量应为 N_SPATIAL x 128 = %d" % (N_SPATIAL * 128)
    print("    PASS：只有 masker.bottleneck.1.weight 从 512 变 %d，共 +%d 个参数"
          % (512 + N_SPATIAL, N_SPATIAL * 128))
    return b


def check_forward(model=None, T=16000):
    print("[2] forward 冒烟（随机波形，%d 样本 = %.2f s）" % (T, T / SR))
    torch.manual_seed(0)
    wav = torch.randn(1, T) * 0.1
    a = load_pretrained()
    b = model if model is not None else build_spatial_model(a)
    L = (T - KERNEL) // STRIDE + 1
    sp = torch.randn(1, N_SPATIAL, L)
    with torch.no_grad():
        ya = a(wav)
        yb = b(wav, sp)
    print("    官方底座输出 %s | 本模型输出 %s | 编码器帧数 L=%d" % (tuple(ya.shape), tuple(yb.shape), L))
    assert ya.shape == yb.shape, "两者输出形状不一致"
    assert yb.dim() == 3 and yb.shape[1] == 2
    print("    PASS")


def check_identity(model=None):
    """零初始化：起点权重下必须与官方底座逐元素一致（含正对照）。"""
    print("[3] 接线自检：起点权重下与官方底座等价 + 正对照")
    T = 16000
    torch.manual_seed(0)
    wav = torch.randn(1, T) * 0.1
    a = load_pretrained()
    b = model if model is not None else build_spatial_model(a)
    L = (T - KERNEL) // STRIDE + 1
    with torch.no_grad():
        ya = a(wav)
        for label, sp in [("全零特征", torch.zeros(1, N_SPATIAL, L)),
                          ("随机特征", torch.randn(1, N_SPATIAL, L) * 5.0)]:
            yb = b(wav, sp)
            d = (ya - yb).abs().max().item()
            print("    %s：max|Δ| = %.3e" % (label, d))
            assert d == 0.0, "起点权重下与官方底座不等价（%s）=> 拼接位置或初始化有问题" % label
        # 正对照：把新列改为非零，输出必须变 —— 证明这条路真的接通了
        conv = b.masker.bottleneck.conv
        n_old = conv.in_channels - N_SPATIAL
        with torch.no_grad():
            conv.weight[:, n_old:] = 0.01
        yc = b(wav, torch.randn(1, N_SPATIAL, L))
        d2 = (ya - yc).abs().max().item()
        print("    正对照（新列置 0.01）：max|Δ| = %.3e  <- 必须 > 0" % d2)
        assert d2 > 0.0, "新列非零时输出仍不变 => 空间通道其实没接上"
        with torch.no_grad():
            conv.weight[:, n_old:].zero_()
    print("    PASS：起点权重下恒等（且与特征取值无关），新列非零时输出确实改变")


def check_data(root=DATA_ROOT):
    print("[4] 真实样本端到端（含帧对齐检查）")
    path, x, sr, sp = load_real_sample(root)
    print("    样本 %s" % os.path.basename(path))
    print("    波形 %s @ %d Hz | 空间特征 %s" % (tuple(x.shape), sr, tuple(sp.shape)))
    T = x.shape[-1]
    L = (T - KERNEL) // STRIDE + 1
    print("    编码器帧数 L = %d | 特征帧数 = %d | 差 %d" % (L, sp.shape[-1], sp.shape[-1] - L))
    assert sp.shape[-1] - L <= MAX_EXTRA_FRAMES, "帧数差超过容忍上限"
    ch1 = x[1:2].unsqueeze(0)          # 3mic 线性阵的参考通道 = ch1
    a = load_pretrained()
    b = build_spatial_model()
    b.eval()
    assert not isinstance(a, ConvTasNetSpatial), "官方底座被就地改造了"
    assert isinstance(b, ConvTasNetSpatial)
    with torch.no_grad():
        ya = a(ch1)
        yb = b(ch1, sp)
        yb_zero = b(ch1, torch.zeros_like(sp))
    print("    官方底座 %s | 本模型 %s" % (tuple(ya.shape), tuple(yb.shape)))
    print("    起点权重 max|Δ(真实特征)| = %.3e | max|Δ(零特征)| = %.3e"
          % ((ya - yb).abs().max().item(), (ya - yb_zero).abs().max().item()))
    assert torch.equal(ya, yb), "真实样本上起点权重与官方底座不等价"
    print("    PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="跑全部验收（1-4）")
    ap.add_argument("--check-shapes", action="store_true")
    ap.add_argument("--check-forward", action="store_true")
    ap.add_argument("--check-identity", action="store_true")
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--root", default=DATA_ROOT)
    args = ap.parse_args()

    print("torch %s | cuda %s" % (torch.__version__, torch.cuda.is_available()))
    print("权重来源: %s" % (local_ckpt_path() or ("HF " + CKPT)))
    do_all = args.selftest or not any([args.check_shapes, args.check_forward,
                                       args.check_identity, args.check_data])
    if do_all or args.check_shapes:
        b = check_shapes()
    else:
        b = None
    if do_all or args.check_forward:
        check_forward(b)
    if do_all or args.check_identity:
        check_identity(b)
    if do_all or args.check_data:
        check_data(args.root)
    print(">>> 全部通过")


if __name__ == "__main__":
    main()
