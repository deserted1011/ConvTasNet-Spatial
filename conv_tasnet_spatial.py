#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conv_tasnet_spatial.py —— 臂 B（空间特征早融合 Conv-TasNet）网络定义，纯 PyTorch。

这个文件在本包里扮演的角色，和 HuggingFace 模型目录里的「建模代码」完全一样：
    读同目录的 config.json  ->  按配方把网络搭出来  ->  灌入 臂B_模型.pt 的权重
**不需要 asteroid、不需要联网、不需要任何外部权重文件。** 装上 torch + numpy 就能加载。

为什么要自带一份，而不是 import asteroid
  1. 交付件要能独立跑：官方那个模型要靠 `asteroid` 才能 `from_pretrained`，
     这里把建模代码直接随包发出，少一层依赖、少一个版本坑（asteroid 0.7.0 之后
     的版本对 ConvTasNet 有过改动，撞上就不一定能加载）。
  2. 结构写在这里才可核对：config.json 是配方，本文件是照配方施工的代码；
     两者对不上会当场抛形状错误，而不是静默算错。
  3. 本项目对官方结构只改了**一处**（见下），自带实现能把这一处写清楚、可验收。

与官方（asteroid 0.7.0）的关系
  本文件是 asteroid 0.7.0 的 `ConvTasNet` / `TDConvNet` / `Conv1DBlock` / `GlobLN`
  / `FreeFB` / `Encoder` / `Decoder` 的逐行等价实现，唯一的改动是 bottleneck：
      官方：GlobLN(512) -> Conv1d(512 -> 128, k=1)
      本包：GlobLN(512) -> [沿通道维拼上 6 路空间特征] -> Conv1d(518 -> 128, k=1)
  除这一层的输入通道数 512 -> 518 外，编码器 / TCN / 解码器 / mask 头一字未改。
  加载 臂B_模型.pt 时 345 个张量**全部**被覆盖，官方预训练权重一个数都不参与运算。

权重张量命名（345 个，与 臂B_模型.pt 严格一一对应）
  encoder.filterbank._filters                  (512, 1, 32)   编码器滤波器组（可学）
  masker.bottleneck.0.gamma / .beta            (512,)         全局层归一化
  masker.bottleneck.1.weight / .bias           (128, 518, 1) / (128,)
  masker.TCN.{0..23}.shared_block.{0..5}.*     24 个 TCN 块 = 3 repeats x 8 blocks
  masker.TCN.{i}.res_conv / .skip_conv         (128, 512, 1) / (128,)
  masker.mask_net.0.weight                     (1,)           PReLU 斜率
  masker.mask_net.1.weight / .bias             (1024, 128, 1) / (1024,)   2 路 x 512
  decoder.filterbank._filters                  (512, 1, 32)   解码器滤波器组
  （注意 encoder / decoder 的滤波器组是**两份独立参数**，不是共享的 —— 训练确实是分开学的）

帧数契约（差一帧不会报错，只会静默错位，所以这里写成显式校验）
  波形 T 个样点 -> 编码器出 Lf = (T - 32) // 16 + 1 帧
  空间特征按 T // 16 帧给出（比 Lf 多 1 帧）-> 送进网络前必须切到前 Lf 帧

用法（当库）
    from conv_tasnet_spatial import load_model
    model = load_model()                      # 读 config.json + 臂B_模型.pt
    est = model(wav_tensor, spatial_tensor)   # [2, T]

用法（自查）
    python conv_tasnet_spatial.py --check      # 按 config.json 建网 + 载权重 + 过一遍前向
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))

# 兜底常量：正常路径下一切以 config.json 为准，这里只用于 config 缺字段时的报错信息
N_SPATIAL = 6
MAX_EXTRA_FRAMES = 2
STRIDE = 16
KERNEL = 32
SR = 16000
DEFAULT_CONFIG = os.path.join(HERE, "config.json")
DEFAULT_WEIGHTS = os.path.join(HERE, "臂B_模型.pt")


def set_numeric_policy(tf32=False):
    """关掉 TF32（默认关）。

    TF32 只影响 GPU：它会把 fp32 卷积偷偷降成 tf32（10 位尾数）。对本模型来说，
    臂 A / 臂 B 的 bottleneck 输入通道不同（512 / 518），cuDNN 会为它们挑到不同精度的
    kernel，单层相对差 ~3e-7，经 24 层 TCN 放大到 ~1e-3。关掉之后 GPU 上结果可复现，
    与 CPU 结果一致。见项目 docs/decision_log.md 的 P-35。
    """
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)


set_numeric_policy(False)


# ------------------------------------------------------------------ 基础层
class GlobLN(nn.Module):
    """全局层归一化（对应 asteroid 的 `_glob_norm` + 逐通道仿射 gamma/beta）。

    在「除 batch 外的所有维度」上求均值方差，所以一帧归一化时会同时用到频率轴与时间轴；
    这正是 gLN 与 chanLN 的区别，也是官方 Conv-TasNet 在非因果配置下的默认。
    """

    def __init__(self, chan):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(chan))
        self.beta = nn.Parameter(torch.zeros(chan))

    def forward(self, x, eps=1e-8):
        dims = list(range(1, x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.var(x, dim=dims, keepdim=True, unbiased=False)
        h = (x - mean) / torch.sqrt(var + eps)
        return (self.gamma * h.transpose(1, -1) + self.beta).transpose(1, -1)


class FreeFB(nn.Module):
    """无约束滤波器组（对应 asteroid `FreeFB`）：一组自由学习的 conv1d 核。

    参数名 `_filters` 与官方权重文件里的键名一致，所以权重可以直接灌进来。
    """

    def __init__(self, n_filters, kernel_size):
        super().__init__()
        self.n_filters = int(n_filters)
        self.kernel_size = int(kernel_size)
        self._filters = nn.Parameter(torch.ones(n_filters, 1, kernel_size))

    def filters(self):
        return self._filters


class _Analysis(nn.Module):
    """编码器：波形 [B, 1, T] -> 表示 [B, n_filters, Lf]。

    等价 nn.Conv1d(1, n_filters, kernel_size, stride, padding=0, bias=False)。
    padding=0 是官方的设定（asteroid Encoder(padding=0)），所以 Lf = (T - kernel) // stride + 1，
    也就是空间特征比它多算出来的那 1 帧必须丢掉。
    """

    def __init__(self, n_filters, kernel_size, stride):
        super().__init__()
        self.filterbank = FreeFB(n_filters, kernel_size)
        self.stride = int(stride)

    def forward(self, wav):
        if wav.dim() != 3 or wav.shape[1] != 1:
            raise ValueError("编码器只接受 [B, 1, T] 单通道波形，实际 %s" % (tuple(wav.shape),))
        return F.conv1d(wav, self.filterbank.filters(), stride=self.stride, padding=0)


class _Synthesis(nn.Module):
    """解码器：表示 [B, n_filters, Lf] -> 波形 [B, 1, (Lf - 1) * stride + kernel]。

    等价 nn.ConvTranspose1d（重叠相加），padding=0 / output_padding=0，与官方一致。
    """

    def __init__(self, n_filters, kernel_size, stride):
        super().__init__()
        self.filterbank = FreeFB(n_filters, kernel_size)
        self.stride = int(stride)

    def forward(self, spec):
        f = self.filterbank.filters()
        if spec.dim() == 3:                       # [B, freq, time]
            return F.conv_transpose1d(spec, f, stride=self.stride, padding=0)
        if spec.dim() == 4:                       # [B, n_src, freq, time]：按官方做法把前面几维折进 batch
            out = F.conv_transpose1d(spec.reshape((-1,) + tuple(spec.shape[-2:])), f,
                                     stride=self.stride, padding=0)
            return out.view(tuple(spec.shape[:-2]) + (-1,))
        raise ValueError("解码器输入维度应为 3 或 4，实际 %d" % spec.dim())


class Conv1DBlock(nn.Module):
    """TCN 里的一个卷积块（对应 asteroid `Conv1DBlock`）。

    shared_block 的 6 个子层序号与原实现一致（0..5），这样权重键名能对上。
    第一层深度卷积用 groups=hid_chan（depthwise），膨胀率按 2**block_index 指数增长。
    """

    def __init__(self, in_chan, hid_chan, skip_out_chan, kernel_size, padding, dilation):
        super().__init__()
        self.shared_block = nn.Sequential(
            nn.Conv1d(in_chan, hid_chan, 1),
            nn.PReLU(),
            GlobLN(hid_chan),
            nn.Conv1d(hid_chan, hid_chan, kernel_size, padding=padding,
                      dilation=dilation, groups=hid_chan),
            nn.PReLU(),
            GlobLN(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        h = self.shared_block(x)
        return self.res_conv(h), self.skip_conv(h)


class FusedBottleneck(nn.Sequential):
    """GlobLN -> [拼空间特征] -> 1x1 卷积。**本包与官方唯一的差别就在这一层。**

    继承 nn.Sequential 是为了让子模块名仍然是 "0" / "1"，权重键名与官方完全一致；
    只有 bottleneck.1.weight 的输入维从 512 变成 518。
    """

    def __init__(self, norm, conv, n_spatial):
        super().__init__(norm, conv)
        self.n_spatial = int(n_spatial)

    @property
    def norm(self):
        return self[0]

    @property
    def conv(self):
        return self[1]

    def forward(self, x, spatial=None):
        h = self[0](x)
        if spatial is None or not self.n_spatial:
            return self[1](h)
        if spatial.shape[-1] != h.shape[-1]:
            raise ValueError("空间特征帧数 %d 与瓶颈帧数 %d 不一致"
                             % (spatial.shape[-1], h.shape[-1]))
        return self[1](torch.cat([h, spatial], dim=1))


def _fit_spatial(spatial, batch, n_frames, n_spatial):
    """把 [N, L] / [B, N, L] 的空间特征对齐到编码器的 n_frames 帧（尾部多余的丢掉）。"""
    if spatial is None:
        return None
    if spatial.dim() == 2:
        spatial = spatial.unsqueeze(0)
    if spatial.dim() != 3:
        raise ValueError("空间特征维度应为 2 或 3，实际 %d" % spatial.dim())
    if spatial.shape[1] != n_spatial:
        raise ValueError("空间特征通道数应为 %d，实际 %d（顺序见 config.json 的 spatial_contract）"
                         % (n_spatial, spatial.shape[1]))
    if spatial.shape[0] == 1 and batch > 1:
        spatial = spatial.expand(batch, -1, -1)
    if spatial.shape[0] != batch:
        raise ValueError("空间特征 batch=%d 与波形 batch=%d 不一致" % (spatial.shape[0], batch))
    n = spatial.shape[-1]
    if n < n_frames:
        raise ValueError("空间特征帧数 %d < 编码器帧数 %d（波形与特征不是同一条录音）" % (n, n_frames))
    if n - n_frames > MAX_EXTRA_FRAMES:
        raise ValueError("空间特征帧数 %d 比编码器帧数 %d 多 %d 帧，超过容忍上限 %d"
                         % (n, n_frames, n - n_frames, MAX_EXTRA_FRAMES))
    return spatial[..., :n_frames]


def _as_norm(name):
    if name in ("gLN", "GlobLN", "glb"):
        return GlobLN
    raise ValueError("本交付件只实现官方配置用的 gLN，config 里给的是 %r" % (name,))


def _as_act(name):
    table = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh,
             "softmax": lambda: nn.Softmax(dim=1), None: nn.Identity}
    if name not in table:
        raise ValueError("不支持的激活 %r" % (name,))
    return table[name]()


# ------------------------------------------------------------------ 网络
class TDConvNetSpatial(nn.Module):
    """Conv-TasNet 的 mask 网络（TDConvNet）+ 空间特征早融合。

    与官方 TDConvNet.forward 逐行等价，唯一区别是 bottleneck 多收一个 spatial 参数。
    """

    def __init__(self, in_chan, n_src, out_chan=None, n_blocks=8, n_repeats=3, bn_chan=128,
                 hid_chan=512, skip_chan=128, conv_kernel_size=3, norm_type="gLN",
                 mask_act="relu", n_spatial=N_SPATIAL, causal=False):
        super().__init__()
        if causal:
            raise ValueError("本交付件不含因果版实现（官方配置 causal=False）")
        norm_cls = _as_norm(norm_type)
        self.in_chan = int(in_chan)
        self.n_src = int(n_src)
        self.out_chan = int(out_chan) if out_chan else int(in_chan)
        self.n_blocks = int(n_blocks)
        self.n_repeats = int(n_repeats)
        self.bn_chan = int(bn_chan)
        self.hid_chan = int(hid_chan)
        self.skip_chan = int(skip_chan)
        self.conv_kernel_size = int(conv_kernel_size)
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = False
        self.n_spatial = int(n_spatial)

        bottleneck_conv = nn.Conv1d(self.in_chan + self.n_spatial, self.bn_chan, 1)
        self.bottleneck = FusedBottleneck(norm_cls(self.in_chan), bottleneck_conv, self.n_spatial)

        self.TCN = nn.ModuleList()
        for _ in range(self.n_repeats):
            for x in range(self.n_blocks):
                padding = (self.conv_kernel_size - 1) * 2 ** x // 2
                self.TCN.append(Conv1DBlock(self.bn_chan, self.hid_chan, self.skip_chan,
                                            self.conv_kernel_size, padding=padding,
                                            dilation=2 ** x))

        mask_conv_inp = self.skip_chan if self.skip_chan else self.bn_chan
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(mask_conv_inp, self.n_src * self.out_chan, 1),
        )
        self.output_act = _as_act(mask_act)

    def forward(self, mixture_w, spatial=None):
        batch, _, n_frames = mixture_w.size()
        sp = _fit_spatial(spatial, batch, n_frames, self.n_spatial)
        output = self.bottleneck(mixture_w, sp)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            residual, skip = layer(output)
            skip_connection = skip_connection + skip
            output = output + residual
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        return self.output_act(score)


class ConvTasNetSpatial(nn.Module):
    """编码器 - mask 网络 - 解码器 三件套（对应官方 `BaseEncoderMaskerDecoder.forward`）。

    输入 1D / 2D / 3D 波形都行：
        1D [T]       -> 输出 [n_src, T]
        2D [B, T]    -> 输出 [B, n_src, T]
        3D [B, 1, T] -> 输出 [B, n_src, T]
    输出长度与输入样点数严格对齐（多则截、少则补零，与官方 pad_x_to_y 一致）。
    """

    def __init__(self, n_src=2, out_chan=None, n_blocks=8, n_repeats=3, bn_chan=128,
                 hid_chan=512, skip_chan=128, conv_kernel_size=3, norm_type="gLN",
                 mask_act="relu", in_chan=512, causal=False, kernel_size=32, n_filters=512,
                 stride=16, encoder_activation=None, n_spatial=N_SPATIAL, sample_rate=SR):
        super().__init__()
        self.encoder = _Analysis(n_filters, kernel_size, stride)
        self.decoder = _Synthesis(n_filters, kernel_size, stride)
        self.enc_activation = _as_act(encoder_activation)
        self.masker = TDConvNetSpatial(in_chan=n_filters, n_src=n_src, out_chan=out_chan,
                                       n_blocks=n_blocks, n_repeats=n_repeats, bn_chan=bn_chan,
                                       hid_chan=hid_chan, skip_chan=skip_chan,
                                       conv_kernel_size=conv_kernel_size, norm_type=norm_type,
                                       mask_act=mask_act, n_spatial=n_spatial, causal=causal)
        self.n_spatial = int(n_spatial)
        self.n_src = int(n_src)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.sample_rate = sample_rate

    def n_frames_of(self, n_samples):
        return n_frames_of(n_samples, self.kernel_size, self.stride)

    def forward(self, wav, spatial=None):
        squeeze = wav.dim() == 1
        if squeeze:
            wav = wav.reshape(1, 1, -1)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)
        if wav.dim() != 3:
            raise ValueError("波形维度应为 1 / 2 / 3，实际 %d" % wav.dim())
        n_samples = int(wav.shape[-1])
        tf_rep = self.enc_activation(self.encoder(wav))
        est_masks = self.masker(tf_rep, spatial)
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        decoded = self.decoder(masked_tf_rep)
        if decoded.shape[-1] > n_samples:
            decoded = decoded[..., :n_samples]
        elif decoded.shape[-1] < n_samples:
            decoded = F.pad(decoded, [0, n_samples - decoded.shape[-1]])
        return decoded.squeeze(0) if squeeze else decoded


def n_frames_of(n_samples, kernel=KERNEL, stride=STRIDE):
    """长度 n_samples 的波形对应的编码器帧数（帧数契约的唯一出处）。"""
    return (int(n_samples) - int(kernel)) // int(stride) + 1


# ------------------------------------------------------------------ 配方 -> 模型
def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    for key in ("model_name", "model_args", "weights_file"):
        if key not in cfg:
            raise ValueError("结构配方 %s 缺少字段 %r" % (path, key))
    return cfg


def build_model(cfg, n_spatial=None):
    """按 config.json 搭一个**随机初始化**的网络（权重随后载入）。"""
    a = dict(cfg["model_args"])
    patch = cfg.get("arm_b_patch", {})
    if n_spatial is None:
        n_spatial = patch.get("spatial_channels", N_SPATIAL)
    fb = str(a.pop("fb_name", "")).lower()
    if fb not in ("freefb", "free"):
        raise ValueError("本交付件只实现官方配置用的 FreeFB 滤波器组，config 里给的是 %r" % (fb,))
    n_blocks = int(a.pop("n_blocks", 8))
    n_repeats = int(a.pop("n_repeats", 3))
    return ConvTasNetSpatial(
        n_src=int(a.pop("n_src", 2)),
        out_chan=a.pop("out_chan", None),
        n_blocks=n_blocks,
        n_repeats=n_repeats,
        bn_chan=int(a.pop("bn_chan", 128)),
        hid_chan=int(a.pop("hid_chan", 512)),
        skip_chan=int(a.pop("skip_chan", 128)),
        conv_kernel_size=int(a.pop("conv_kernel_size", 3)),
        norm_type=a.pop("norm_type", "gLN"),
        mask_act=a.pop("mask_act", "relu"),
        in_chan=int(a.pop("in_chan", 512)),
        causal=bool(a.pop("causal", False)),
        kernel_size=int(a.pop("kernel_size", 32)),
        n_filters=int(a.pop("n_filters", 512)),
        stride=int(a.pop("stride", 16)),
        encoder_activation=a.pop("encoder_activation", None),
        n_spatial=int(n_spatial),
        # 官方配方里的 sample_rate 是 8000（写错的值，见 config.json 的 _model_args_note）；
        # 它不参与任何计算，只是记录，所以直接沿用原值，不在这里"顺手修正"。
        sample_rate=a.pop("sample_rate", SR),
    )


def _torch_load(path):
    """兼容各版 torch：新版本默认 weights_only=True，老权重里可能有非张量对象。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                      # torch < 1.13 没有 weights_only 参数
        return torch.load(path, map_location="cpu")
    except Exception:                      # 权重里含非张量对象
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(blob):
    """把「纯 state_dict」与「训练 checkpoint」两种格式统一成 state_dict。"""
    if isinstance(blob, dict) and "model" in blob and isinstance(blob["model"], dict):
        return blob["model"], blob
    if isinstance(blob, dict):
        return blob, {}
    raise ValueError("权重文件格式不认识：%r" % type(blob))


def load_state_dict_strict(model, sd, source=""):
    """严格加载：缺键 / 多键 / 形状不符 -> 当场报错，绝不静默丢权重。"""
    cur = model.state_dict()
    missing = [k for k in cur if k not in sd]
    extra = [k for k in sd if k not in cur]
    bad = [(k, tuple(sd[k].shape), tuple(cur[k].shape))
           for k in sd if k in cur and tuple(sd[k].shape) != tuple(cur[k].shape)]
    if missing or extra or bad:
        raise RuntimeError(
            "权重与网络结构不匹配（%s）：缺 %d 键 / 多 %d 键 / 形状不符 %d 处；"
            "前几处形状冲突 %s" % (source, len(missing), len(extra), len(bad), bad[:3]))
    model.load_state_dict(sd)
    return len(cur)


def load_model(config=None, weights=None, device="cpu", n_spatial=None):
    """一把梭：读配方 -> 建网 -> 严格载权重 -> 回到 eval 模式。

    Returns:
        (model, info)；info 里有 num_params / n_spatial / weights / config 等，供日志与追溯。
    """
    config = config or DEFAULT_CONFIG
    cfg = load_config(config)
    if weights is None:
        weights = os.path.join(os.path.dirname(os.path.abspath(config)), cfg["weights_file"])
    model = build_model(cfg, n_spatial=n_spatial)
    sd, meta = extract_state_dict(_torch_load(weights))
    n_keys = load_state_dict_strict(model, sd, source=os.path.basename(weights))
    dev = str(device)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[conv_tasnet_spatial] CUDA 不可用，回退 CPU")
        dev = "cpu"
    model.to(torch.device(dev)).eval()
    info = {
        "config": os.path.abspath(config),
        "weights": os.path.abspath(weights),
        "n_keys": n_keys,
        "n_spatial": int(model.n_spatial),
        "num_params": int(sum(p.numel() for p in model.parameters())),
        "device": dev,
        "epoch": meta.get("epoch"),
        "sample_rate": int(model.sample_rate) if model.sample_rate else None,
    }
    return model, info


# ------------------------------------------------------------------ 自查
def check():
    """按 config.json 建网、载权重、过一遍前向，打印逐项结论。"""
    import numpy as np
    cfg = load_config()
    model, info = load_model()
    print("config      : %s" % info["config"])
    print("weights     : %s" % info["weights"])
    print("参数量      : %d（%.4f M）｜ 空间特征通道 %d ｜ 张量数 %d"
          % (info["num_params"], info["num_params"] / 1e6, info["n_spatial"], info["n_keys"]))
    expect = int(cfg.get("num_params", 0))
    ok = info["num_params"] == expect
    print("参数与配方一致：%d == %d  %s" % (info["num_params"], expect, "OK" if ok else "BAD"))
    T = 16000
    wav = torch.from_numpy(np.random.RandomState(0).randn(1, T).astype("float32"))
    Lf = n_frames_of(T)
    sp = torch.from_numpy(np.random.RandomState(1).randn(1, info["n_spatial"], T // 16).astype("float32"))
    with torch.no_grad():
        y = model(wav, sp)
    shape_ok = tuple(y.shape) == (1, int(cfg["model_args"]["n_src"]), T)
    print("前向形状    : %s（期望 (1, %d, %d)）  %s"
          % (tuple(y.shape), int(cfg["model_args"]["n_src"]), T, "OK" if shape_ok else "BAD"))
    print("编码器帧数  : Lf = (T - 32) // 16 + 1 = %d；空间特征 %d 帧（多 1 帧，内部自动丢弃）"
          % (Lf, T // 16))
    # 帧数闸：多给 3 帧必须报错（多 1 帧是正常的，多 3 帧说明配错数据）
    try:
        with torch.no_grad():
            model(wav, torch.from_numpy(np.random.randn(1, info["n_spatial"], T // 16 + 3).astype("float32")))
        print("帧数闸      : BAD（多 3 帧竟然没报错）")
        ok = False
    except ValueError:
        print("帧数闸      : OK（空间特征多 3 帧被拒绝）")
    # 通道数闸：把 6 通道切成 5 通道必须报错
    try:
        with torch.no_grad():
            model(wav, torch.from_numpy(np.random.randn(1, info["n_spatial"] - 1, T // 16).astype("float32")))
        print("通道数闸    : BAD（通道数不对竟然没报错）")
        ok = False
    except ValueError:
        print("通道数闸    : OK（空间特征通道数不对被拒绝）")
    print("自查总体    : %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="臂 B 网络定义（纯 PyTorch，自描述）")
    ap.add_argument("--check", action="store_true", help="建网 + 载权重 + 前向自查")
    args = ap.parse_args()
    return check() if args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())
