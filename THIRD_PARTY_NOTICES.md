# 第三方组件、许可与署名

本仓库**按部分适用不同许可**。逐件对照如下，许可原文与署名一并列出，便于直接复用。

| 部分 | 来源 | 许可 |
|---|---|---|
| `conv_tasnet_spatial.py`、`训练记录/代码/conv_tasnet_spatial.py` 里的网络定义 | asteroid 0.7.0 | **MIT** |
| 本仓库其余原创代码与文档 | 拾音客 | **BSD-3-Clause**（根目录 [`LICENSE`](LICENSE)） |
| `训练记录/代码/pretrained/ConvTasNet_Libri2Mix_sepclean_16k/pytorch_model.bin`（原样再分发） | Cosentino Joris | **CC BY-SA 4.0** |
| `臂B_模型.pt`、`训练记录/ckpt_B_ep30.pt`（微调权重与训练检查点，**改编物**） | 拾音客（改编自上一行） | **CC BY-SA 4.0** |
| `素材/` 中的 6 条样本（**已做修改**：干声 + 自建 RIR 仿真混合） | LibriSpeech + 拾音客 | 语音 **CC BY 4.0**；RIR 自建 |
| 房间冲激响应（RIR）的仿真方法 | pyroomacoustics | **MIT** |
| 算法 | Conv-TasNet（Luo & Mesgarani, TASLP 2019） | 论文 |

一句话口径：**代码分两种（原创 BSD-3-Clause / 移植自 asteroid 的 MIT），权重分两种（起点权重与微调权重都是 CC BY-SA 4.0），素材里的语音是 CC BY 4.0。**

---

## 1. asteroid —— MIT

`conv_tasnet_spatial.py`（纯 PyTorch 版）与留档的 `训练记录/代码/conv_tasnet_spatial.py`（训练侧版本）
中的 Conv-TasNet 网络定义（`FreeFB` / `Conv1DBlock` / `TCN` / `Masker` / `SeparationModel` 等）
均来自 [asteroid](https://github.com/asteroid-team/asteroid) 0.7.0 的对应模块：前者是它的**逐行等价移植**
（目的是去掉 `asteroid` 依赖、只留 `torch`），后者在训练时直接复用其实现。
因此这部分代码按 asteroid 的 MIT 许可使用与再分发。

```
MIT License

Copyright (c) 2019 Pariente Manuel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

> 该 MIT 许可只覆盖上述移植而来的网络定义；本仓库其余代码与文档是原创的，按 BSD-3-Clause 授权。

---

## 2. 起点权重与微调权重 —— CC BY-SA 4.0

**起点权重**：HuggingFace [`JorisCos/ConvTasNet_Libri2Mix_sepclean_16k`](https://huggingface.co/JorisCos/ConvTasNet_Libri2Mix_sepclean_16k)
（作者 **Cosentino Joris**），随包再分发的文件是
`训练记录/代码/pretrained/ConvTasNet_Libri2Mix_sepclean_16k/pytorch_model.bin`
（20 394 640 B / md5 `42e901d57d7c2f79b9d8a74a8077b7b0`），**原样未改动**。

**微调权重**：`臂B_模型.pt`（以及训练检查点 `训练记录/ckpt_B_ep30.pt`，内含微调后的参数）
是上件权重的**改编物（Adapted Material）**，
改动只有两处：

- 在自建仿真远场训练集上微调 **30 轮**；
- 把 `masker.bottleneck` 的 1×1 卷积输入通道由 **512 扩到 518**，接入 6 路 ITD/ILD 空间特征。

其余结构（编码器 / TCN / 解码器 / mask 头）一字未改。

**署名与许可**：

- 署名（Attribution）：**Cosentino Joris**
- 许可（License）：**Creative Commons Attribution-ShareAlike 4.0 International（CC BY-SA 4.0）**
  —— 许可原文 <https://creativecommons.org/licenses/by-sa/4.0/legalcode>
- 因此本仓库的 `臂B_模型.pt` 与 `训练记录/ckpt_B_ep30.pt` **同样以 CC BY-SA 4.0 提供**。
  您再分发或改编它时，请保留上面的署名，并以同样许可发布您的版本。
- 模型卡自述：该权重是 *LibriSpeech ASR corpus*（CC BY 4.0）的衍生作品。
  其许可标注本身前后不一致（元数据写 `cc-by-sa-4.0`，正文写 `Attribution-ShareAlike 3.0 Unported`），
  本仓库按其更严格的口径，统一按 **CC BY-SA 4.0** 处理。

---

## 3. LibriSpeech —— CC BY 4.0

`素材/` 中的 6 条样本 = LibriSpeech 干声（**已做修改**：用自建 RIR 混合成 7 通道远场录音） + 自建 RIR（见第 4 节）。干声部分按 LibriSpeech 的许可使用。

- 署名：**Vassil Panayotov** 等（*LibriSpeech: An ASR corpus based on public domain audio books*, ICASSP 2015）
- 许可：**CC BY 4.0** —— <https://creativecommons.org/licenses/by/4.0/>
- 来源：<https://www.openslr.org/12>

---

## 4. pyroomacoustics —— MIT

训练与素材里的房间冲激响应由 [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)
的镜像源法（image source method）仿真生成。RIR 文件本身是自建产物，生成工具按 MIT 许可使用。

```
MIT License

Copyright (c) 2014-2017 EPFL-LCAV

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 5. 算法出处（论文）

Yi Luo, Nima Mesgarani, *Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking
for Speech Separation*, IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2019.
<https://arxiv.org/abs/1809.02108>
