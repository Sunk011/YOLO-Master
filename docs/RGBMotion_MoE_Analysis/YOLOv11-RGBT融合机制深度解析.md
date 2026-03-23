# YOLOv11-RGBT 红外与可见光融合机制深度解析

## 一、文档概述

本文档详细分析 YOLOv11-RGBT 项目中红外（Infrared）与可见光（Visible）图像的融合机制，包括数据加载层、通道融合、特征融合三个层面。

---

## 二、数据加载层的融合

### 2.1 核心文件

```
ultralytics/data/base.py
├── BaseDataset 类
├── load_and_preprocess_image() 函数
├── _merge_channels() 函数
└── _resize_images() 函数
```

### 2.2 图像加载流程

```python
# 位置: ultralytics/data/base.py 第 177-226 行

def load_and_preprocess_image(self, file_path, use_simotm=None, pairs_rgb=None, pairs_ir=None):
    if use_simotm is None:
        use_simotm = self.use_simotm

    # =========================================================
    # 模式 1: RGBT 模式 - 4通道融合
    # RGB (3通道) + 红外灰度图 (1通道)
    # =========================================================
    elif use_simotm == 'RGBT':
        im_visible = imread(file_path)  # BGR 格式读取
        im_infrared = imread(file_path.replace(pairs_rgb, pairs_ir), cv2.IMREAD_GRAYSCALE)

        if im_visible is None or im_infrared is None:
            raise FileNotFoundError(f"Image Not Found {file_path}")

        im_visible, im_infrared = self._resize_images(im_visible, im_infrared)
        im = self._merge_channels(im_visible, im_infrared)  # 合并为 4 通道

    # =========================================================
    # 模式 2: RGBRGB6C 模式 - 6通道融合
    # RGB (3通道) + 红外RGB (3通道)
    # =========================================================
    elif use_simotm == 'RGBRGB6C':
        im_visible = imread(file_path)  # BGR
        im_infrared = imread(file_path.replace(pairs_rgb, pairs_ir))  # BGR

        if im_visible is None or im_infrared is None:
            raise FileNotFoundError(f"Image Not Found {file_path}")

        im_visible, im_infrared = self._resize_images(im_visible, im_infrared)
        im = self._merge_channels_rgb(im_visible, im_infrared)  # 合并为 6 通道
```

### 2.3 通道合并核心实现

```python
# =========================================================
# 4通道合并: RGB(3ch) + IR(1ch) = 4通道
# 位置: ultralytics/data/base.py 第 248-251 行
# =========================================================
def _merge_channels(self, im_visible, im_infrared):
    """将可见光图像与红外灰度图合并为4通道图像"""
    b, g, r = cv2.split(im_visible)  # 分离 BGR 通道
    im = cv2.merge((b, g, r, im_infrared))  # 合并: B + G + R + IR
    return im

# =========================================================
# 6通道合并: RGB(3ch) + RGB-IR(3ch) = 6通道
# 位置: ultralytics/data/base.py 第 253-257 行
# =========================================================
def _merge_channels_rgb(self, im_visible, im_infrared):
    """将可见光图像与红外RGB图像合并为6通道图像"""
    b, g, r = cv2.split(im_visible)      # 分离可见光 BGR
    b2, g2, r2 = cv2.split(im_infrared)  # 分离红外 BGR
    im = cv2.merge((b, g, r, b2, g2, r2))  # 合并: B + G + R + IR_B + IR_G + IR_R
    return im
```

### 2.4 图像尺寸对齐

```python
# 位置: ultralytics/data/base.py 第 228-246 行

def _resize_images(self, im_visible, im_infrared):
    """确保可见光和红外图像尺寸一致"""
    h_vis, w_vis = im_visible.shape[:2]
    h_inf, w_inf = im_infrared.shape[:2]

    if h_vis != h_inf or w_vis != w_inf:
        r_vis = self.imgsz / max(h_vis, w_vis)
        r_inf = self.imgsz / max(h_inf, w_inf)

        if r_vis != 1:
            interp = cv2.INTER_LINEAR if (self.augment or r_vis > 1) else cv2.INTER_AREA
            im_visible = cv2.resize(im_visible, (
                min(math.ceil(w_vis * r_vis), self.imgsz),
                min(math.ceil(h_vis * r_vis), self.imgsz)),
                interpolation=interp)
        if r_inf != 1:
            interp = cv2.INTER_LINEAR if (self.augment or r_inf > 1) else cv2.INTER_AREA
            im_infrared = cv2.resize(im_infrared, (
                min(math.ceil(w_inf * r_inf), self.imgsz),
                min(math.ceil(h_inf * r_inf), self.imgsz)),
                interpolation=interp)
    return im_visible, im_infrared
```

### 2.5 数据集配置参数

```python
# 位置: ultralytics/data/base.py 第 65-96 行

class BaseDataset(Dataset):
    def __init__(
        self,
        ...
        use_simotm="RGB",           # 多模态融合模式
        pairs_rgb_ir=['visible', 'infrared']  # 目录名映射
    ):
        self.use_simotm = use_simotm
        self.pairs_rgb_ir = pairs_rgb_ir  # 路径替换: 'visible' -> 'infrared'
```

### 2.6 路径自动替换机制

```python
# 核心原理: 将可见光路径中的 'visible' 替换为 'infrared' 以获取红外图像路径

# 示例:
# 可见光路径: dataset/visible/train/image001.jpg
# 红外路径:   dataset/infrared/train/image001.jpg

pairs_rgb_ir = ['visible', 'infrared']
pairs_rgb, pairs_ir = pairs_rgb_ir

im_infrared = imread(file_path.replace(pairs_rgb, pairs_ir), ...)
# file_path = "dataset/visible/train/image001.jpg"
# 替换后:   "dataset/infrared/train/image001.jpg"
```

---

## 三、支持的融合模式

### 3.1 模式总览

| 模式 | 通道数 | 描述 | 输入格式 |
|------|--------|------|----------|
| `Gray` | 1 | 单通道灰度图 | uint8 灰度 |
| `Gray16bit` | 1 | 16位单通道灰度图 | uint16 灰度 |
| `SimOTM` | 3 | 灰度转3通道伪彩色 | 单通道 + 模糊 + 边缘 |
| `SimOTMBBS` | 3 | 灰度转3通道 (Blur×2) | 单通道 + 模糊×2 |
| `BGR` | 3 | 标准RGB彩色图 | BGR |
| `RGBT` | **4** | RGB + 红外灰度 | BGR + Gray |
| `RGBRGB6C` | **6** | RGB + 红外RGB | BGR + BGR |
| `Multispectral` | N | 任意通道多光谱 | N通道 |
| `Multispectral_16bit` | N | 16位多光谱 | N通道 uint16 |

### 3.2 融合模式详解

```
┌─────────────────────────────────────────────────────────────────┐
│                     RGBT 模式 (4通道)                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │    B    │ │    G    │ │    R    │ │   IR    │            │
│  │ (可见光) │ │ (可见光) │ │ (可见光) │ │ (红外)  │            │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘            │
│       通道0        通道1        通道2        通道3               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   RGBRGB6C 模式 (6通道)                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│  │  B_vis  │ │  G_vis  │ │  R_vis  │ │  B_ir   │ │  G_ir   │ │  R_ir   │
│  │ (可见光) │ │ (可见光) │ │ (可见光) │ │ (红外)  │ │ (红外)  │ │ (红外)  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
│       通道0        通道1        通道2        通道3        通道4        通道5
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、特征层融合模块

### 4.1 融合模块总览

YOLOv11-RGBT 提供了多种特征层融合模块：

| 模块名称 | 类型 | 文件位置 | 描述 |
|----------|------|----------|------|
| `CrossC2f` | CSP跨分支 | block.py | 双分支交叉连接 |
| `CrossC3k2` | CSP跨分支 | block.py | CrossC2f 的快速版本 |
| `NiNfusion` | 1×1卷积融合 | conv.py | 简单通道拼接+卷积 |
| `TransformerFusionBlock` | Transformer | conv.py | 交叉注意力融合 |
| `MultiHeadCrossAttention` | 交叉注意力 | block.py | 多头交叉注意力 |
| `CrossTransformerFusion` | Transformer | transformer.py | Transformer编码器融合 |

### 4.2 CrossC2f 交叉连接模块

**位置**: `ultralytics/nn/modules/block.py` 第 759-802 行

```python
class CrossC2f(nn.Module):
    """Cross-Connected CSP Bottleneck - 双分支交叉连接"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, ratio=0.15):
        super(CrossC2f, self).__init__()
        self.c = int(c2 * e)  # hidden channels
        self.ratio = ratio     # 残差连接系数

        # 输入通道翻倍 (2个分支的输入)
        self.cv1 = Conv(c1 * 2, 2 * self.c, 1, 1)  # 融合两个分支的特征
        self.cv2 = Conv(c1, c2, 1, 1)
        self.cv3 = Conv(self.c * (n + 1), c1, 1, 1)

        # 两组 Bottleneck 分支
        self.m1 = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))
        self.m2 = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        """前向传播: 接收两个分支的特征 [x1, x2]"""
        x1, x2 = x  # 解包两个分支

        # 1. 拼接两个分支的特征
        x_concat = torch.cat([x1, x2], dim=1)

        # 2. 1×1 卷积融合后分割
        y = self.cv1(x_concat).split(self.c, dim=1)  # [part_a, part_b]

        # 3. 交叉连接 - 关键创新点！
        out_1 = [y[1]]  # 分支1从 part_b 开始
        out_2 = [y[0]]  # 分支2从 part_a 开始

        # 4. 各自通过 Bottleneck
        for m1, m2 in zip(self.m1, self.m2):
            out_1.append(m1(out_1[-1]))  # 分支1继续处理
            out_2.append(m2(out_2[-1]))  # 分支2继续处理

        # 5. 拼接中间结果
        out_1 = torch.cat(out_1, dim=1)
        out_2 = torch.cat(out_2, dim=1)

        # 6. 共享卷积 + 残差连接
        out_1 = x1 * self.ratio + self.cv3(out_1)
        out_2 = x2 * self.ratio + self.cv3(out_2)

        # 7. 输出融合 (加法而非拼接)
        out = out_1 + out_2

        return [out_1, out_2, self.cv2(out)]  # 返回3个输出供后续使用
```

**架构图解**:
```
输入: [x1, x2] 两个分支
           │
           ▼
    ┌──────────────────┐
    │ Concat [x1, x2]  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  Conv 1×1 融合   │
    └────────┬─────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
   [part_a]     [part_b]
      │             │
      ▼             ▼
   out_2[0]     out_1[0]
      │             │
      ▼             ▼
   ┌─────────┐ ┌─────────┐
   │Bottleneck│ │Bottleneck│
   │   (m2)   │ │   (m1)   │
   └────┬─────┘ └────┬─────┘
        │             │
        ▼             ▼
   ┌─────────┐ ┌─────────┐
   │Bottleneck│ │Bottleneck│
   │   (m2)   │ │   (m1)   │
   └────┬─────┘ └────┬─────┘
        │             │
        ▼             ▼
   out_2 concat   out_1 concat
        │             │
        ▼             ▼
   ┌─────────┐ ┌─────────┐
   │Ratio*x2  │ │Ratio*x1  │
   │  + Conv  │ │  + Conv  │
   └────┬─────┘ └────┬─────┘
        │             │
        ▼             ▼
       out_2    +    out_1
                   │
                   ▼
               [out_1, out_2, cv2(out)]
```

### 4.3 NiNfusion 模块

**位置**: `ultralytics/nn/modules/conv.py` 第 470-482 行

```python
class NiNfusion(nn.Module):
    """
    Network in Network Fusion
    简单高效的特征融合模块
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()
        self.concat = Concat(dimension=1)  # 通道维度拼接
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        x: 包含多个特征图的列表 [fea1, fea2, ...]
        """
        y = self.concat(x)    # 拼接所有特征
        y = self.act(self.conv(y))  # 1×1 卷积融合 + 激活
        return y
```

### 4.4 TransformerFusionBlock 模块

**位置**: `ultralytics/nn/modules/conv.py` 第 687-793 行

```python
class TransformerFusionBlock(nn.Module):
    """
    基于 Transformer 的跨模态特征融合模块
    使用交叉注意力机制融合可见光和红外特征
    """

    def __init__(self, d_model, vert_anchors=16, horz_anchors=16,
                 h=8, block_exp=4, n_layer=1, ...):
        super(TransformerFusionBlock, self).__init__()

        # 位置编码 (可学习)
        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, d_model))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, d_model))

        # 自适应池化
        self.avgpool = AdaptivePool2d(vert_anchors, horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(vert_anchors, horz_anchors, 'max')

        # 可学习权重融合
        self.vis_coefficient = LearnableWeights()  # 可见光融合权重
        self.ir_coefficient = LearnableWeights()   # 红外融合权重

        # 交叉 Transformer
        self.crosstransformer = nn.Sequential(
            *[CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, ...)
              for layer in range(n_layer)]
        )

        # 输出融合
        self.concat = Concat(dimension=1)
        self.conv1x1_out = Conv(d_model * 2, d_model, k=1, s=1)

    def forward(self, x):
        """
        x: [rgb_fea, ir_fea] 两个模态的特征
        """
        rgb_fea = x[0]
        ir_fea = x[1]

        # 1. 双分支池化融合 (Avg + Max)
        new_rgb_fea = self.vis_coefficient(self.avgpool(rgb_fea), self.maxpool(rgb_fea))
        new_ir_fea = self.ir_coefficient(self.avgpool(ir_fea), self.maxpool(ir_fea))

        # 2. Flatten + 位置编码
        bs, c, h, w = rgb_fea.shape
        rgb_fea_flat = new_rgb_fea.view(bs, c, -1).permute(0, 2, 1) + self.pos_emb_vis
        ir_fea_flat = new_ir_fea.view(bs, c, -1).permute(0, 2, 1) + self.pos_emb_ir

        # 3. 交叉 Transformer 处理
        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])

        # 4. 恢复空间维度 + 残差连接
        rgb_fea_CFE = rgb_fea_flat.contiguous().view(bs, h, w, c).permute(0, 3, 1, 2)
        new_rgb_fea = rgb_fea_CFE + rgb_fea  # 残差连接

        ir_fea_CFE = ir_fea_flat.contiguous().view(bs, h, w, c).permute(0, 3, 1, 2)
        new_ir_fea = ir_fea_CFE + ir_fea  # 残差连接

        # 5. 通道拼接 + 1×1 卷积输出
        new_fea = self.concat([new_rgb_fea, new_ir_fea])
        new_fea = self.conv1x1_out(new_fea)

        return new_fea
```

### 4.5 MultiHeadCrossAttention 模块

**位置**: `ultralytics/nn/modules/block.py` 第 2826-2919 行

```python
class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力机制
    用于可见光和红外特征的深度交互
    """

    def __init__(self, model_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # 可见光 QKV
        self.query_vis = nn.Linear(model_dim, model_dim)
        self.key_vis = nn.Linear(model_dim, model_dim)
        self.value_vis = nn.Linear(model_dim, model_dim)

        # 红外 QKV
        self.query_inf = nn.Linear(model_dim, model_dim)
        self.key_inf = nn.Linear(model_dim, model_dim)
        self.value_inf = nn.Linear(model_dim, model_dim)

        # 输出投影
        self.fc_out_vis = nn.Linear(model_dim, model_dim)
        self.fc_out_inf = nn.Linear(model_dim, model_dim)

    def forward(self, vis, inf):
        """
        vis: 可见光特征 (batch, seq_len, dim)
        inf: 红外特征 (batch, seq_len, dim)
        """
        # 生成 QKV
        Q_vis = self.query_vis(vis)
        K_vis = self.key_vis(vis)
        V_vis = self.value_vis(vis)

        Q_inf = self.query_inf(inf)
        K_inf = self.key_inf(inf)
        V_inf = self.value_inf(inf)

        # Reshape for multi-head attention
        Q_vis = Q_vis.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K_vis = K_vis.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V_vis = V_vis.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

        # 交叉注意力计算
        # 可见光查询关注红外键
        scores_vis_inf = torch.matmul(Q_vis, K_inf.transpose(-1, -2)) / sqrt(head_dim)
        attention_inf = torch.softmax(scores_vis_inf, dim=-1)
        out_inf = torch.matmul(attention_inf, V_inf)

        # 红外查询关注可见光键
        scores_inf_vis = torch.matmul(Q_inf, K_vis.transpose(-1, -2)) / sqrt(head_dim)
        attention_vis = torch.softmax(scores_inf_vis, dim=-1)
        out_vis = torch.matmul(attention_vis, V_vis)

        # 输出投影
        out_vis = self.fc_out_vis(out_vis)
        out_inf = self.fc_out_inf(out_inf)

        return out_vis, out_inf
```

---

## 五、融合策略分类

### 5.1 融合层级分类

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           融合策略总览                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
│  │  数据层融合  │    │  特征层融合  │    │  决策层融合  │                │
│  │ (Early Fuse)│    │(Middle Fuse)│    │(Late Fusion)│                │
│  └─────────────┘    └─────────────┘    └─────────────┘                │
│         │                │                  │                         │
│         ▼                ▼                  ▼                         │
│  ┌─────────────────────────────────────────────────────────┐          │
│  │                                                        │          │
│  │   原始图像 → 通道拼接 → 统一特征提取                    │          │
│  │                                                        │          │
│  │   可见光 ──┐                                        │          │
│  │            │ → Concat → Conv → 共享Backbone          │          │
│  │   红外 ────┘                                        │          │
│  │                                                        │          │
│  └─────────────────────────────────────────────────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 YOLOv11-RGBT 融合策略实现

| 策略 | 实现方式 | 特点 |
|------|----------|------|
| **早期融合 (Early Fusion)** | 数据加载层 `_merge_channels()` | 通道拼接，统一特征提取 |
| **中期融合 (Middle Fusion)** | `CrossC2f`, `NiNfusion`, `TransformerFusionBlock` | 特征级交互 |
| **晚期融合 (Late Fusion)** | 需单独实现 | 独立分支预测后加权 |
| **权重共享融合** | 标准 YOLO 架构 | 共享 Backbone |

---

## 六、代码调用链路

### 6.1 完整调用流程

```
训练脚本 (train.py / train_RGBT.py)
    │
    ▼
YOLO(...).train(data='xxx.yaml', use_simotm='RGBT', channels=4)
    │
    ▼
BaseDataset.__init__()
    ├── self.use_simotm = 'RGBT'
    ├── self.pairs_rgb_ir = ['visible', 'infrared']
    │
    ▼
BaseDataset.load_image()
    │
    ▼
load_and_preprocess_image(file_path, use_simotm='RGBT')
    │
    ├── im_visible = imread(file_path)           # 读取可见光
    ├── im_infrared = imread(file_path.replace(
    │                    'visible', 'infrared')) # 读取红外
    │
    ▼
_resize_images(im_visible, im_infrared)         # 尺寸对齐
    │
    ▼
_merge_channels(im_visible, im_infrared)        # 通道合并
    │  └── cv2.merge((B, G, R, IR))              # 4通道图像
    │
    ▼
返回 4通道图像 → DataLoader → Model
```

### 6.2 模型配置示例

```yaml
# FLIR_aligned-rgbt.yaml

# 数据集路径
path: /path/to/dataset
train: visible/train
val: visible/test

# 类别配置
nc: 3
names: ["person", "car", "bicycle"]

# YOLOv11-RGBT 特有参数 (在训练时指定)
# use_simotm: RGBT    # 融合模式
# channels: 4         # 输入通道数
# pairs_rgb_ir: ['visible', 'infrared']
```

### 6.3 训练脚本使用示例

```python
# train_RGBT.py

from ultralytics import YOLO

model = YOLO('yolo11n.yaml')

results = model.train(
    data='FLIR_aligned-rgbt.yaml',
    epochs=300,
    batch=16,
    imgsz=640,

    # YOLOv11-RGBT 特有参数
    use_simotm='RGBT',      # 融合模式
    channels=4,              # 输入通道数
    pairs_rgb_ir=['visible', 'infrared'],  # 目录映射
)
```

---

## 七、关键代码位置索引

| 功能 | 文件 | 行号 | 函数/类 |
|------|------|------|---------|
| 模式选择 | base.py | 65-66 | BaseDataset.__init__ |
| 图像加载 | base.py | 177-226 | load_and_preprocess_image |
| RGBT合并 | base.py | 248-251 | _merge_channels |
| RGBRGB6C合并 | base.py | 253-257 | _merge_channels_rgb |
| 尺寸对齐 | base.py | 228-246 | _resize_images |
| CrossC2f | block.py | 759-802 | CrossC2f.forward |
| NiNfusion | conv.py | 470-482 | NiNfusion.forward |
| TransformerFusion | conv.py | 687-793 | TransformerFusionBlock.forward |
| MultiHeadCrossAttn | block.py | 2826-2919 | MultiHeadCrossAttention.forward |
| CrossTransformerBlock | conv.py | 615-683 | CrossTransformerBlock.forward |

---

## 八、与 YOLO-Master 的对比

### 8.1 框架差异

| 特性 | YOLOv11-RGBT | YOLO-Master |
|------|-------------|-------------|
| **多模态支持** | ✅ 原生支持 | ❌ 需适配 |
| **融合模式** | RGBT, RGBRGB6C 等 | 无内置 |
| **通道配置** | channels 参数 | 无专用参数 |
| **MoE 支持** | ❌ 无 | ✅ 完整 MoE |
| **动态路由** | ❌ 无 | ✅ ES-MoE |

### 8.2 迁移要点

将 YOLOv11-RGBT 的融合机制迁移到 YOLO-Master 需要:

1. **数据层适配**
   - 复制 `BaseDataset.load_and_preprocess_image()`
   - 复制 `_merge_channels()` 和 `_merge_channels_rgb()`
   - 添加 `use_simotm` 和 `pairs_rgb_ir` 参数

2. **特征融合模块**
   - 复用 `CrossC2f`, `NiNfusion`, `TransformerFusionBlock`
   - 或使用 YOLO-Master 的 MoE 模块进行特征选择

3. **配置兼容**
   - 修改 YAML 配置支持 `ch: 4` 或 `ch: 6`
   - 确保模型解析正确处理多通道输入

---

*文档版本: v1.0*
*最后更新: 2026-03-22*
