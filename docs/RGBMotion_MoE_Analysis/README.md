# RGB + Motion MoE 方案总结

## 核心问题回答

### Q1: 这个方案可行吗？

**答案: 是的，方案可行。**

主要理由:
1. **框架同源**: YOLO-Master 和 YOLOv11-RGBT 都基于 Ultralytics 框架，代码兼容性高
2. **MoE 成熟**: YOLO-Master 已实现完整的 MoE 模块体系，可直接复用
3. **多模态验证**: YOLOv11-RGBT 已验证双模态输入管道，无需从零开发
4. **运动信息优势**: 运动信息不受光照影响，对运动目标检测有天然优势

### Q2: 实现难度如何？

| 难度等级 | 说明 |
|---------|------|
| 🟢 低 | 数据集管道适配（复用 BaseDataset） |
| 🟡 中 | 运动图质量优化（需实验调参） |
| 🟡 中 | MoE 专家通道数适配（少量代码修改） |

**总体难度: 中等偏下** - 主要是集成工作，而非从零开发

### Q3: 预期性能如何？

| 指标 | 预期 |
|------|------|
| mAP 提升 | +3-5% (运动目标、低照度场景更明显) |
| 推理延迟 | 增加 30-50%（主要是运动提取） |
| 参数量 | 增加 ~50%（通道数翻倍） |

### Q4: 主要风险点？

1. **运动图质量**: 单应性估计失败时运动图会有噪声 → 解决方案：添加质量检测与回退机制
2. **实时性**: SIFT/ORB 计算量大 → 解决方案：使用 CUDA-SIFT 或轻量替代方案
3. **训练稳定性**: MoE + 多模态可能收敛慢 → 解决方案：适当增大学习率预热

---

## 快速开始指南

### 第一步: 环境准备

```bash
# 1. 克隆 HomoMotion
cd 3rdparty
git clone -b rtx4090 https://github.com/Polaris-F/E-Sift.git esift  # 可选 GPU 加速

# 2. 安装依赖
pip install -r requirements.txt
pip install -e .
```

### 第二步: 准备数据集

```
dataset/
├── visible/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 第三步: 配置模型

复制 `yolo-master-n.yaml` → `yolo-master-rgbmotion-n.yaml`
- 修改 `ch: 6` (RGB 3ch + Motion 3ch)

### 第四步: 开始训练

```python
from ultralytics import YOLO

model = YOLO('yolo-master-rgbmotion-n.yaml')
model.train(
    data='rgbmotion.yaml',
    channels=6,
    motion_mode='extract',
    epochs=300,
)
```

---

## 关键技术点

### 1. 运动图生成

```python
# 简化版运动提取
def extract_motion(frames):
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diff = cv2.absdiff(gray[0], gray[1])
    return diff  # 单通道运动图
```

### 2. 通道融合

```python
# RGB + Motion → 6通道
merged = np.concatenate([rgb_image, motion_3ch], axis=2)
```

### 3. MoE 配置

```yaml
# YAML 中的 MoE 模块配置
- [-1, 1, ModularRouterExpertMoE, [512, 4, 2]]
#                                              ↑  ↑  ↑
#                                        专家数  top_k
```

---

## 推荐实现顺序

```
1. 先用简化运动提取（帧差分）验证流程
2. 集成 HomoMotion 提升运动图质量
3. 适配 MoE 模块
4. 性能优化
```

---

## 参考资源

| 资源 | 位置 |
|------|------|
| 详细可行性分析 | `docs/RGBMotion_MoE_Analysis/可行性分析报告.md` |
| 技术实现方案 | `docs/RGBMotion_MoE_Analysis/详细实现方案.md` |
| YOLO-Master MoE | `ultralytics/nn/modules/moe/` |
| HomoMotion | `3rdparty/HomoMotion/` |
| YOLOv11-RGBT 参考 | `3rdparty/YOLOv11-RGBT/` |
