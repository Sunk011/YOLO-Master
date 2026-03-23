# HomoMotion 快速使用指南

## 一、基本使用示例

### 1.1 基础运动检测

```python
from homomotion import MotionDetector
import cv2

# 初始化检测器
detector = MotionDetector(
    feature_type='ORB',      # 或 'SIFT', 'CUDA_SIFT'
    num_frames=3,             # 使用3帧
    frame_interval=1,        # 连续帧
    diff_mode='mean',        # 'mean' 或 'subtract'
    nfeatures=2000,
    debug=False
)

# 加载图像序列
images = [
    cv2.imread('frame_t-2.jpg'),
    cv2.imread('frame_t-1.jpg'),
    cv2.imread('frame_t.jpg')
]

# 处理
result = detector.process(images)

# 访问结果
print(f"运动图形状: {result.motion_map.shape}")
print(f"有效区域比例: {result.valid_mask.sum() / result.valid_mask.size:.2%}")
print(f"单应性矩阵数量: {len(result.homography_matrices)}")

# 可视化
cv2.imshow('Motion Map', result.motion_map)
cv2.imshow('Valid Mask', result.valid_mask)
cv2.waitKey(0)
```

### 1.2 GPU 加速版本

```python
# 使用 CUDA-SIFT (需要编译 E-Sift)
detector = MotionDetector(
    feature_type='CUDA_SIFT',  # GPU 加速
    nfeatures=2000,
    num_frames=3
)

# 性能对比 (1920x1080, 3帧):
# SIFT (CPU): ~1500ms → 0.6 FPS
# CUDA_SIFT (GPU): ~75-100ms → 10-14 FPS
```

---

## 二、配置预设

```python
from homomotion.config import DefaultConfig

# 高质量配置 (较慢)
config_hq = DefaultConfig.for_high_quality()
# feature_type='SIFT', nfeatures=3000, ratio_threshold=0.75, min_matches=20

# 快速配置 (较快)
config_fast = DefaultConfig.for_fast_processing()
# feature_type='ORB', nfeatures=1000, min_matches=8
```

---

## 三、进阶使用

### 3.1 忽略水印区域

```python
# 定义忽略区域 (如图像水印)
ignore_regions = [
    {"x1": 10, "y1": 10, "x2": 110, "y2": 60, "name": "watermark"},
    {"x1": 1700, "y1": 950, "x2": 1910, "y2": 1070, "name": "timestamp"}
]

detector = MotionDetector(
    ignore_regions=ignore_regions,
    ...
)
```

### 3.2 方向性运动检测

```python
# 输出3通道方向性运动图
detector = MotionDetector(
    directional_motion=True,
    diff_mode='mean'
)

result = detector.process(images)

# 通道说明:
# result.motion_map[:,:,0] = 运动强度 (绝对值)
# result.motion_map[:,:,1] = 正向运动 (物体进入)
# result.motion_map[:,:,2] = 负向运动 (物体离开)
```

### 3.3 调试模式

```python
# 启用调试获取中间结果
detector = MotionDetector(debug=True)

result = detector.process(images)

# 访问调试信息
if result.warped_frames:
    for i, warped in enumerate(result.warped_frames):
        cv2.imwrite(f'warped_frame_{i}.jpg', warped)

if result.individual_masks:
    for i, mask in enumerate(result.individual_masks):
        cv2.imwrite(f'mask_{i}.jpg', mask)

if result.match_info:
    print(result.match_info)
```

---

## 四、帧索引计算

```python
detector = MotionDetector(num_frames=3, frame_interval=2)

# 获取需要的帧索引
indices = detector.get_frame_indices()
print(indices)  # [-4, -2, 0]

# 意味着需要提供: Frame(t-4), Frame(t-2), Frame(t)
```

---

## 五、批量处理工具

### 5.1 命令行使用

```bash
python tools/batch_process_directional_motion.py \
    --input_dir "path/to/sequence" \
    --output_root "output/dir" \
    --num_frames 3 \
    --feature_type ORB
```

### 5.2 批量处理代码

```python
from pathlib import Path
from homomotion.core.motion_detector import MotionDetector

def process_video_frames(video_dir: Path, output_dir: Path):
    detector = MotionDetector(num_frames=3)
    frames = []

    for img_path in sorted(video_dir.glob('*.jpg')):
        frame = cv2.imread(str(img_path))
        frames.append(frame)

        if len(frames) == 3:
            result = detector.process(frames)

            # 保存运动图
            output_path = output_dir / f"{img_path.stem}_motion.png"
            cv2.imwrite(str(output_path), result.motion_map)

            frames.pop(0)  # 滑动窗口

# 使用
process_video_frames(
    Path('dataset/frames'),
    Path('dataset/motion_maps')
)
```

---

## 六、性能基准测试

```python
# tools/benchmark/profile_performance.py

import time
import cv2
from homomotion import MotionDetector

detector = MotionDetector(feature_type='ORB', num_frames=3)

# 加载测试图像
images = [cv2.imread(f'frame_{i}.jpg') for i in range(3)]

# 基准测试
iterations = 100
start = time.time()

for _ in range(iterations):
    result = detector.process(images)

elapsed = time.time() - start
print(f"平均耗时: {elapsed/iterations*1000:.1f}ms")
print(f"估计 FPS: {iterations/elapsed:.1f}")
```
