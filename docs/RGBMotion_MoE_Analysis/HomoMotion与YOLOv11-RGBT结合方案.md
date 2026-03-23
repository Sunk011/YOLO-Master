# HomoMotion 运动信息提取机制深度解析与 YOLOv11-RGBT 结合方案

## 一、HomoMotion 核心算法流程

### 1.1 整体处理流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HomoMotion 处理流程                                    │
└─────────────────────────────────────────────────────────────────────────────┘

输入: 连续帧序列 [Frame(t-n), ..., Frame(t-1), Frame(t)]
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: 特征提取 (Feature Extraction)                                      │
│  ┌─────────────────┐    ┌─────────────────┐                                  │
│  │  ORB/SIFT/CUDA │ → │  关键点 + 描述子  │                                  │
│  └─────────────────┘    └─────────────────┘                                  │
│  对所有帧提取特征点                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: 特征匹配 (Feature Matching)                                         │
│  ┌─────────────────┐    ┌─────────────────┐                                  │
│  │ BFMatcher/FLANN │ → │  匹配点对        │                                  │
│  │  + LoweRatio     │    │  src_pts, dst_pts │                                  │
│  └─────────────────┘    └─────────────────┘                                  │
│  匹配相邻帧间的特征点                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 单应性矩阵估计 (Homography Estimation)                              │
│  ┌─────────────────┐    ┌─────────────────┐                                  │
│  │ cv2.findHomography│ → │  H 矩阵 (3×3)    │                                  │
│  │  + RANSAC       │    │  + 内点Mask      │                                  │
│  └─────────────────┘    └─────────────────┘                                  │
│  计算前一帧到当前帧的透视变换                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: 图像变换 (Image Warping)                                           │
│  ┌─────────────────┐    ┌─────────────────┐                                  │
│  │cv2.warpPerspect│ → │  对齐帧          │                                  │
│  │   ive(H)       │    │  warped_frame   │                                  │
│  └─────────────────┘    └─────────────────┘                                  │
│  将前一帧变换到当前帧的坐标系                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5: 有效区域计算 (Valid Region)                                        │
│  ┌─────────────────┐    ┌─────────────────┐                                  │
│  │ 角点变换 + Mask │ → │  valid_mask     │                                  │
│  │  + 形态学操作   │    │  (只保留重叠区) │                                  │
│  └─────────────────┘    └─────────────────┘                                  │
│  计算两帧的像素级重叠区域                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 6: 帧差计算 (Motion Computation)                                     │
│  ┌─────────────────┐    ┌─────────────────┐                                  │
│  │ |warped - ref| │ → │  motion_map     │                                  │
│  │  + 增强处理    │    │  (单通道/3通道)  │                                  │
│  └─────────────────┘    └─────────────────┘                                  │
│  计算运动区域并增强对比度                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
输出: MotionResult(homography_matrices, motion_map, valid_mask)
```

---

## 二、核心模块详解

### 2.1 特征提取器 (FeatureExtractor)

**文件**: `homomotion/core/feature_extractor.py`

```python
# 支持的特征提取算法
class FeatureExtractor(ABC):
    """抽象基类"""

class ORBExtractor(FeatureExtractor):
    """ORB - 快速，二值描述子，汉明距离匹配"""
    def __init__(self, nfeatures: int = 2000):
        self.detector = cv2.ORB_create(nfeatures=nfeatures)

class SIFTExtractor(FeatureExtractor):
    """SIFT - 鲁棒，浮点描述子，L2距离匹配"""
    def __init__(self, nfeatures: int = 2000):
        self.detector = cv2.SIFT_create(nfeatures=nfeatures)

class CudaSIFTExtractor(FeatureExtractor):
    """CUDA-SIFT - GPU加速，60倍加速"""
    # 需要编译 E-Sift 库
```

**性能对比**:

| 算法 | 速度 | 精度 | 适用场景 |
|------|------|------|---------|
| ORB | 最快 (~50fps) | 中等 | 实时应用 |
| SIFT | 中等 (~5fps) | 高 | 精度优先 |
| CUDA-SIFT | 快 (~30fps) | 高 | 有GPU的实时应用 |

### 2.2 特征匹配器 (FeatureMatcher)

**文件**: `homomotion/core/feature_matcher.py`

```python
class FeatureMatcher:
    def __init__(self,
                 matcher_type='auto',  # 'auto', 'bf', 'flann'
                 ratio_threshold=0.7,   # Lowe比率阈值
                 min_matches=10):
        self.ratio_threshold = ratio_threshold
        self.min_matches = min_matches

    def match(self, desc1, desc2, kp1, kp2):
        """
        匹配流程:
        1. 根据描述子类型选择匹配器
           - 二值描述子(ORB) → BFMatcher (汉明距离)
           - 浮点描述子(SIFT) → FLANN (L2距离)

        2. Lowe's Ratio Test 过滤
           - 对于每个匹配，找到最近和次近邻
           - 如果 最近/次近 < ratio_threshold，保留

        3. 返回优质匹配点对
        """
```

### 2.3 单应性矩阵估计 (HomographyEstimator)

**文件**: `homomotion/core/homography.py`

```python
class HomographyEstimator:
    def __init__(self,
                 method=cv2.RANSAC,
                 ransac_reproj_threshold=5.0,  # 像素
                 max_iters=2000,
                 confidence=0.995):
        self.ransac_threshold = ransac_reproj_threshold

    def estimate(self, src_pts, dst_pts):
        """
        计算单应性矩阵 H 使得: dst_pts ≈ H @ src_pts

        输入:
        - src_pts: 前一帧的匹配点 (N, 1, 2)
        - dst_pts: 当前帧的匹配点 (N, 1, 2)

        输出:
        - H: 3×3 单应性矩阵
        - mask: 内点掩码

        验证条件:
        - 条件数(condition number) < 1e6
        - 行列式(determinant) ≠ 0
        - 所有元素有限
        """

    def warp_image(self, image, H, output_shape):
        """
        使用单应性矩阵变换图像

        cv2.warpPerspective(src, H, dsize)
        """
```

### 2.4 图像处理工具 (ImageUtils)

**文件**: `homomotion/utils/image_utils.py`

#### 2.4.1 有效区域计算

```python
def compute_valid_region_mask(image, warped_image, H, source_shape):
    """
    计算变换后的有效重叠区域

    两种计算方式:
    1. 内容掩码: warped_image 非零像素区域
    2. 几何掩码: 源图四角经H变换后的多边形区域

    最终 = 内容掩码 & 几何掩码
    """
```

#### 2.4.2 运动图计算

```python
def compute_motion_map(warped_frames, reference_frame, diff_mode='mean'):
    """
    计算运动图

    diff_mode='mean':
        motion = mean(|warped_i - ref|)

    diff_mode='subtract':
        motion = mean(|warped_i - warped_{i+1}|)
        # 减少远距离物体的重影

    输出:
        - 单通道: 运动强度
        - directional=True: 3通道 [强度, 正向运动, 负向运动]
    """
```

#### 2.4.3 运动增强

```python
def motion_enhance_v3(diff_img, noise_threshold=10.0,
                     saturation_val=60.0, enhancement_power=2.0):
    """
    运动图增强算法

    公式: motion = ((diff - noise_threshold) / range) ^ power * 255

    效果:
    - diff=10 → 0 (过滤背景噪声)
    - diff=35 → ~64 (中等增强)
    - diff=60+ → 255 (饱和为高亮)
    """
```

---

## 三、运动检测核心算法

### 3.1 MotionDetector 主流程

```python
class MotionDetector:
    def process(self, images: List[np.ndarray]) -> MotionResult:
        """
        images: [Frame(t-n), ..., Frame(t-1), Frame(t)]
                 最后一帧作为参考帧

        返回:
        MotionResult:
        - homography_matrices: 单应性矩阵列表
        - motion_map: 运动图
        - valid_mask: 有效区域掩码
        """
        reference_frame = images[-1]  # 最后一帧作为参考
        prev_frames = images[:-1]

        for prev_frame in prev_frames:
            # Step 1: 特征提取
            kp_prev, desc_prev = self.extractor.extract(prev_frame)
            kp_ref, desc_ref = self.extractor.extract(reference_frame)

            # Step 2: 特征匹配
            good_matches = self.matcher.match(desc_prev, desc_ref, kp_prev, kp_ref)

            # Step 3: 单应性估计
            H = self.homography_estimator.estimate(src_pts, dst_pts)

            # Step 4: 图像变换
            warped = cv2.warpPerspective(prev_frame, H, (w, h))

            # Step 5: 有效区域
            valid_mask = compute_valid_region_mask(reference_frame, warped, H)

            # Step 6: 运动图
            motion = |warped - reference| * valid_mask
```

### 3.2 坐标系变换详解

```
原始帧坐标系                    参考帧(当前帧)坐标系
┌─────────────────┐           ┌─────────────────┐
│                 │           │                 │
│   Frame(t-1)    │  ──H──→   │   Frame(t)       │
│   相机固定       │  单应性    │   参考帧         │
│   背景移动       │  变换     │                 │
└─────────────────┘           └─────────────────┘

通过 H 将 Frame(t-1) 变换到 Frame(t) 的坐标系后，
像素级的差异就只来自于运动物体
```

---

## 四、与 YOLOv11-RGBT 的结合方案

### 4.1 数据流整合架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    整合后的 RGB + Motion 数据流                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                         原始视频序列
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌─────────────────┐   ┌─────────────────┐
          │  RGB 可见光帧    │   │  连续帧序列      │
          │  (用于检测)     │   │  (用于运动提取)   │
          └────────┬────────┘   └────────┬────────┘
                   │                   │
                   ▼                   ▼
          ┌─────────────────┐   ┌─────────────────┐
          │  YOLOv11-RGBT   │   │  HomoMotion     │
          │  数据加载管道    │   │  运动检测       │
          │  _merge_channels│   │                 │
          └────────┬────────┘   └────────┬────────┘
                   │                   │
                   │                   ▼
                   │         ┌─────────────────┐
                   │         │  motion_map     │
                   │         │  (已对齐的运动图) │
                   │         └────────┬────────┘
                   │                   │
                   └─────────┬─────────┘
                             ▼
                   ┌─────────────────┐
                   │  融合特征图      │
                   │  RGB(3ch) +     │
                   │  Motion(3ch)    │
                   │  = 6通道输入    │
                   └────────┬────────┘
                            ▼
                   ┌─────────────────┐
                   │  YOLO-Master    │
                   │  Backbone       │
                   └─────────────────┘
```

### 4.2 像素级对齐的重要性

**HomoMotion 的关键价值**: 实现像素级精确对齐

```
未对齐的问题:
┌────────────────────────────────────────┐
│                                        │
│   Frame(t-1)    │    Frame(t)           │
│                 │                      │
│     物体 ────────┼────── 物体            │
│    (位置A)      │     (位置B)           │
│                 │                      │
│   直接做差分会产生大量错误运动            │
│                                        │
└────────────────────────────────────────┘

对齐后 (HomoMotion):
┌────────────────────────────────────────┐
│                                        │
│   Frame(t-1)    →    Frame(t)           │
│   (经H变换)     │                      │
│                 │                      │
│     物体 ───────┼────── 物体            │
│    (对齐到B)   │     (位置B)           │
│                 │                      │
│   只有真正的运动物体产生差异            │
│                                        │
└────────────────────────────────────────┘
```

### 4.3 详细结合步骤

#### Step 1: 创建运动信息提取模块

```python
# ultralytics/data/motion_extractor.py (新建)

import sys
sys.path.append('3rdparty/HomoMotion')

from homomotion.core.motion_detector import MotionDetector
from homomotion.utils.image_utils import ImageUtils
import numpy as np
import cv2

class HomoMotionExtractor:
    """
    HomoMotion 运动信息提取器
    用于从连续帧序列中提取精确对齐的运动图
    """

    def __init__(
        self,
        feature_type: str = 'ORB',  # 'ORB', 'SIFT', 'CUDA_SIFT'
        num_frames: int = 3,
        frame_interval: int = 1,
        diff_mode: str = 'mean',
        nfeatures: int = 2000,
        ransac_threshold: float = 5.0,
        output_channels: int = 3,  # 输出通道数: 1 或 3
        directional: bool = False,
        enable_enhancement: bool = True
    ):
        self.num_frames = num_frames
        self.output_channels = output_channels

        self.detector = MotionDetector(
            feature_type=feature_type,
            num_frames=num_frames,
            frame_interval=frame_interval,
            diff_mode=diff_mode,
            nfeatures=nfeatures,
            ransac_threshold=ransac_threshold,
            directional_motion=directional,
            enable_enhancement=enable_enhancement,
            debug=False
        )

        # 帧缓冲
        self.frame_buffer = {}

    def extract(self, current_frame: np.ndarray, seq_id: str = 'default') -> np.ndarray:
        """
        从当前帧提取运动图

        Args:
            current_frame: BGR 格式当前帧
            seq_id: 序列ID (用于区分不同视频流)

        Returns:
            motion_map: 运动图 (单通道或3通道)
        """
        # 更新帧缓冲
        if seq_id not in self.frame_buffer:
            self.frame_buffer[seq_id] = []

        self.frame_buffer[seq_id].append(current_frame)

        # 保持固定数量的帧
        max_frames = self.num_frames
        if len(self.frame_buffer[seq_id]) > max_frames:
            self.frame_buffer[seq_id].pop(0)

        frames = self.frame_buffer[seq_id]

        # 帧数不足时返回空白
        if len(frames) < self.num_frames:
            h, w = current_frame.shape[:2]
            if self.output_channels == 1:
                return np.zeros((h, w), dtype=np.uint8)
            else:
                return np.zeros((h, w, 3), dtype=np.uint8)

        # 处理获取运动图
        result = self.detector.process(frames)
        motion_map = result.motion_map

        # 确保输出通道数正确
        if self.output_channels == 1:
            if len(motion_map.shape) == 3:
                motion_map = cv2.cvtColor(motion_map, cv2.COLOR_BGR2GRAY)
        else:
            if len(motion_map.shape) == 2:
                # 单通道转伪彩色
                motion_map = cv2.applyColorMap(motion_map, cv2.COLOR_JET)
            elif motion_map.dtype == np.float32:
                # 归一化 float32 到 uint8
                motion_map = np.clip(motion_map * 255, 0, 255).astype(np.uint8)

        return motion_map

    def reset(self, seq_id: str = 'default'):
        """重置指定序列的帧缓冲"""
        if seq_id in self.frame_buffer:
            self.frame_buffer[seq_id] = []
```

#### Step 2: 修改数据集类

```python
# ultralytics/data/dataset_rgbmotion.py (新建)

from ultralytics.data import BaseDataset
from .motion_extractor import HomoMotionExtractor

class RGBMotionDataset(BaseDataset):
    """
    RGB + Motion 多模态数据集

    特点:
    1. 使用 HomoMotion 提取精确对齐的运动信息
    2. 运动图与 RGB 图像像素级对齐
    3. 支持实时运动提取和预计算运动图两种模式
    """

    def __init__(
        self,
        *args,
        motion_mode: str = 'extract',  # 'extract', 'load'
        motion_config: dict = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.motion_mode = motion_mode

        # 运动提取器配置
        if motion_config is None:
            motion_config = {
                'feature_type': 'ORB',      # ORB 速度快
                'num_frames': 3,
                'frame_interval': 1,
                'diff_mode': 'mean',
                'nfeatures': 2000,
                'output_channels': 3,
                'enable_enhancement': True
            }
        self.motion_config = motion_config

        if motion_mode == 'extract':
            self.motion_extractor = HomoMotionExtractor(**motion_config)
            self._frame_buffers = {}  # 每个序列的帧缓冲
        else:
            self.motion_extractor = None

    def load_and_preprocess_image(self, file_path, **kwargs):
        """加载并预处理 RGB + Motion 图像"""
        # 1. 加载 RGB 图像
        im_rgb = imread(file_path)  # BGR

        # 2. 获取/提取运动图
        if self.motion_mode == 'extract':
            # 从文件路径推断序列ID
            seq_id = self._get_sequence_id(file_path)
            frame_idx = self._get_frame_index(file_path)

            # 更新帧缓冲
            self._update_frame_buffer(seq_id, frame_idx, im_rgb)

            # 提取运动图
            motion_map = self.motion_extractor.extract(im_rgb, seq_id)

        elif self.motion_mode == 'load':
            # 加载预计算的运动图
            motion_path = self._get_motion_path(file_path)
            motion_map = imread(motion_path)

            if len(motion_map.shape) == 2:
                motion_map = cv2.cvtColor(motion_map, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Unknown motion_mode: {self.motion_mode}")

        # 3. 合并 RGB + Motion
        im = self._merge_rgb_motion(im_rgb, motion_map)

        return im

    def _merge_rgb_motion(self, rgb, motion):
        """合并 RGB 和 Motion 通道"""
        h_rgb, w_rgb = rgb.shape[:2]
        h_m, w_m = motion.shape[:2]

        # 尺寸对齐
        if h_rgb != h_m or w_rgb != w_m:
            motion = cv2.resize(motion, (w_rgb, h_rgb))

        # 早期融合: 6通道
        return np.concatenate([rgb, motion], axis=2)
```

#### Step 3: 图像对准细粒度处理

```python
# 关键: 确保 HomoMotion 的对准质量

class MotionQualityController:
    """
    运动提取质量控制器
    确保图像对准的精确性
    """

    def __init__(self):
        self.min_inliers = 10
        self.max_reproj_error = 5.0
        self.min_inlier_ratio = 0.3

    def validate_homography(self, H_stats: dict) -> bool:
        """
        验证单应性矩阵质量

        检查项:
        1. 内点数量 >= min_inliers
        2. 重投影误差 <= max_reproj_error
        3. 内点比例 >= min_inlier_ratio
        """
        if H_stats is None:
            return False

        num_inliers = H_stats.get('num_inliers', 0)
        inlier_ratio = H_stats.get('inlier_ratio', 0)

        # 条件数检查
        cond = H_stats.get('condition_number', float('inf'))
        if cond > 1e6:
            return False

        return (
            num_inliers >= self.min_inliers and
            inlier_ratio >= self.min_inlier_ratio
        )

    def get_quality_mask(self, result: MotionResult) -> np.ndarray:
        """
        获取运动图质量掩码

        基于:
        1. 有效区域 (valid_mask)
        2. 单应性矩阵质量
        """
        quality_mask = result.valid_mask.copy()

        # 可选: 基于匹配质量进一步筛选
        if result.match_info:
            for info in result.match_info:
                if not self.validate_match_info(info):
                    # 降低该帧的权重
                    quality_mask = cv2.bitwise_and(
                        quality_mask,
                        np.zeros_like(quality_mask)
                    )

        return quality_mask
```

#### Step 4: 批量预处理工具

```python
# tools/precompute_motion_maps.py (批量预处理工具)

"""
批量预处理运动图
用于预先计算并保存运动图，加速训练
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from homomotion.core.motion_detector import MotionDetector
from concurrent.futures import ThreadPoolExecutor

def process_sequence(seq_dir: Path, output_dir: Path, config: dict):
    """处理单个序列"""
    detector = MotionDetector(**config)

    image_files = sorted(seq_dir.glob('*.jpg')) + sorted(seq_dir.glob('*.png'))

    frames = []
    for img_path in image_files:
        frames.append(cv2.imread(str(img_path)))

        if len(frames) >= config['num_frames']:
            # 处理
            result = detector.process(frames[-config['num_frames']:])

            # 保存运动图
            output_path = output_dir / f"{img_path.stem}_motion.png"
            cv2.imwrite(str(output_path), result.motion_map)

            frames.pop(0)  # 滑动窗口

def batch_process(input_root: Path, output_root: Path, config: dict, num_workers: int = 4):
    """批量处理所有序列"""
    sequences = [d for d in input_root.iterdir() if d.is_dir()]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for seq_dir in sequences:
            output_seq_dir = output_root / seq_dir.name
            output_seq_dir.mkdir(parents=True, exist_ok=True)

            future = executor.submit(
                process_sequence, seq_dir, output_seq_dir, config
            )
            futures.append(future)

        for f in tqdm(futures):
            f.result()
```

---

## 五、实施清单

### 5.1 Phase 1: 基础集成 (Week 1-2)

| 任务 | 状态 | 说明 |
|------|------|------|
| 创建 motion_extractor.py | TODO | HomoMotion 封装类 |
| 创建 dataset_rgbmotion.py | TODO | RGBMotion 数据集类 |
| 修改 BaseDataset | TODO | 添加 motion 相关参数 |
| 测试运动提取 | TODO | 验证运动图质量 |

### 5.2 Phase 2: 对齐优化 (Week 2-3)

| 任务 | 状态 | 说明 |
|------|------|------|
| 实现质量控制器 | TODO | 验证单应性矩阵质量 |
| 实现降级策略 | TODO | 对准失败时的处理 |
| 优化有效区域计算 | TODO | 精确的重叠区域 |

### 5.3 Phase 3: 性能优化 (Week 3-4)

| 任务 | 状态 | 说明 |
|------|------|------|
| 实现帧缓冲管理 | TODO | 高效的序列处理 |
| 添加 CUDA-SIFT 支持 | TODO | GPU 加速 (可选) |
| 批量预处理工具 | TODO | 离线运动图计算 |

### 5.4 Phase 4: 融合验证 (Week 4-6)

| 任务 | 状态 | 说明 |
|------|------|------|
| 数据集验证 | TODO | RGB + Motion 训练 |
| 对比实验 | TODO | vs RGB-only, vs RGBT |
| 消融实验 | TODO | 不同运动提取参数 |

---

## 六、关键代码位置

### 6.1 HomoMotion 核心文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `homomotion/core/motion_detector.py` | 1-423 | MotionDetector 主类 |
| `homomotion/core/feature_extractor.py` | 1-358 | 特征提取器 |
| `homomotion/core/feature_matcher.py` | 1-197 | 特征匹配器 |
| `homomotion/core/homography.py` | 1-256 | 单应性估计 |
| `homomotion/utils/image_utils.py` | 1-596 | 图像处理工具 |

### 6.2 YOLOv11-RGBT 参考文件

| 文件 | 行数 | 功能 |
|------|------|------|
| `ultralytics/data/base.py` | 177-226 | load_and_preprocess_image |
| `ultralytics/data/base.py` | 248-257 | _merge_channels |

### 6.3 新建文件清单

```
ultralytics/
├── data/
│   ├── motion_extractor.py      # HomoMotion 封装
│   └── dataset_rgbmotion.py    # RGBMotion 数据集

tools/
└── precompute_motion_maps.py   # 批量预处理工具

cfg/
└── models/master/rgbmotion/
    └── yolo-master-rgbmotion-n.yaml  # 模型配置
```

---

## 七、参数调优建议

### 7.1 运动提取参数

```python
# 推荐配置 (精度优先)
config_high_quality = {
    'feature_type': 'SIFT',       # 或 'CUDA_SIFT'
    'num_frames': 3,
    'frame_interval': 1,
    'diff_mode': 'mean',
    'nfeatures': 3000,
    'ransac_threshold': 3.0,
    'min_matches': 20,
    'ratio_threshold': 0.75,
}

# 推荐配置 (速度优先)
config_fast = {
    'feature_type': 'ORB',
    'num_frames': 3,
    'frame_interval': 1,
    'diff_mode': 'mean',
    'nfeatures': 1000,
    'ransac_threshold': 7.0,
    'min_matches': 8,
}
```

### 7.2 融合参数

```python
# 输出通道配置
output_channels = 3  # 推荐: 3通道伪彩色运动图

# 或单通道
output_channels = 1  # 节省通道，但信息损失

# 运动增强参数
motion_config = {
    'noise_threshold': 10.0,     # 背景噪声阈值
    'saturation_val': 60.0,      # 饱和值
    'enhancement_power': 2.0,    # 增强幂次
}
```

---

*文档版本: v1.0*
*最后更新: 2026-03-22*
