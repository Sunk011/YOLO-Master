"""
可视化每个类别的目标样本：从数据集中随机抽取每个类别的目标，
裁剪为正方形并拼成 5x5 的大图，按类别分别保存。

用法：
    python scripts/visualize_class_samples.py --yaml RGB+X-Dataset/RGB+X.yaml --output tmp
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def parse_image_paths(index_file: Path, dataset_root: Path) -> list[Path]:
    """从 index 文件（每行是相对路径）解析出完整图片路径列表。"""
    paths = []
    with open(index_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # train.txt 中的路径是相对于 dataset root 的
            paths.append(dataset_root / line)
    return paths


def load_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    加载 YOLO 格式标签，返回 [(class_id, cx, cy, w, h), ...]。
    """
    annotations = []
    if not label_path.exists():
        return annotations
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            annotations.append((int(cls), cx, cy, w, h))
    return annotations


def crop_object(img: np.ndarray, cx: float, cy: float, w: float, h: float,
                 padding: float = 0.5) -> np.ndarray | None:
    """
    从图片中裁剪出目标区域，扩展 padding 倍的 bbox 宽高，
    并将裁剪区域 resize 为正方形返回。

    img: HWC, BGR
    cx/cy/w/h: 归一化坐标（0~1）
    padding: 在 bbox 基础上额外扩展的比例（0.5 表示宽高各扩展 50%）

    Returns: 正方形 BGR 图片，或 None（超出边界时）
    """
    H, W = img.shape[:2]

    # 转换为像素坐标
    x1 = int((cx - w * (0.5 + padding)) * W)
    y1 = int((cy - h * (0.5 + padding)) * H)
    x2 = int((cx + w * (0.5 + padding)) * W)
    y2 = int((cy + h * (0.5 + padding)) * H)

    # 限制在图片范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    square = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    return square


def stitch_grid(images: list[np.ndarray], cols: int = 5) -> np.ndarray:
    """
    将一组图片排列成 cols 列的网格，返回拼接后的大图（无边框）。
    images 数量不足 25 时，左侧和底部会留空（填黑）。

    Args:
        images: BGR 图片列表（须同尺寸）
        cols:   列数，默认为 5（5x5=25 张）

    Returns:
        拼接好的 BGR 大图
    """
    rows = int(np.ceil(len(images) / cols))
    h, w = images[0].shape[:2] if images else (224, 224)
    channels = images[0].shape[2] if images else 3

    grid = np.zeros((rows * h, cols * w, channels), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

    return grid


def collect_samples(image_paths: list[Path], class_id: int,
                    target_count: int, rng: random.Random) -> list[np.ndarray]:
    """
    从图片列表中随机抽取包含指定类别的目标，裁剪并返回最多 target_count 张图片。

    Returns:
        list of BGR 正方形图片（已 resize 到 224x224）
    """
    samples = []
    shuffled = image_paths.copy()
    rng.shuffle(shuffled)

    for img_path in shuffled:
        if len(samples) >= target_count:
            break

        label_path = img_path.with_suffix(".txt")
        annotations = load_labels(label_path)

        for (cid, cx, cy, w, h) in annotations:
            if cid != class_id:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            crop = crop_object(img, cx, cy, w, h, padding=0.5)
            if crop is None:
                continue

            samples.append(crop)
            break  # 每张原图最多取一个该类别目标

    return samples


def main():
    parser = argparse.ArgumentParser(description="按类别可视化目标样本")
    parser.add_argument("--yaml", type=str, default="RGB+X-Dataset/RGB+X.yaml",
                        help="数据集 YAML 配置文件路径")
    parser.add_argument("--output", type=str, default="tmp",
                        help="输出目录（默认为 tmp）")
    parser.add_argument("--num", type=int, default=25,
                        help="每个类别抽取的目标数量（默认 25）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42）")
    parser.add_argument("--size", type=int, default=224,
                        help="每张裁剪目标图的边长（默认 224）")
    parser.add_argument("--cols", type=int, default=5,
                        help="网格列数（默认 5，即 5x5）")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="从哪些子集取样，逗号分隔（默认 train,val,test）")
    args = parser.parse_args()

    yaml_data = load_yaml(args.yaml)
    dataset_root = Path(yaml_data["path"]).expanduser().resolve()
    class_names = yaml_data.get("names", {})
    splits = [s.strip() for s in args.splits.split(",")]

    print(f"数据集根目录: {dataset_root}")
    print(f"类别数量: {len(class_names)}")

    # 合并所有 split 的图片路径（去重）
    all_paths: list[Path] = []
    for split in splits:
        key = "test" if split == "test" else split  # yaml 中 test 字段名就是 test
        if key not in yaml_data:
            continue
        idx_file = dataset_root / yaml_data[key]
        if not idx_file.exists():
            print(f"警告: 索引文件不存在 {idx_file}，跳过")
            continue
        paths = parse_image_paths(idx_file, dataset_root)
        for p in paths:
            if p not in all_paths:
                all_paths.append(p)
        print(f"  {split}: {len(paths)} 张图片")

    if not all_paths:
        raise RuntimeError("未找到任何图片路径，请检查 YAML 配置。")

    # 收集每个类别的样本
    rng = random.Random(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_id, name in sorted(class_names.items()):
        print(f"\n处理类别 {class_id}: {name} ...")
        samples = collect_samples(all_paths, class_id, args.num, rng)
        print(f"  收集到 {len(samples)} 张样本")

        if not samples:
            print(f"  警告: 类别 {class_id} 没有找到任何目标，跳过。")
            continue

        grid = stitch_grid(samples, cols=args.cols)

        out_name = f"class_{class_id:03d}"
        if name:
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in str(name).strip())
            out_name += f"_{safe_name}"
        out_path = output_dir / f"{out_name}.jpg"

        cv2.imwrite(str(out_path), grid)
        print(f"  已保存: {out_path}")

    print(f"\n完成！共 {len(class_names)} 张类别样本图，输出至: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
