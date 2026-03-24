#!/usr/bin/env python3
"""
检验 YOLO 风格数据集目录下图片与标签是否一一对应（按文件名 stem 匹配）。

目录结构示例（每个 split 下）::
    train/
        images/
        images_pevent/   # 可选，与其它图片目录一并视为“有图”
        labels/          # *.txt，stem 与图片相同

- 有标签无图片：labels 中存在 *.txt，但在所有图片子目录中找不到同 stem 的图片。
- 有图片无标签：某图片子目录中存在图片，但 labels 中无同 stem 的 *.txt。

不匹配的文件会分别复制到用户指定的两个输出目录，并在各自目录下生成清单 txt。

用法示例::

    python scripts/check_image_label_pairs.py \\
        --root RGB+X-Dataset \\
        --out-label-no-image ./tmp/label_without_image \\
        --out-image-no-label ./tmp/image_without_label

依赖：仅 Python 标准库。
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path


# 常见图片后缀（小写比较）
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jp2", ".pbm", ".pgm", ".ppm"}
)


def collect_images_by_stem(image_dirs: list[Path]) -> dict[str, list[Path]]:
    """stem -> 该 stem 下所有图片路径（可能来自多个子目录）。"""
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for d in image_dirs:
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf not in IMAGE_EXTENSIONS:
                continue
            by_stem[p.stem].append(p.resolve())
    return dict(by_stem)


def collect_labels_by_stem(labels_dir: Path) -> dict[str, Path]:
    """stem -> 标签 txt 路径（每个 stem 只保留一个，若重复则取字典序第一个并警告）。"""
    by_stem: dict[str, list[Path]] = defaultdict(list)
    if not labels_dir.is_dir():
        return {}
    for p in labels_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".txt":
            continue
        by_stem[p.stem].append(p.resolve())
    out: dict[str, Path] = {}
    for stem, paths in by_stem.items():
        paths_sorted = sorted(paths)
        out[stem] = paths_sorted[0]
        if len(paths_sorted) > 1:
            print(f"  警告: stem={stem!r} 在 labels 中有 {len(paths_sorted)} 个 txt，仅使用 {paths_sorted[0]}")
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_with_layout(src: Path, dataset_root: Path, dest_root: Path) -> Path:
    """
    将 src 复制到 dest_root / relative_to_dataset_root(src)。
    例如: root/train/labels/a.txt -> dest_root/train/labels/a.txt
    """
    rel = src.resolve().relative_to(dataset_root.resolve())
    dst = dest_root / rel
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def process_split(
    split: str,
    dataset_root: Path,
    image_subdirs: list[str],
    out_label_no_image: Path,
    out_image_no_label: Path,
    list_label_no_image: list[str],
    list_image_no_label: list[str],
    dry_run: bool,
) -> tuple[int, int]:
    """
    返回 (有标签无图片数量, 有图片无标签数量)。
    """
    split_root = dataset_root / split
    labels_dir = split_root / "labels"
    image_dirs = [split_root / sub for sub in image_subdirs]

    images_by_stem = collect_images_by_stem(image_dirs)
    labels_by_stem = collect_labels_by_stem(labels_dir)

    image_stems = set(images_by_stem.keys())
    label_stems = set(labels_by_stem.keys())

    # 有标签无图片
    orphan_label_stems = sorted(label_stems - image_stems)
    n_orphan_labels = 0
    for stem in orphan_label_stems:
        src = labels_by_stem[stem]
        n_orphan_labels += 1
        line = str(src)
        list_label_no_image.append(line)
        if dry_run:
            continue
        dst = copy_with_layout(src, dataset_root, out_label_no_image)
        list_label_no_image[-1] = f"{src}\t->\t{dst}"

    # 有图片无标签
    orphan_image_stems = sorted(image_stems - label_stems)
    n_orphan_images = 0
    for stem in orphan_image_stems:
        for src in sorted(images_by_stem[stem]):
            n_orphan_images += 1
            list_image_no_label.append(str(src))
            if dry_run:
                continue
            dst = copy_with_layout(src, dataset_root, out_image_no_label)
            list_image_no_label[-1] = f"{src}\t->\t{dst}"

    return n_orphan_labels, n_orphan_images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检验 train/val/test 下图片与 labels 是否按 stem 对应，并复制不匹配项到指定目录。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="数据集根目录（其下含 train、val、test 等 split）",
    )
    parser.add_argument(
        "--out-label-no-image",
        type=Path,
        required=True,
        help="「有标签无图片」时，将孤立标签 txt 复制到此目录（会保留 train/val/test/labels/... 结构）",
    )
    parser.add_argument(
        "--out-image-no-label",
        type=Path,
        required=True,
        help="「有图片无标签」时，将孤立图片复制到此目录（会保留 train/val/test/images/... 结构）",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="要检查的 split，逗号分隔，默认 train,val,test",
    )
    parser.add_argument(
        "--image-dirs",
        type=str,
        default="images,images_pevent",
        help="每个 split 下作为图片目录的子文件夹名，逗号分隔；默认 images,images_pevent。若只需 images 可写 images",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计与生成清单，不复制文件",
    )
    parser.add_argument(
        "--manifest-label-no-image",
        type=str,
        default="orphan_labels.txt",
        help="有标签无图片清单文件名（写在 --out-label-no-image 目录下）",
    )
    parser.add_argument(
        "--manifest-image-no-label",
        type=str,
        default="orphan_images.txt",
        help="有图片无标签清单文件名（写在 --out-image-no-label 目录下）",
    )
    args = parser.parse_args()

    dataset_root = args.root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"数据集根目录不存在或不是目录: {dataset_root}")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    image_subdirs = [s.strip() for s in args.image_dirs.split(",") if s.strip()]
    if not splits:
        raise SystemExit("未指定有效的 --splits")

    out_label_no_image = args.out_label_no_image.expanduser().resolve()
    out_image_no_label = args.out_image_no_label.expanduser().resolve()

    list_label_no_image: list[str] = []
    list_image_no_label: list[str] = []

    print(f"数据集根目录: {dataset_root}")
    print(f"检查 splits: {splits}")
    print(f"图片子目录（每个 split 下）: {image_subdirs}")
    if args.dry_run:
        print("模式: dry-run（不复制文件）")
    print()

    total_labels = 0
    total_images = 0
    for sp in splits:
        if not (dataset_root / sp).is_dir():
            print(f"跳过不存在的 split: {sp}")
            continue
        n_l, n_i = process_split(
            sp,
            dataset_root,
            image_subdirs,
            out_label_no_image,
            out_image_no_label,
            list_label_no_image,
            list_image_no_label,
            args.dry_run,
        )
        total_labels += n_l
        total_images += n_i
        print(f"[{sp}] 有标签无图片: {n_l} 个文件; 有图片无标签: {n_i} 个文件")

    # 写清单（UTF-8）
    if not args.dry_run:
        ensure_dir(out_label_no_image)
        ensure_dir(out_image_no_label)

    man_l = out_label_no_image / args.manifest_label_no_image
    man_i = out_image_no_label / args.manifest_image_no_label

    man_l.parent.mkdir(parents=True, exist_ok=True)
    man_i.parent.mkdir(parents=True, exist_ok=True)

    man_l.write_text(
        "\n".join(list_label_no_image) + ("\n" if list_label_no_image else ""),
        encoding="utf-8",
    )
    man_i.write_text(
        "\n".join(list_image_no_label) + ("\n" if list_image_no_label else ""),
        encoding="utf-8",
    )

    print()
    print(f"合计 — 有标签无图片: {total_labels}; 有图片无标签: {total_images}")
    print(f"清单已写入:\n  {man_l}\n  {man_i}")
    if args.dry_run and (total_labels or total_images):
        print("（dry-run：上述路径下仅写了清单；未复制文件。去掉 --dry-run 后执行复制。）")


if __name__ == "__main__":
    main()
