"""
RGB+X 多模态数据集训练脚本
用于在 RGB+X 数据集上训练 YOLO-Master 模型

数据集包含: RGB图像 + 额外模态（深度/事件/红外等）
类别: person, car, bicycle, bus
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def train_rgb_x(
    model_config: str = "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml",
    data_config: str = "RGB+X-Dataset/RGB+X.yaml",
    epochs: int = 300,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "rgb_x_experiment",
    exist_ok: bool = False,
    workers: int = 8,
    patience: int = 50,
    save: bool = True,
    save_period: int = -1,
    cache: bool = False,
    pretrained: bool = True,
    optimizer: str = "SGD",
    verbose: bool = True,
    seed: int = 0,
    deterministic: bool = True,
    single_cls: bool = False,
    rect: bool = False,
    cos_lr: bool = False,
    close_mosaic: int = 10,
    resume: bool = False,
    amp: bool = True,
    fraction: float = 1.0,
    profile: bool = False,
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    warmup_momentum: float = 0.8,
    warmup_bias_lr: float = 0.1,
    box: float = 7.5,
    cls: float = 0.5,
    dfl: float = 1.5,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    flipud: float = 0.0,
    fliplr: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
):
    """
    训练 RGB+X 多模态数据集

    Args:
        model_config: 模型配置文件路径
        data_config: 数据集配置文件路径
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 输入图像尺寸
        device: 训练设备 (e.g., "0" for GPU 0, "cpu", "0,1,2,3" for multi-GPU)
        project: 项目保存路径
        name: 实验名称
        exist_ok: 如果已存在是否覆盖
        workers: 数据加载线程数
        patience: 早停耐心值
        save: 是否保存模型
        save_period: 模型保存周期 (-1 表示只在最后保存)
        cache: 是否缓存图像
        pretrained: 是否使用预训练权重
        optimizer: 优化器选择 (SGD, Adam, AdamW)
        verbose: 是否详细输出
        seed: 随机种子
        deterministic: 是否使用确定性模式
        single_cls: 是否单类别训练
        rect: 是否使用矩形训练
        cos_lr: 是否使用余弦学习率调度
        close_mosaic: 最后多少轮关闭mosaic增强
        resume: 是否恢复训练
        amp: 是否使用自动混合精度
        fraction: 使用数据集的比例
        profile: 是否使用 profiler
        lr0: 初始学习率
        lrf: 最终学习率 (lr0 * lrf)
        momentum: SGD 动量 / Adam beta1
        weight_decay: 权重衰减
        warmup_epochs: 预热轮数
        warmup_momentum: 预热动量
        warmup_bias_lr: 预热偏置学习率
        box: 边界框损失权重
        cls: 分类损失权重
        dfl: DFL 损失权重
        hsv_h: HSV-H 增强
        hsv_s: HSV-S 增强
        hsv_v: HSV-V 增强
        degrees: 旋转角度增强
        translate: 平移增强
        scale: 缩放增强
        shear: 剪切增强
        perspective: 透视增强
        flipud: 上下翻转
        fliplr: 左右翻转
        mosaic: mosaic 增强
        mixup: mixup 增强
        copy_paste: copy-paste 增强

    Returns:
        训练结果对象
    """
    # 确保数据集配置存在
    data_path = PROJECT_ROOT / data_config
    if not data_path.exists():
        print(f"警告: 数据集配置文件不存在: {data_path}")
        print("尝试使用绝对路径...")

    # 解析设备
    if device == "cuda" or device == "gpu":
        device = "0"

    print("=" * 60)
    print("RGB+X 多模态数据集训练配置")
    print("=" * 60)
    print(f"模型配置: {model_config}")
    print(f"数据集配置: {data_config}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch}")
    print(f"图像尺寸: {imgsz}")
    print(f"训练设备: {device}")
    print(f"实验名称: {name}")
    print("=" * 60)

    # 加载模型
    print("正在加载模型...")
    model = YOLO(model_config)

    # 开始训练
    print("开始训练...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        exist_ok=exist_ok,
        workers=workers,
        patience=patience,
        save=save,
        save_period=save_period,
        cache=cache,
        pretrained=pretrained,
        optimizer=optimizer,
        verbose=verbose,
        seed=seed,
        deterministic=deterministic,
        single_cls=single_cls,
        rect=rect,
        cos_lr=cos_lr,
        close_mosaic=close_mosaic,
        resume=resume,
        amp=amp,
        fraction=fraction,
        profile=profile,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        warmup_momentum=warmup_momentum,
        warmup_bias_lr=warmup_bias_lr,
        box=box,
        cls=cls,
        dfl=dfl,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        shear=shear,
        perspective=perspective,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
    )

    return results


def train_with_moe(
    data_config: str = "RGB+X-Dataset/RGB+X.yaml",
    epochs: int = 300,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "rgb_x_moe_experiment",
    moe_num_experts: int = 8,
    moe_top_k: int = 2,
    moe_balance_loss: float = 0.01,
    **kwargs,
):
    """
    使用 MoE (Mixture of Experts) 模式训练 RGB+X 数据集

    Args:
        data_config: 数据集配置文件路径
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 输入图像尺寸
        device: 训练设备
        project: 项目保存路径
        name: 实验名称
        moe_num_experts: MoE 专家数量
        moe_top_k: 每次激活的 top-k 专家
        moe_balance_loss: MoE 平衡损失权重
        **kwargs: 其他训练参数

    Returns:
        训练结果对象
    """
    model_config = "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml"

    print("=" * 60)
    print("RGB+X MoE 训练配置")
    print("=" * 60)
    print(f"MoE 专家数量: {moe_num_experts}")
    print(f"Top-K 激活专家: {moe_top_k}")
    print(f"MoE 平衡损失权重: {moe_balance_loss}")
    print("=" * 60)

    print("正在加载模型...")
    model = YOLO(model_config)

    print("开始 MoE 训练...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
        moe_balance_loss=moe_balance_loss,
        **kwargs,
    )

    return results


def train_with_lora(
    data_config: str = "RGB+X-Dataset/RGB+X.yaml",
    epochs: int = 300,
    batch: int = 16,
    imgsz: int = 640,
    device: str = "0",
    project: str = "runs/train",
    name: str = "rgb_x_lora_experiment",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_gradient_checkpointing: bool = True,
    **kwargs,
):
    """
    使用 LoRA (Low-Rank Adaptation) 模式训练 RGB+X 数据集

    Args:
        data_config: 数据集配置文件路径
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 输入图像尺寸
        device: 训练设备
        project: 项目保存路径
        name: 实验名称
        lora_r: LoRA rank
        lora_alpha: LoRA alpha (通常设置为 2*r)
        lora_dropout: LoRA dropout
        lora_gradient_checkpointing: 是否使用梯度检查点
        **kwargs: 其他训练参数

    Returns:
        训练结果对象
    """
    print("=" * 60)
    print("RGB+X LoRA 训练配置")
    print("=" * 60)
    print(f"LoRA Rank: {lora_r}")
    print(f"LoRA Alpha: {lora_alpha}")
    print(f"LoRA Dropout: {lora_dropout}")
    print(f"梯度检查点: {lora_gradient_checkpointing}")
    print("=" * 60)

    # 使用 YOLO11 作为基础模型
    model = YOLO("yolo11n.pt")

    print("开始 LoRA 训练...")
    results = model.train(
        data=data_config,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_gradient_checkpointing=lora_gradient_checkpointing,
        **kwargs,
    )

    # 保存 LoRA 适配器
    lora_path = f"{project}/{name}/weights/best_lora.pt"
    model.save_lora_only(lora_path)
    print(f"LoRA 适配器已保存至: {lora_path}")

    return results


def main():
    """主函数 - 运行默认训练配置"""
    import argparse

    parser = argparse.ArgumentParser(description="RGB+X 多模态数据集训练")
    parser.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=["normal", "moe", "lora"],
        help="训练模式: normal (标准), moe (混合专家), lora (低秩适配)",
    )
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--device", type=str, default="0", help="训练设备")
    parser.add_argument("--name", type=str, default="rgb_x_train", help="实验名称")

    args = parser.parse_args()

    if args.mode == "moe":
        train_with_moe(
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            name=args.name,
        )
    elif args.mode == "lora":
        train_with_lora(
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            name=args.name,
        )
    else:
        train_rgb_x(
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            name=args.name,
        )


if __name__ == "__main__":
    main()
