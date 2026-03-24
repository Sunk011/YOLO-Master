import os
import argparse
from PIL import Image

def overlay_images(img1_path, img2_path, alpha, output_dir):
    """叠加图片：img2 透明叠加在 img1 上"""
    img1 = Image.open(img1_path).convert("RGBA")
    img2 = Image.open(img2_path).convert("RGBA")
    img2 = img2.resize(img1.size, Image.Resampling.LANCZOS)

    combined = Image.blend(img1, img2, alpha=alpha)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"overlay_{os.path.basename(img1_path).split('.')[0]}_{os.path.basename(img2_path).split('.')[0]}.png"
    save_path = os.path.join(output_dir, filename)
    combined.convert("RGB").save(save_path)
    return save_path

def concat_images(img1_path, img2_path, output_dir, direction="leftright"):
    """
    拼接图片：不改变像素，直接拼接
    direction: leftright（左右）/ updown（上下）
    """
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    if direction == "leftright":  # 左右拼接
        new_width = img1.width + img2.width
        new_height = max(img1.height, img2.height)
        new_img = Image.new("RGB", (new_width, new_height))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))
    else:  # 上下拼接
        new_width = max(img1.width, img2.width)
        new_height = img1.height + img2.height
        new_img = Image.new("RGB", (new_width, new_height))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (0, img1.height))

    os.makedirs(output_dir, exist_ok=True)
    filename = f"concat_{direction}_{os.path.basename(img1_path).split('.')[0]}_{os.path.basename(img2_path).split('.')[0]}.png"
    save_path = os.path.join(output_dir, filename)
    new_img.save(save_path)
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图片叠加 / 拼接工具")
    parser.add_argument("img1", help="底图/第一张图路径")
    parser.add_argument("img2", help="叠加图/第二张图路径")
    parser.add_argument("--alpha", type=float, default=0.3, help="透明度 0~1（仅叠加模式使用）")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    # 👇 这里按你要求修改
    parser.add_argument("--mode", required=True, choices=["overlay", "concat_leftright", "concat_updown"], 
                        help="模式：overlay=叠加，concat_leftright=左右拼接，concat_updown=上下拼接")

    args = parser.parse_args()

    if args.mode == "overlay":
        result = overlay_images(args.img1, args.img2, args.alpha, args.output_dir)
        print(f"✅ 叠加完成：{result}")

    elif args.mode == "concat_leftright":
        result = concat_images(args.img1, args.img2, args.output_dir, direction="leftright")
        print(f"✅ 左右拼接完成：{result}")

    elif args.mode == "concat_updown":
        result = concat_images(args.img1, args.img2, args.output_dir, direction="updown")
        print(f"✅ 上下拼接完成：{result}")