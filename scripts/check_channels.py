import argparse
from PIL import Image

def get_image_channels(image_path):
    """
    获取图片的通道数
    :param image_path: 图片路径
    :return: 通道数 (1=灰度, 3=RGB, 4=RGBA)
    """
    try:
        with Image.open(image_path) as img:
            return len(img.getbands())
    except Exception as e:
        return f"错误：{str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python 检测图片通道数（灰度1/RGB3/RGBA4）")
    parser.add_argument("image", help="图片路径")
    
    args = parser.parse_args()
    channels = get_image_channels(args.image)
    print(channels)