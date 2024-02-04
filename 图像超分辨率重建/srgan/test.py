from utils import *
from models import Generator
import time
from PIL import Image
import torch

# 定义模型参数
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 4 # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 预训练模型
    srgan_checkpoint = "./results/checkpoint_srgan.pth"

    # 加载模型 Generator
    generator = Generator(large_kernel_size=large_kernel_size,
                          small_kernel_size=small_kernel_size,
                          n_channels=n_channels,
                          n_blocks=n_blocks,
                          scaling_factor=scaling_factor)

    # 将模型移至 GPU（如果可用）
    if torch.cuda.is_available():
        generator = generator.to('cuda')

    # 加载模型权重
    checkpoint = torch.load(srgan_checkpoint, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()

    # 加载图像
    imgPath = '/Users/lvzhongming/Desktop/视觉/图像超分辨率重建/image/image_15.png'
    img = Image.open(imgPath, mode='r')
     #img = img.convert('RGB')

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    # 将图像移至 GPU（如果可用）
    lr_img = lr_img.to(device)

    # 模型推理
with torch.no_grad():
    sr_img = generator(lr_img).squeeze(0).cpu().detach()
    sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')

    # 将彩色图像转换为灰度
    sr_img_gray = sr_img.convert('L')


    sr_img_gray.save('./results/test_srgan2.jpg')
    dirname_read='/Users/lvzhongming/Desktop/视觉/图像超分辨率重建/image/image_15.png'
    names

