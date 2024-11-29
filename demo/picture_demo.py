import os
import sys
import cv2
import argparse
import numpy as np
import torch

from lib.network.rtpose_vgg import get_model
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='../experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str, default='pose_model.pth')
parser.add_argument('opts', help="Modify config options using the command-line",
                    default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

model = get_model('vgg19')
model.load_state_dict(torch.load(args.weight))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

test_image = 'test.jpg'
oriImg = cv2.imread(test_image)  # B,G,R order
shape_dst = np.min(oriImg.shape[0:2])
def visualize_heatmaps(oriImg, heatmaps):
    # 上采样热力图到原图大小
    heatmap_upsamp = cv2.resize(
        heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    for i in range(heatmap_upsamp.shape[2]):  # 假设 heatmaps 维度为 [H, W, C]
        # 归一化热力图到 [0, 255]
        heatmap_norm = cv2.normalize(heatmap_upsamp[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
        overlayed_img = cv2.addWeighted(oriImg, 0.5, heatmap_colored, 0.5, 0)

        # 保存叠加结果
        cv2.imwrite(f"heatmap.jpg", overlayed_img)


def visualize_paf(oriImg, pafs):
    # 上采样 PAF 到原图大小
    paf_upsamp = cv2.resize(
        pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    # 绘制每对关节的矢量场
    num_limbs = paf_upsamp.shape[2] // 2  # 每对关节有 x 和 y 两个通道
    for i in range(num_limbs):
        # 提取 x 和 y 的向量场
        paf_x = paf_upsamp[:, :, 2 * i]
        paf_y = paf_upsamp[:, :, 2 * i + 1]

        # 在每隔固定步长的位置绘制箭头
        step = 10
        for y in range(0, paf_x.shape[0], step):
            for x in range(0, paf_x.shape[1], step):
                start_point = (x, y)
                end_point = (int(x + paf_x[y, x] * step), int(y + paf_y[y, x] * step))
                cv2.arrowedLine(oriImg, start_point, end_point, (0, 255, 0), 1, tipLength=0.3)

        # 保存矢量图结果
        cv2.imwrite(f"paf.jpg", oriImg)


# Get results of original image
with torch.no_grad():
    paf, heatmap, im_scale = get_outputs(oriImg, model, 'rtpose')

visualize_heatmaps(oriImg.copy(), heatmap)  # 可视化关键点热力图
visualize_paf(oriImg.copy(), paf)  # 可视化姿态向量图

print(im_scale)
humans = paf_to_pose_cpp(heatmap, paf, cfg)

out = draw_humans(oriImg, humans)
cv2.imwrite('test-output.jpg', out)
