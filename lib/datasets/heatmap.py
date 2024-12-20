# coding=utf-8

import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage


"""Implement the generate of every channel of ground truth heatmap.
:param centerA: int with shape (2,), every coordinate of person's keypoint.
:param accumulate_confid_map: one channel of heatmap, which is accumulated, 
       np.log(100) is the max value of heatmap.
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""


def putGaussianMaps(center, accumulate_confid_map, sigma, grid_y, grid_x, stride):

    start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2  # 计算每个点距离中心点的距离
    exponent = d2 / 2.0 / sigma / sigma  # 计算高斯
    mask = exponent <= 4.6052  # 如果这个点距离中心点高斯超过 就置为0
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map += cofid_map  # 多个点的热力图会叠加的
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0  # 若累加之后值大于1了 就置为1

    return accumulate_confid_map
