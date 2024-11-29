# coding=utf-8
"""Implement Part Affinity Fields
:param centerA: int with shape (2,), centerA will pointed by centerB.
:param centerB: int with shape (2,), centerB will point to centerA.
:param accumulate_vec_map: one channel of paf.
:param count: store how many pafs overlaped in one coordinate of accumulate_vec_map.
:param params_transform: store the value of stride and crop_szie_y, crop_size_x                 
"""
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc, ndimage


def putVecMaps(centerA, centerB, accumulate_vec_map, count, grid_y, grid_x, stride):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    thre = 1  # 躯干宽度
    centerB = centerB / stride  # 映射到特征图中
    centerA = centerA / stride

    limb_vec = centerB - centerA  # A->B的向量  但是我们需要的是方向
    norm = np.linalg.norm(limb_vec)  # 求范数
    if norm == 0.0:
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm  # 单位向量
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)  # 得到躯干大致的区域  是个包含躯干的长方形
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # 计算区域上每个点到A点的距离  # the vector from (x,y) to centerA 根据位置判断是否在该区域上（分别得到X和Y方向的）
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])  # 点C到矢量AB的垂直距离 = |矢量AC * 矢量AB|
    mask = limb_width < thre  # 计算这个点在不在躯干里面 躯干默认宽度为1  因为长方形只是大致位置

    vec_map = np.copy(accumulate_vec_map) * 0.0  # 构建一个全0的矩阵 在躯干上的后面会赋值1

    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]  # 在该区域上的都用对应的方向向量表示（根据mask结果表示是否在）

    mask = np.logical_or.reduce(  # 这个时候就不是长方形了，而是A到B的躯干宽度的区域
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))  # 在特征图中（46*46）中 哪些区域是该躯干所在区域

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[:, :, np.newaxis])  # 后面要返回重叠部分的平均向量 所以这里先累加
    accumulate_vec_map += vec_map  # 加上当前关节点位置形成的向量
    count[mask == True] += 1  # 该区域计算次数都+1 方便计算重叠部分次数

    mask = count == 0

    count[mask == True] = 1  # 没有被计算过的地方就等于自身（因为一会要除法，任何数除1还是本身）

    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])  # 算平均向量
    count[mask == True] = 0  # 还原回去

    return accumulate_vec_map, count
