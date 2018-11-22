# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 标准方格大小
GRID_WIDTH = 40
GRID_HEIGHT = 40
# 标准数字大小
NUM_WIDTH = 20
NUM_HEIGHT = 20
# 判定非零像素书最小阈值
N_MIN_ACTIVE_PIXELS = 30


def extract_grid(im_number):
    """
    将校正后图像划分为9x9的棋盘，取出小方格；二值化；去除离中心较远的像素（排除边框干扰）；统计非零像素数（判断方格中是否有数字）
    :param im_number: 方格图像
    :param x: 方格横坐标 (0~9)
    :param y: 方格纵坐标 (0~9)
    :return: im_number_thresh: 二值化及处理后图像
             n_active_pixels: 非零像素数
    """

    # 二值化
    retVal, im_number_thresh = cv2.threshold(im_number, 150, 255, cv2.THRESH_BINARY)

    # 去除离中心较远的像素点（排除边框干扰）
    for i in range(im_number.shape[0]):
        for j in range(im_number.shape[1]):
            dist_center = np.sqrt(np.square(GRID_WIDTH // 2 - i) + np.square(GRID_HEIGHT // 2 - j))
            if dist_center > GRID_WIDTH // 2 - 2:
                im_number_thresh[i, j] = 0

    # 统计非零像素数，以判断方格中是否有数字
    n_active_pixels = cv2.countNonZero(im_number_thresh)

    return [im_number_thresh, n_active_pixels]


def find_biggest_bounding_box(im_number_thresh):
    """
    找出小方格中外接矩形面积最大的轮廓，返回其外接矩形参数
    :param im_number_thresh: 当前方格的二值化及处理后图像
    :return: 外接矩形参数（左上坐标及长和宽）
    """
    # 轮廓检测
    b, contour, hierarchy1 = cv2.findContours(im_number_thresh.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
    # 找出外接矩形面积最大的轮廓
    biggest_bound_rect = []
    bound_rect_max_size = 0
    for i in range(len(contour)):
        bound_rect = cv2.boundingRect(contour[i])
        size_bound_rect = bound_rect[2] * bound_rect[3]
        if size_bound_rect > bound_rect_max_size:
            bound_rect_max_size = size_bound_rect
            biggest_bound_rect = bound_rect

    # 将外接矩形扩大一个像素
    x_b, y_b, w, h = biggest_bound_rect
    x_b = x_b - 1
    y_b = y_b - 1
    w = w + 2
    h = h + 2
    return [x_b, y_b, w, h]


def recognize_number(im_number, x, y):
    """
    判断当前方格是否存在数字并存储该数字
    :param im_number: 方格图像
    :param x: 方格横坐标 (0~9)
    :param y: 方格纵坐标 (0~9)
    :return: 数字的一维数组
    """
    # 提取并处理方格图像
    [im_number_thresh, n_active_pixels] = extract_grid(im_number)

    # 条件1：非零像素大于设定的最小值
    if n_active_pixels > N_MIN_ACTIVE_PIXELS:

        # 找出外接矩形
        [x_b, y_b, w, h] = find_biggest_bounding_box(im_number_thresh)

        # 计算矩形中心与方格中心距离
        cX = x_b + w // 2
        cY = y_b + h // 2
        d = np.sqrt(np.square(cX - GRID_WIDTH // 2) + np.square(cY - GRID_HEIGHT // 2))

        # 条件2: 外接矩形中心与方格中心距离足够小
        if d < GRID_WIDTH // 4:

            # 取出方格中数字
            number_roi = im_number[y_b:y_b + h, x_b:x_b + w]

            # 扩充数字图像为正方形，边长取长宽较大者
            h1, w1 = np.shape(number_roi)
            if h1 > w1:
                number = np.zeros(shape=(h1, h1))
                number[:, (h1 - w1) // 2:(h1 - w1) // 2 + w1] = number_roi
            else:
                number = np.zeros(shape=(w1, w1))
                number[(w1 - h1) // 2:(w1 - h1) // 2 + h1, :] = number_roi

            # 将数字缩放为标准大小
            number = cv2.resize(number, (NUM_WIDTH, NUM_HEIGHT), interpolation=cv2.INTER_LINEAR)

            retVal, number = cv2.threshold(number, 50, 255, cv2.THRESH_BINARY)

            # 转换为1维数组并返回
            return True, number.reshape(1, NUM_WIDTH * NUM_HEIGHT)

    # 没有数字，则返回全零1维数组
    return False, np.zeros(shape=(1, NUM_WIDTH * NUM_HEIGHT))

