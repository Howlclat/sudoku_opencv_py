# -*- coding: utf-8 -*-
import cv2
import numpy as np
import plotCVImg

IMAGE_WIDTH = 40
IMAGE_HEIGHT = 40
SUDOKU_SIZE = 9
N_MIN_ACTIVE_PIXELS = 30
SIZE_PUZZLE = IMAGE_WIDTH * SUDOKU_SIZE
DEBUG = 0


def correct(img_original):
    # 灰度化
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        plotCVImg.plotImg(img_gray, "gray")

    # 中值滤波
    img_blur = cv2.medianBlur(img_gray, 1)
    if DEBUG:
        plotCVImg.plotImg(img_blur, "median Blur")

    # 高斯滤波
    img_blur = cv2.GaussianBlur(img_blur, (3, 3), 0)
    if DEBUG:
        plotCVImg.plotImg(img_blur, "Gaussian Blur")

    # 将每个像素除以闭操作后的像素，可以调整图像亮度
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    close = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, kernel)
    div = np.float32(img_blur) / close
    img_brightness_adjust = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    if DEBUG:
        plotCVImg.plotImg(img_brightness_adjust, "brightness adjust")

    # 自适应阈值二值化，注意其返回值只有一个
    img_thresh = cv2.adaptiveThreshold(img_brightness_adjust, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 7)
    if DEBUG:
        img_thresh = cv2.medianBlur(img_thresh, 3)
        plotCVImg.plotImg(img_thresh, "adaptive Threshold")

    # 寻找轮廓
    binary, contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        img_contours = img_original.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)
        plotCVImg.plotImg(img_contours, "contours")

    # 找到最大轮廓
    max_area = 0
    biggest_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            biggest_contour = cnt

    # mask操作
    mask = np.zeros(img_brightness_adjust.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], 0, 255, cv2.FILLED)
    cv2.drawContours(mask, [biggest_contour], 0, 0, 2)
    image_with_mask = cv2.bitwise_and(img_brightness_adjust, mask)
    if DEBUG:
        plotCVImg.plotImg(image_with_mask, "image_with_mask")

    # 角点检测
    dst = cv2.cornerHarris(image_with_mask, 2, 3, 0.04)
    if DEBUG:
        plotCVImg.plotImg(dst, "image_cornerHarris")

    # x方向Sobel算子，膨胀操作连接断线，边缘检测找出竖线
    dx = cv2.Sobel(image_with_mask, cv2.CV_16S, 1, 0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
    ret, close = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernelx, iterations=1)

    binary, contour, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if h / w > 5:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)

    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, None, iterations=2)
    closex = close.copy()

    # Y方向，找出横线
    dy = cv2.Sobel(image_with_mask, cv2.CV_16S, 0, 2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy, dy, 0, 255, cv2.NORM_MINMAX)
    retVal, close = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernely)

    binary, contour, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if w / h > 5:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)

    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, None, iterations=2)
    closey = close.copy()

    # 求x,y交点
    res = cv2.bitwise_and(closex, closey)
    if DEBUG:
        plotCVImg.plotImg(res, "dots")

    # 查找轮廓，求每个轮廓的质心centroids
    img_dots = cv2.cvtColor(img_brightness_adjust, cv2.COLOR_GRAY2BGR)
    binary, contour, hierarchy = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contour:
        if cv2.contourArea(cnt) > 20:
            mom = cv2.moments(cnt)
            (x, y) = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
            cv2.circle(img_dots, (x, y), 4, (0, 255, 0), -1)
            centroids.append((x, y))
    centroids = np.array(centroids, dtype=np.float32)
    c = centroids.reshape((100, 2))
    c2 = c[np.argsort(c[:, 1])]

    b = np.vstack([c2[i * 10:(i + 1) * 10][np.argsort(c2[i * 10:(i + 1) * 10, 0])] for i in range(10)])
    bm = b.reshape((10, 10, 2))

    res2 = cv2.cvtColor(img_brightness_adjust, cv2.COLOR_GRAY2BGR)
    output = np.zeros((450, 450, 3), np.uint8)
    for i, j in enumerate(b):
        ri = i // 10
        ci = i % 10
        if ci != 9 and ri != 9:
            src = bm[ri:ri + 2, ci:ci + 2, :].reshape((4, 2))
            dst = np.array([[ci * 50, ri * 50], [(ci + 1) * 50 - 1, ri * 50], [ci * 50, (ri + 1) * 50 - 1],
                            [(ci + 1) * 50 - 1, (ri + 1) * 50 - 1]], np.float32)
            retval = cv2.getPerspectiveTransform(src, dst)
            warp = cv2.warpPerspective(res2, retval, (450, 450))
            output[ri * 50:(ri + 1) * 50 - 1, ci * 50:(ci + 1) * 50 - 1] = warp[ri * 50:(ri + 1) * 50 - 1,
                                                                           ci * 50:(ci + 1) * 50 - 1].copy()
    img_correct = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    img_puzzle = cv2.adaptiveThreshold(img_correct, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 7)
    img_puzzle = cv2.resize(img_puzzle, (SIZE_PUZZLE, SIZE_PUZZLE), interpolation=cv2.INTER_LINEAR)
    return img_puzzle


def correct2(img_original):

    if DEBUG:
        plotCVImg.plotImg(img_original, "original")

    # gray image
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    if DEBUG:
        plotCVImg.plotImg(img_gray, "gray")

    # median Blur
    img_Blur = cv2.medianBlur(img_gray, 5)
    if DEBUG:
        plotCVImg.plotImg(img_Blur, "median Blur")

    # Gaussian Blur
    img_Blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    if DEBUG:
        plotCVImg.plotImg(img_Blur, "GaussianBlur")

    # adaptive threshold
    img_thresh = cv2.adaptiveThreshold(img_Blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if DEBUG:
        plotCVImg.plotImg(img_thresh, "adaptiveThreshold")

    # find the contours RETR_EXTERNAL
    binary, contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        img_contours = img_original.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 0, 255), 2)
        plotCVImg.plotImg(img_contours, "contours")

    # find the biggest contours
    size_rectangle_max = 0
    index_biggest = 0
    for i in range(len(contours)):
        size_rectangle = cv2.contourArea(contours[i])
        # store the index of the biggest
        if size_rectangle > size_rectangle_max:
            size_rectangle_max = size_rectangle
            index_biggest = i

    # 多边形拟合
    epsilon = 0.1 * cv2.arcLength(contours[index_biggest], True)
    biggest_rectangle = cv2.approxPolyDP(contours[index_biggest], epsilon, True)

    if DEBUG:
        # copy the original image to show the border
        img_border = img_original.copy()
        # 画出数独方格的边界
        for x in range(len(biggest_rectangle)):
            cv2.line(img_border,
                     (biggest_rectangle[(x % 4)][0][0], biggest_rectangle[(x % 4)][0][1]),
                     (biggest_rectangle[((x + 1) % 4)][0][0], biggest_rectangle[((x + 1) % 4)][0][1]),
                     (255, 0, 0), 2)
        plotCVImg.plotImg(img_border, "border")

    # sort the corners to remap the image
    def sortCornerPoints(rcCorners):
        point = rcCorners.reshape((4, 2))
        mean = rcCorners.sum() / 8
        cornerPoint = np.zeros((4, 2), dtype=np.float32)
        for i in range(len(point)):
            if point[i][0] < mean:
                if point[i][1] < mean:
                    cornerPoint[0] = point[i]
                else:
                    cornerPoint[2] = point[i]
            else:
                if point[i][1] < mean:
                    cornerPoint[1] = point[i]
                else:
                    cornerPoint[3] = point[i]
        return cornerPoint

    # 透视变换
    cornerPoints = sortCornerPoints(biggest_rectangle)
    puzzlePoints = np.float32([[0, 0], [SIZE_PUZZLE, 0], [0, SIZE_PUZZLE], [SIZE_PUZZLE, SIZE_PUZZLE]])
    PerspectiveMatrix = cv2.getPerspectiveTransform(cornerPoints, puzzlePoints)
    img_puzzle = cv2.warpPerspective(img_thresh, PerspectiveMatrix, (SIZE_PUZZLE, SIZE_PUZZLE))
    if DEBUG:
        plotCVImg.plotImg(img_puzzle, "puzzle")

    return img_puzzle
