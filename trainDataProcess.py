# -*- coding: UTF-8 -*-
import glob as gb
import cv2
import numpy as np
# from matplotlib import pyplot as plt


# labels = []
# samples = []
#
# for num in range(9):
#     img_path = gb.glob("number\\%s\\*" % str(num+1))
#
#     for path in img_path:
#         img = cv2.imread(path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # 增强对比度
#         rows, cols = gray.shape
#         a = 1.2
#         b = 100
#         for i in range(rows):
#             for j in range(cols):
#                 color = gray[i, j] * a + b
#                 if color > 255:
#                     gray[i, j] = 255
#                 elif color < 0:
#                     gray[i, j] = 0
#
#         retVal, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
#
#         # plt.imshow(thresh, cmap='Greys_r'), plt.axis("off")
#         # plt.show()
#
#         x = np.array(thresh)
#         x = x.reshape(-1, 400).astype(np.float32)
#         samples.append(x)
#         labels.append(float(num + 1))
#
#
# print(len(samples))
# print(len(labels))
# samples = np.array(samples, np.float32)
# samples = samples.reshape((450, 400))
# labels = np.array(labels, np.float32)
# labels = labels.reshape((labels.size, 1))
# print(samples.shape)
# print(labels.shape)
# np.save('samples.npy', samples)
# np.save('label.npy', labels)


labels = []
samples = []

for num in range(9):
    img_path = gb.glob("D:\\mnist_data\\%s.*" % str(num+1))
    print(num)
    for path in img_path:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        number = cv2.resize(gray, (20, 20), interpolation=cv2.INTER_LINEAR)
        retVal, thresh = cv2.threshold(number, 75, 255, cv2.THRESH_BINARY)
        x = np.array(thresh)
        x = x.reshape(-1, 400).astype(np.float32)
        samples.append(x)
        labels.append(float(num + 1))
print(len(samples))
print(len(labels))
samples = np.array(samples, np.float32)
samples = samples.reshape((len(labels), 400))
labels = np.array(labels, np.float32)
labels = labels.reshape((labels.size, 1))
print(samples.shape)
print(labels.shape)
np.save('samples_mnist.npy', samples)
np.save('label_mnist.npy', labels)
