import cv2
from matplotlib import pyplot as plt


# 转换图像格式，并使用matplotlib显示opencv图像，单幅图像
def plotImg(img, title=""):
    if img.ndim == 3:
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        plt.imshow(img), plt.axis("off")
    else:
        plt.imshow(img, cmap='gray'), plt.axis("off")
    plt.title(title)
    plt.show()


# 转换图像格式，并使用matplotlib显示opencv图像
def plotImgs(img1, img2):
    if img1.ndim == 3:
        b, g, r = cv2.split(img1)
        img1 = cv2.merge([r, g, b])
        plt.subplot(121), plt.imshow(img1), plt.axis("off")
    else:
        plt.subplot(121), plt.imshow(img1, cmap='gray'), plt.axis("off")
    if img2.ndim == 3:
        b, g, r = cv2.split(img2)
        img2 = cv2.merge([r, g, b])
        plt.subplot(122), plt.imshow(img2), plt.axis("off")
    else:
        plt.subplot(122), plt.imshow(img2, cmap='gray'), plt.axis("off")
    plt.show()
