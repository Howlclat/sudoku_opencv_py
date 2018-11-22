# -*- coding: utf-8 -*-
import cv2
import numpy as np


def knn_ocr_normal(test):
    # 训练knn模型
    samples = np.load('samples.npy')
    labels = np.load('label.npy')

    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, labels)

    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    return result


def knn_ocr_handwritten(test):
    # read data set
    img_digits = cv2.imread('./images/digits.png')
    img_digits_gray = cv2.cvtColor(img_digits, cv2.COLOR_BGR2GRAY)

    # Now we split the image to 5000 cells, each 20x20 size
    cells = [np.hsplit(row, 100) for row in np.vsplit(img_digits_gray, 50)]

    # Make it into a Numpy array. It size will be (50,100,20,20)
    x = np.array(cells)
    # Now we prepare train_data and test_data.
    train = x[5:, :100].reshape(-1, 400).astype(np.float32)  # Size = (5000,400)

    # Create labels for train and test data
    k = np.arange(1, 10)
    train_labels = np.repeat(k, 500)[:, np.newaxis]

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    return result


def knn_ocr_handwritten_mnist(test):

    samples = np.load('samples_mnist.npy')
    labels = np.load('label_mnist.npy')

    knn = cv2.ml.KNearest_create()
    knn.train(samples, cv2.ml.ROW_SAMPLE, labels)

    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    return result
