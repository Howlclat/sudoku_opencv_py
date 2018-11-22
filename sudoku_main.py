# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import knn_ocr
import plotCVImg
import sudoku_solver
import correction
import extractNumber

# 是否查看中间过程
DEBUG = 1
# 标准方格大小
GRID_WIDTH = 40
GRID_HEIGHT = 40
# 标准数字大小
NUM_WIDTH = 20
NUM_HEIGHT = 20
# 数独尺寸
SUDOKU_SIZE = 9


# 存储题目的数组 shape=(9*9, 20*20)
sudoku = np.zeros(shape=(9 * 9, NUM_WIDTH * NUM_HEIGHT))

# 读取图片 read image
img_original = cv2.imread('./images/c3.png')
if DEBUG:
    plotCVImg.plotImg(img_original, "original")

# 预处理及图像校正 pre-processing and image correction
img_puzzle = correction.correct(img_original)
if DEBUG:
    plotCVImg.plotImg(img_puzzle, "pre-process")


# 识别并记录序号 detect numbers and extract them
indexes_numbers = []
for i in range(SUDOKU_SIZE):
    for j in range(SUDOKU_SIZE):
        img_number = img_puzzle[i * GRID_HEIGHT:(i + 1) * GRID_HEIGHT][:, j * GRID_WIDTH:(j + 1) * GRID_WIDTH]
        hasNumber, sudoku[i * 9 + j, :] = extractNumber.extract_number(img_number)
        if hasNumber:
            indexes_numbers.append(i * 9 + j)

# 显示提取数字结果
if DEBUG:
    print("There are", len(indexes_numbers), "numbers and the indexes of them are:")
    print(indexes_numbers)
    # 创建子图
    rows = len(indexes_numbers) // 5 + 1
    f, axarr = plt.subplots(rows, 5)
    row = 0
    for x in range(len(indexes_numbers)):
        ind = indexes_numbers[x]

        if (x % 5) == 0 and x != 0:
            row = row + 1
        axarr[row, x % 5].imshow(cv2.resize(sudoku[ind, :].reshape(NUM_WIDTH, NUM_HEIGHT),
                                            (NUM_WIDTH * 5, NUM_HEIGHT * 5)), cmap=plt.gray())
    for i in range(rows):
        for j in range(5):
            axarr[i, j].axis("off")
    plt.show()

# 构建测试数据集 build a test data set for knn
test = np.zeros(shape=(len(indexes_numbers), NUM_WIDTH * NUM_HEIGHT))
for num in range(len(indexes_numbers)):
    test[num] = sudoku[indexes_numbers[num]]
test = test.reshape(-1, NUM_WIDTH * NUM_HEIGHT).astype(np.float32)

result = knn_ocr.knn_ocr_normal(test)

# 使用识别结果构建数独问题的二维数组，其他数字用0表示 build a puzzle 2D-array using the result of OCR
sudoku_puzzle = np.zeros(SUDOKU_SIZE * SUDOKU_SIZE)
for num in range(len(indexes_numbers)):
    sudoku_puzzle[indexes_numbers[num]] = result[num]
sudoku_puzzle = sudoku_puzzle.reshape((SUDOKU_SIZE, SUDOKU_SIZE)).astype(np.int32)
print(sudoku_puzzle)

# 保存提取出的图片（可选） save the numbers image (optional)
for num in range(len(indexes_numbers)):
    number_path1 = "number\\%s" % (str(num)) + '.png'
    img_num = sudoku[indexes_numbers[num]].reshape(20, 20)
    cv2.imwrite(number_path1, img_num)

# 显示识别出的数字 show the numbers
img_puzzle_white = img_puzzle.copy()
img_puzzle_white = cv2.bitwise_not(img_puzzle_white)
img_puzzle_recognize = cv2.cvtColor(img_puzzle_white, cv2.COLOR_GRAY2BGR)
for i in range(9):
    for j in range(9):
        x = int(i * GRID_WIDTH + 10)
        y = int(j * GRID_WIDTH + GRID_WIDTH - 8)
        if sudoku_puzzle[j][i] > 0:
            cv2.putText(img_puzzle_recognize, str(sudoku_puzzle[j][i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
plotCVImg.plotImgs(img_puzzle_white, img_puzzle_recognize)

# 解数独 solve the puzzle
sudoku_solved = sudoku_solver.solveSudoku(sudoku_puzzle)
sudoku_solved = np.array(sudoku_solved)
print(sudoku_solved)

# 显示结果 show the answer
img_puzzle_solved = img_puzzle_recognize.copy()
for i in range(9):
    for j in range(9):
        x = int(i * GRID_WIDTH + 10)
        y = int(j * GRID_WIDTH + GRID_WIDTH - 8)
        if sudoku_puzzle[j][i] == 0:
            cv2.putText(img_puzzle_solved, str(sudoku_solved[j][i]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                        2)
plotCVImg.plotImg(img_puzzle_solved)

