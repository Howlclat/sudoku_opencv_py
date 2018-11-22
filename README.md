# 基于opencv-Python 的数独问题识别与求解

使用 OpenCV 3 和 Python 3，识别图片中的数独问题并求解。

软件版本：
- OpenCV 3.4
- Python 3.6

![origin puzzle](https://github.com/Howlclat/sudoku_opencv_py/raw/master/images/c3.png)

```
There are 26 numbers and the indexes of them are:
[3, 5, 6, 9, 11, 17, 23, 25, 28, 31, 34, 35, 36, 44, 45, 46, 49, 52, 55, 57, 63, 69, 71, 74, 75, 77]

[[0 0 0 6 0 4 7 0 0]
 [7 0 6 0 0 0 0 0 9]
 [0 0 0 0 0 5 0 8 0]
 [0 7 0 0 2 0 0 9 3]
 [8 0 0 0 0 0 0 0 5]
 [4 3 0 0 1 0 0 7 0]
 [0 5 0 2 0 0 0 0 0]
 [3 0 0 0 0 0 2 0 8]
 [0 0 2 3 0 1 0 0 0]]

cost time: 0:00:00.047003
times: 33146
[[5 8 3 6 9 4 7 2 1]
 [7 1 6 8 3 2 5 4 9]
 [2 9 4 1 7 5 3 8 6]
 [6 7 1 5 2 8 4 9 3]
 [8 2 9 7 4 3 1 6 5]
 [4 3 5 9 1 6 8 7 2]
 [1 5 8 2 6 7 9 3 4]
 [3 6 7 4 5 9 2 1 8]
 [9 4 2 3 8 1 6 5 7]]
```


详细说明请访问我的博客：https://blog.csdn.net/howlclat