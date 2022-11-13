import os
from sudokuDetector import number_detect
from sudokuSolver import solver
from sudokuPrinter import Print

# 创建用于存储数独的矩阵grid
grid = []

# 抓取数独题目的图片
os.system("python sudokuCrawler.py")

# 由图片识别数字并储存到grid中
grid = number_detect(grid)

# 自动解数独
grid = solver(grid)

# 在网页上完成数独题目
Print(grid)
