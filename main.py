import sys
import os
import time
import pyautogui
from detect2 import number_detect
from Solution import solver,valid
from sudokuSolver import solveSudoku, Print
import Solution
# import sudokuSolver
grid = []
os.system("python sele.py")
# os.system("python detect2.py")
print(grid)
grid = number_detect(grid)

print(grid)
# grid = solveSudoku(grid)
grid = solver(grid)
print(type(grid))
print(grid)

# print(solveSudoku(grid))

Print(grid)