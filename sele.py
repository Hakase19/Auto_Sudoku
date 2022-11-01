from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import selenium

from string import ascii_uppercase, digits

import time
import cv2
import pytesseract
import base64

from PIL import Image


def traverse():
    for i in range(9):
        for j in range(9):
            yield (i, j)


difficulty = "facile"

driver = webdriver.Edge('D:\Program Files\python-3.8.10\msedgedriver.exe')

driver.set_page_load_timeout(10)
try:
    driver.get(f"https://sudoku.com")
except:
    pass  # hehe

# get board
canvas = driver.find_element(By.XPATH, '/html/body/div[2]/div/div[1]/div[2]/div[3]/div[5]/canvas')

# https://stackoverflow.com/questions/38316402/how-to-save-a-canvas-as-png-in-selenium
canvas_base64 = driver.execute_script("return arguments[0].toDataURL('image/png').substring(21);", canvas)
canvas_png = base64.b64decode(canvas_base64)
with open(r"canvas.png", 'wb') as f:
    f.write(canvas_png)
