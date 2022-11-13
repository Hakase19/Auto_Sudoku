# import cv2
# import pytesseract
#
#
# # 加载图片
# image = cv2.imread("canvas.png")
#
# # 截取矩形区域
# # 格式[y1: y2, x1: x2] , (x1,y1)矩形左上角，(x2,y2)矩形右下角.
# # image = image[13:229, 233:445]
#
# # 灰度转换
# GrayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # 二值化
# # cv2Type： int类型
# # 0. cv2.THRESH_BINARY
# # 表示阈值的二值化操作，大于阈值使用maxval表示，小于阈值使用0表示
# #
# # 1. cv2.THRESH_BINARY_INV
# # 表示阈值的二值化翻转操作，大于阈值的使用0表示，小于阈值的使用最大值表示
# # ret, thresh2 = cv2.threshold(GrayImage, 88, 255, cv2.THRESH_BINARY_INV)
#
# # cv2Threshold 阈值
#
# GaussianBlurImage = cv2.GaussianBlur(GrayImage, (5, 5), 0)
# ret, thresh2 = cv2.threshold(GaussianBlurImage, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("test", thresh2)
# cv2.waitKey(0)
# # cv2.imshow("1", thresh2)
# # cv2.waitKey(0)
# # ocr 辨识
# # output_type=Output.DICT 将获取具体辨识数据，用于后期处理。
# # results = pytesseract.image_to_data(thresh2, output_type=Output.DICT, lang='eng')
#
# # print(pytesseract.image_to_data(thresh2))
# # print(pytesseract.image_to_string(thresh2, config='outputbase digits'))
#
# # string = pytesseract.image_to_string(thresh2, lang='eng', config='--psm 6 --oem 3 -c '
# #                                                                  'tessedit_char_whitelist'
# #                                                                  '=0123456789')
#
# # string = pytesseract.image_to_string(thresh2, lang="eng", config="--psm 7")
# # string = string[0]
# # if string == '\x0c':
# #     string = '0'
# # list = []
# # list.append(int(string))
# # print(list)
# delta = int(2000 / 9)
# for i in range(0, 9):
#     for j in range(0, 9):
#         tempimg = thresh2[0 + i * delta+10: delta + i * delta-10, 0 + j * delta+10: delta + j * delta-10]
#         # string = pytesseract.image_to_string(tempimg, lang="eng", config="--psm 7")
#
#         string = pytesseract.image_to_string(tempimg, lang='chi_sim', config='--psm 6 --oem 3 -c '
#                                                                          'tessedit_char_whitelist'
#                                                                          '=0123456789')
#         # string = string[0]
#         # if string != '1' and string != '2' and string != '3' and string != '4' and string != '5' and string != '6' and string != '7' and string != '8' and string != '9':
#         #     string = '0'
#         print(string)
#         cv2.imshow('test', tempimg)
#         cv2.waitKey(0)
#
#


import cv2
import numpy as np
import operator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json


def number_detect(grid):
    classifier = load_model("./digit_model.h5")

    marge = 4
    case = 28 + 2 * marge
    taille_grille = 9 * case

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1080, 620))


    frame = cv2.imread("canvas.png")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > 25000:
            peri = cv2.arcLength(c, True)
            polygone = cv2.approxPolyDP(c, 0.01 * peri, True)
            if area > maxArea and len(polygone) == 4:
                contour_grille = polygone
                maxArea = area

    cv2.drawContours(frame, [contour_grille], 0, (0, 255, 0), 2)
    points = np.vstack(contour_grille).squeeze()
    points = sorted(points, key=operator.itemgetter(1))
    if points[0][0] < points[1][0]:
        if points[3][0] < points[2][0]:
            pts1 = np.float32([points[0], points[1], points[3], points[2]])
        else:
            pts1 = np.float32([points[0], points[1], points[2], points[3]])
    else:
        if points[3][0] < points[2][0]:
            pts1 = np.float32([points[1], points[0], points[3], points[2]])
        else:
            pts1 = np.float32([points[1], points[0], points[2], points[3]])
    pts2 = np.float32([[0, 0], [taille_grille, 0], [0, taille_grille], [
                      taille_grille, taille_grille]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    grille = cv2.warpPerspective(frame, M, (taille_grille, taille_grille))
    grille = cv2.cvtColor(grille, cv2.COLOR_BGR2GRAY)
    grille = cv2.adaptiveThreshold(
        grille, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)


    grille_txt = []
    for y in range(9):
        ligne = ""
        for x in range(9):
            y2min = y * case + marge
            y2max = (y + 1) * case - marge
            x2min = x * case + marge
            x2max = (x + 1) * case - marge
            img = grille[y2min:y2max, x2min:x2max]
            x = img.reshape(1, 28, 28, 1)
            if x.sum() > 10000:
                prediction = classifier.predict_classes(x)
                ligne += "{:d}".format(prediction[0])
            else:
                ligne += "{:d}".format(0)
        grille_txt.append(ligne)

    # grid = []
    print(grille_txt)
    for i in range(0,9):
        str = grille_txt[i]
        print(str)
        templist = []
        for j in range(0,9):
            templist.append(int(str[j]))
        grid.append(templist)

    return grid;

# print(grille_txt)
# print(grid)

