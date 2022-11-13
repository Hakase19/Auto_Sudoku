import cv2
import numpy as np
import operator
from tensorflow.keras.models import load_model


def number_detect(grid):
    # 载入已有模型
    classifier = load_model("./digit_model.h5")

    marge = 4
    case = 28 + 2 * marge
    taille_grill = 9 * case

    # 读取数独题目图片
    frame = cv2.imread("canvas.png")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxarea = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > 25000:
            peri = cv2.arcLength(c, True)
            polygone = cv2.approxPolyDP(c, 0.01 * peri, True)
            if area > maxarea and len(polygone) == 4:
                contour_grille = polygone
                maxarea = area

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
    pts2 = np.float32([[0, 0], [taille_grill, 0], [0, taille_grill], [
                      taille_grill, taille_grill]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    grille = cv2.warpPerspective(frame, M, (taille_grill, taille_grill))
    grille = cv2.cvtColor(grille, cv2.COLOR_BGR2GRAY)
    grille = cv2.adaptiveThreshold(
        grille, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 3)

    grille_txt = []
    for y in range(9):
        line = ""
        for x in range(9):
            y2min = y * case + marge
            y2max = (y + 1) * case - marge
            x2min = x * case + marge
            x2max = (x + 1) * case - marge
            img = grille[y2min:y2max, x2min:x2max]
            x = img.reshape(1, 28, 28, 1)
            if x.sum() > 10000:
                prediction = classifier.predict_classes(x)
                line += "{:d}".format(prediction[0])
            else:
                line += "{:d}".format(0)
        grille_txt.append(line)

    for i in range(0, 9):
        string = grille_txt[i]
        templist = []
        for j in range(0, 9):
            templist.append(int(string[j]))
        grid.append(templist)

    return grid
