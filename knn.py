# -*- coding: UTF-8 -*-
import numpy as np
import cv2
import glob as gb

# 获取文件夹下原始数字图片
img_path = gb.glob("numbers\\*")
# img_path = cv2.imread()


k = 0
labels = []
samples = []

# 1.遍历文件夹下原始数字图片
# 2.对每张图片进行轮廓提取操作，只提取外围轮廓
# 3.求轮廓外包矩形，并根据矩形大小信息筛选出所有的数字轮廓
# 4.然后根据位置信息对数字框排序，显然第一排依次是12345，第二排依次是67890；
# 5.提取每一个数字所在的矩形框，作为ROI取出
# 对每一个轮廓等级有：[Next,Previous,First_Child,Parent])

# for path in range(10):
for path in img_path:
    # print(path)
    # img = cv2.imread('numbers/'+str(path+1)+'.jpg')
    img = cv2.imread(path)
    # img's shape: (pixel,pixel, 3 channels--> B,G,R)
    # print("image's shape:{}"+str(img.shape))
    # 转化为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # #高斯滤波
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # adaptive Threshold自适应阈值
    # 将灰度图像转换为二值图像
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # 提取轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("hierarchy's shape:{}".format(hierarchy.shape))
    # print("contours's shape:{}".format(contours.shape))
    # print(, sep, end, file, flush)

    # 求轮廓外包矩形，并根据矩形大小信息筛选出所有的数字轮廓
    height, width = img.shape[:2]
    w = width / 5
    recl_list = []
    list1 = []
    list2 = []
    # j = 0
    # labels =[]
    for cnt in contours:
        # 计算并返回指定点集的最小边界矩形
        [x, y, w, h] = cv2.boundingRect(cnt)

        if w > 30 and h > (height / 4):
            if y < (height / 2):
                list1.append([x, y, w, h])
            else:
                list2.append([x, y, w, h])

    # key = lambda 元素: 元素[字段索引] x:x[]可以随意修改，排序方式按照括号[]里面的维度进行
    # [0] 第一维排序、 [1]第一维排序...以此类推
    list1_sorted = sorted(list1, key=lambda t: t[0])
    list2_sorted = sorted(list2, key=lambda t: t[0])

    # 提取出每一个数字所在的矩形框，作为ROI取出
    for i in range(5):
        [x1, y1, w1, h1] = list1_sorted[i]
        [x2, y2, w2, h2] = list2_sorted[i]
        number_roi1 = gray[y1:y1 + h1, x1:x1 + w1]  # Cut the frame to size
        number_roi2 = gray[y2:y2 + h2, x2:x2 + w2]  # ...

        # 数据预处理
        # 1. 把每一张ROI转换为40x20 ##########注意Opencv 里是宽x长
        resized_roi1 = cv2.resize(number_roi1, (20, 40))
        resized_roi2 = cv2.resize(number_roi2, (20, 40))

        # 2.阈值分割 灰度图-> 0-1二值图
        thresh1 = cv2.adaptiveThreshold(resized_roi1, 255, 1, 1, 11, 2)
        thresh2 = cv2.adaptiveThreshold(resized_roi2, 255, 1, 1, 11, 2)

        # 3.创建文件夹
        # path_1 = "F:\\opencv_learning\\opencv_knn_soduko\\New_datasets\\"+str(i+1)+"\\"+str(k)+".jpg"
        # j = 0
        number_path1 = "datasets_hzc\\%s\\%d" % (str(i + 1), k) + '.jpg'

        j = i + 6
        if j == 10:
            j = 0
        # path_2 = "F:\\opencv_learning\\opencv_knn_soduko\\New_datasets\\"+str(j)+"\\"+str(k)+".jpg"
        number_path2 = "datasets_hzc\\%s\\%d" % (str(j), k) + '.jpg'
        k += 1

        # 4.Nomalized
        # normalized_roi1
        normalized_roil = thresh1 / 255
        normalized_roi2 = thresh2 / 255
        # 5.write into the files
        # P.S Don't forget to annotate them, because u dont have to operate it twice
        # cv2.imwrite(number_path1,thresh1)
        # cv2.imwrite(number_path2,thresh2)
        # 把处理完的二值图像展开成一行，以便knn处理
        sample_1 = normalized_roil.reshape((1, 800))
        samples.append(sample_1[0])
        labels.append(float(i + 1))

        # 保存sample供训练用
        sample_2 = normalized_roi2.reshape((1, 800))
        samples.append(sample_2[0])
        # 把数字标签按照数字的保存顺序对应保存成训练用的数据
        labels.append(float(j))

        cv2.imwrite(number_path1, thresh1)
        cv2.imwrite(number_path2, thresh2)
        cv2.imshow("number", normalized_roil)
        cv2.waitKey(5)
        # cv2.imshow("1", thresh1)
        # cv2.imshow("2", thresh2)
        # cv2.waitKey(300)
    # print(list1_sorted)
    # print(list2_sorted)
    cv2.imshow("train_pic", img)
    # cv2.waitKey(300)

# 保运数据供KNN用 k邻近算法
print(np.array(labels).shape)

print("\n" * 3)
samples = np.array(samples, np.float32)
# (100,800) 1x100, 800
print("train_data's dimension:{}".format(samples.shape))
labels = np.array(labels, np.float32)
labels = labels.reshape((labels.size, 1))
print("labels's dimension:{}".format(labels.shape))
np.save('samples.npy', samples)
np.save('label.npy', labels)
