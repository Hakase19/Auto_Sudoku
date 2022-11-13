# -*- coding: UTF-8 -*-
import numpy as np
import cv2

# 利用K邻近算法来进行数字识别
# 1.加载上面保存的样本和标签数据；
# 2.分别用80个作为训练数据，20个作为测试数据；
# 3.用opencv自带的knn训练模型；
# 4.用训练好的模型识别测试数据中的数字；
# 5.输出预测值和实际标签值。

# print(labels.T)
# print(np.sort(labels, axis=0, kind='quicksort')
# print("samples's shape:{}".format(samples.shape))

## 数独求解算法，回溯法。来源见下面链接，有细微改动。
## http://stackoverflow.com/questions/1697334/algorithm-for-solving-sudoku


samples = np.load('samples.npy')
labels = np.load('label.npy')

k=80
train_label = labels[:k]
test_label = labels[k:]
train_input = samples[:k]
test_input = samples[k:]
print("train_label's shape:{}".format(train_label.shape))
print("train_label's shape:{}".format(train_input.shape))
print("test_label's shape:{}".format(test_label.shape))
print("test_input's shape:{}".format(test_input.shape))

#create the model
model = cv2.ml.KNearest_create()
model.train(train_input,cv2.ml.ROW_SAMPLE,train_label)

#retval:返回值类型说明
# retval,results,neigh_resp,dists = model.findNearest(test_input,1)
# string = results.ravel()
# print(string)
# print(test_label.reshape(1,len(test_label))[0])
# print(string.shape)
# print("The original label:{}".format(test_label.reshape(1,len(test_label))[0]))
# print("The prediction:{}".format(str(string)))

# # print(test_label.T)
# count = 0
# string = string.reshape(1,len(string))
# print(string.shape)
# string = np.array(string).T

# for index,value in enumerate(test_label):
#     if value != string[index]:
#         count+=1
# print(count)

# accuracy = 1 - (count / len(string))
# print("The test accuracy:{}".format(accuracy))

img = cv2.imread('canvas.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

##阈值分割
ret,thresh = cv2.threshold(gray,200,255,1)

# in ord to do morphological operation: returns a structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
#Brief dilated an image by a specific structuring element
#膨胀
dilated = cv2.dilate(thresh,kernel)

#提取轮廓
contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#提取81个小方格
#hierarchy[Next,Previous,First_Child,Parent])
boxes = []
for i in range(len(hierarchy[0])):
    if hierarchy[0][i][3] == 0:
        boxes.append(hierarchy[0][i])

print("81个小方格: dimension:{}".format(np.array(boxes).shape))
# print(boxes)

height,width = img.shape[:2]

#"/9" 9行9列
box_h = height/9
box_w = height/9
number_boxes = []

#数独转化为零阵
soduko = np.zeros((9,9),np.int32)

for j in range(len(boxes)):
    if boxes[j][2] != -1:
        #Calculates and returns the minimal up-right bounding
        #rectangle for the specified point set
        x,y,w,h = cv2.boundingRect(contours[boxes[j][2]])
        number_boxes.append([x,y,w,h])
        #process the data that was extracted
        number_roi = gray[y:y+h,x:x+w]
        #unit the size
        resized_roi = cv2.resize(number_roi,(20,40))
        thresh1 = cv2.adaptiveThreshold(resized_roi,255,1,1,11,2)
        #归一化像素值
        normalized_roi = thresh1/255

        #展开一行让knn识别
        sample1 = normalized_roi.reshape((1,800))
        sample1 = np.array(sample1,np.float32)

        retval,result,neigh_resp,dists = model.findNearest(sample1,1)
        number = int(result.ravel()[0])
        # print(results.ravel())

        #识别结果
        # cv2.putText(img,str(number),(x+w+1,y+h-20), 3, 2., (255, 0, 0), 2, cv2.LINE_AA)

        #矩阵中位置
        soduko[int(y/box_h)][int(x/box_w)] = number

        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", img)
        # cv2.imshow("normalized_roi",normalized_roi)
        # cv2.waitKey(120)

print("\n生成的数独\n")
print(soduko)