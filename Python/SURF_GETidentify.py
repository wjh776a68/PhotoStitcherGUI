import cv2 
import numpy as np 

filepath = input("please input the file path ")
img = cv2.imread(filepath)
 
#参数为hessian矩阵的阈值
surf = cv2.xfeatures2d.SURF_create(4000)
#找到关键点和描述符
key_query,desc_query = surf.detectAndCompute(img,None)
#把特征点标记到图片上
img=cv2.drawKeypoints(img,key_query,img)

cv2.imshow('sp',img)

cv2.imwrite(filepath + ".jpg",img)

cv2.waitKey(0)
