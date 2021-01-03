import os
import cv2
import imutils
import numpy as np
 
import sys

#print('该程序名为:%s' % sys.argv[0])
#print('传入参数为：%s' % str(sys.argv[1:]), end = ' ')

#temper=input("continue")

img_dir = str(sys.argv[1])

#img_dir = 'raw'                                 #img_dir文件路径

names = os.listdir(img_dir)
 
images = []
for name in names:
    img_path = os.path.join(img_dir, name)
    image = cv2.imread(img_path)
    images.append(image)
 
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
status, stitched = stitcher.stitch(images)
 
# 四周填充黑色像素，再得到阈值图
stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
 
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key=cv2.contourArea)
 
mask = np.zeros(thresh.shape, dtype="uint8")
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
 
minRect = mask.copy()
sub = mask.copy()
 
# 开始while循环，直到sub中不再有前景像素
while cv2.countNonZero(sub) > 0:
    minRect = cv2.erode(minRect, None)
    sub = cv2.subtract(minRect, thresh)
 
cnts, hierarchy = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(cnts, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(cnt)
 
# 使用边界框坐标提取最终的全景图
stitched = stitched[y:y + h, x:x + w]

#print(img_dir + '/final.jpg')
cv2.imwrite(img_dir + '/final.jpg', stitched)
