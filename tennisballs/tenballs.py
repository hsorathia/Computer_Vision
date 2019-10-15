import cv2 as cv
import numpy as np

# while(1):
# fram = cv.imread('')
image = cv.imread('data\\batchtwo2.JPEG')

# size = cv.cvGetSize(image)
low_green = np.array([28, 47, 47])
high_green = np.array([47, 160, 255])

blurred = cv.GaussianBlur(image, (11, 11), 0)

hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)  # convert to HSV
# makes it so we only see the green color that we changed
mask = cv.inRange(hsv, low_green, high_green)
# smoothens out the picture
mask = cv.erode(mask, None, iterations=2)
# removes most of the little blobs in the picture
mask = cv.dilate(mask, None, iterations=2)

res = cv.bitwise_and(image, image, mask=mask)

circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, 20, param1=20, param2=10,
                          minRadius=0, maxRadius=100)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv.circle(image, (i[0], i[1]), i[2], (0, 255, 255), 2)
    print("x: ", i[0], " y: ", i[1], " d: ", i[2])
    cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
cv.imshow('image', image)
cv.imshow('mask', mask)
cv.imshow('bruhh', res)
# cv.imshow('mask2', mask2)
# cv.imshow('res', res)
# cv.imshow('blurred', blurred)

cv.waitKey(0)


# hsv = cv.cvCreateImage(size, IPL_DEPTH_8U, 3)
# cv.cvCvtColor(image, hsv, CV_BGR2HSV)
# mask = cv.cvCreate
