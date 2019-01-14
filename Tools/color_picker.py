import numpy
import cv2

img = cv2.imread("C:/Users/User1/Desktop/Vision Images/sample1.jpg")

def printHSV (event, x, y, flags, param):
    print(img[y, x], x, y)

while True:
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', printHSV)

    cv2.imshow('image', img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cv2.destroyAllWindows
