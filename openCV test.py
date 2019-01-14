import numpy as np
import cv2
import os
import time

lower_green = np.array([0, 130, 130])
upper_green = np.array([60, 255, 255])
cap = cv2.VideoCapture(0)
totalCoordsX = 0
totalCoordsY = 0
files = os.listdir('C:/Users/User1/Desktop/Vision Images/')
index = 0
sensorLength = 10
focalLength = 20
fieldOfView = 60
screenWidth = 720
acrossRat = 0
degreesAcross = 0
while True:
    file = files[index]
    img = cv2.imread('C:/Users/User1/Desktop/Vision Images/' + file)
    # ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    cv2.erode(mask, None, dst=mask, iterations=2)
    cv2.dilate(mask, None, dst=mask, iterations=2)

    contours = cv2.findContours(mask, 1, 2) [-2]
    sorted(contours, key=cv2.contourArea)
    
    if len(contours) > 2:
        cnt1 = contours[len(contours) - 1]
        cnt2 = contours[len(contours) - 2]
        
    rect1 = cv2.minAreaRect(cnt1)
    box1 = cv2.boxPoints(rect1)
    box1 = np.int0(box1)
    cv2.drawContours(img, [box1], -1, (0, 0, 255), 1)

    rect2 = cv2.minAreaRect(cnt2)
    box2 = cv2.boxPoints(rect2)
    box2 = np.int0(box2)
    cv2.drawContours(img, [box2], -1, (0, 0, 255), 1)    

    box1Average = np.average(box1, axis=0)
    box2Average = np.average(box2, axis=0)

    centerX = int((box1Average[0] + box2Average[0]) / 2)
    centerY = int((box1Average[1] + box2Average[1]) / 2)
    #centerX = box1Average
    
    cv2.circle(img, (centerX, centerY), 2, (0, 0, 255), -1)

    acrossRat = screenWidth / centerX

    degreesAcross = fieldOfView / acrossRat - 30

    print(degreesAcross)
    cv2.imshow("IMG", img)
    cv2.imshow("MASK", mask)
    if cv2.waitKey(1) & 0xFF == ord('g'):
        cv2.destroyAllWindows()
        index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(2)
cv2.destroyAllWindows()
cap.release()
