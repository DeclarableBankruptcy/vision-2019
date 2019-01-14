import numpy as np
import cv2

lower_green = np.array([0, 130, 130])
upper_green = np.array([60, 255, 255])
sensorLength = 10
focalLength = 20
fieldOfView = 60
screenWidth = 720
acrossRat = 0
degreesAcross = 0
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours = cv2.findContours(mask, 1, 2) [-2]
    sorted(contours, key=cv2.contourArea)

    #cv2.erode(mask, None, dst=mask, iterations=4)
    #cv2.dilate(mask, None, dst=mask, iterations=4)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)

    rect1 = cv2.minAreaRect(cnt)
    box1 = cv2.boxPoints(rect1)
    box1 = np.int0(box1)
    cv2.drawContours(img, [box1], -1, (0, 0, 255), 1)

    box1Average = np.average(box1, axis=0)
    centreX = box1Average

    acrossRat = screenWidth / centreX

    degreesAcross = fieldOfView / acrossRat

    print(degreesAcross)
    cv2.imshow("IMG", img)
    cv2.imshow("MASK", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # time.sleep(2)
cv2.destroyAllWindows()
cap.release()

    
