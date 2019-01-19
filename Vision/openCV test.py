import numpy as np
import cv2
import os


lower_green = np.array([60, 177, 177])
upper_green = np.array([160, 255, 255])
lower_white = np.array([0, 70, 70])
upper_white = np.array([60, 255, 255])
index = 0
files = os.listdir("C:/vision-2019/Sample_Images/Ground Tape/High Exposure/")


def trackRetroTape(self, fieldOfView, screenWidth):
    files = os.listdir("C:/Users/User1/Desktop/Vision Images/Retro Tape/Low Exposure/")
    index = 0

    while True:
        file = files[index]
        img = cv2.imread(
            "C:/Users/User1/Desktop/Vision Images/Retro Tape/Low Exposure/" + file
        )
        # ret, img = cap.read()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

        # 'Opens' the image. Erodes and then dilates, wiping away unwanted noise.
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)

        contours = cv2.findContours(mask, 1, 2)[-2]
        contours.sort(key=cv2.contourArea, reverse=True)

        if len(contours) > 1:
            cnt1 = contours[0]
            cnt2 = contours[1]

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

        cv2.circle(img, (centerX, centerY), 2, (0, 0, 255), -1)

        acrossRat = screenWidth / centerX
        degreesAcross = fieldOfView / acrossRat - (fieldOfView / 2)

        print(degreesAcross)
        print(rect1[2], rect2[2])

        cv2.imshow("IMG", img)
        cv2.imshow("MASK", mask)
        if cv2.waitKey(1) & 0xFF == ord("g"):
            cv2.destroyAllWindows()
            index += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


index = 0


def trackGroundTape(img):
    # centerArray = []
    boxPointsOnScreen = []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    screenWidth = np.shape(img)[0]
    screenHeight = np.shape(img)[1]
    halfScreenWidth = screenWidth / 2
    halfScreenHeight = screenHeight / 2
    
    cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None)

    contours = cv2.findContours(mask, 1, 2)[-2]

    if len(contours) > 0:
        # index1 = 0
        # for cnt in contours:
        #     if cv2.contourArea(cnt) < 100:
        #         contours.pop(index1)
        #     index1 += 1
        # for cnt in contours:
        #     rect = cv2.minAreaRect(cnt)
        #     xCoord = rect[0][0]
        #     distanceFromCenter = abs((screenWidth / 2) - xCoord)

        #     centerArray.append((distanceFromCenter, rect))

            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(img, [box], -1, (0, 0, 255), 1)

        # centerArray.sort(key=lambda a: a[0])
        # box = cv2.boxPoints(centerArray[0][1])
        # box = np.int0(box)
        contours.sort(key=cv2.contourArea, reverse=True)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(img, [box], -1, (0, 0, 255), 1)

    cv2.imshow("IMG", img)
    cv2.imshow("MASK", mask)

    for point in box:
        if point[0] > 5 and point[1] > 5 and point[0] < screenWidth - 5 and point[1] < screenHeight - 5:
            boxPointsOnScreen.append(point)
    print(rect[2], len(boxPointsOnScreen))
    if len(boxPointsOnScreen) == 0:
        # Error code 999 means that there were 0 points found on screen.
        return ((999, 999), rect[2])
    if len(boxPointsOnScreen) == 1:
        # Error code 998 means that there was 1 point found on screen.
        return ((998, 998), rect[2])
    else:
        # No error; just return angle and distance of x and y.
        average = [((boxPointsOnScreen[0][0] + boxPointsOnScreen[1][0]) / 2 - halfScreenWidth) / screenWidth, ((boxPointsOnScreen[1][0] + boxPointsOnScreen[1][1]) / 2 - halfScreenHeight) / screenHeight]
        return (average, rect[2]) 
        print(((boxPointsOnScreen[0][0] + boxPointsOnScreen[1][0]) / 2 - halfScreenWidth) / screenWidth, ((boxPointsOnScreen[1][0] + boxPointsOnScreen[1][1]) / 2 - halfScreenHeight) / screenHeight)


cv2.destroyAllWindows()

while True:
    file = files[index]
    img = cv2.imread("C:/vision-2019/Sample_Images/Ground Tape/High Exposure/" + file)
    trackGroundTape(img)
    if cv2.waitKey(1) & 0xFF == ord("g"):
        cv2.destroyAllWindows()
        index += 1