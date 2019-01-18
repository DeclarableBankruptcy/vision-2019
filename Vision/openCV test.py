import numpy as np
import cv2
import os


lower_green = np.array([60, 177, 177])
upper_green = np.array([160, 255, 255])
lower_white = np.array([0, 0, 190])
upper_white = np.array([360, 60, 255])
index = 0
files = os.listdir("C:/vision-2019/Sample_Images/Ground Tape/High Exposure/")


def trackRetroTape(self, fieldOfView, screenWidth):
    leftTape = np.array()
    rightTape = np.array()
    pairs = np.array([[]])
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
        sorted(contours, key=cv2.contourArea)

        if len(contours) > 1:
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                if rect[2] < -13 and rect[2] > -16:
                    leftTape.append(rect)
                elif rect[2] < -70 and rect[2] > -77:
                    rightTape.append(rect)

        # rect1 = cv2.minAreaRect(cnt1)
        # box1 = cv2.boxPoints(rect1)
        # box1 = np.int0(box1)
        # cv2.drawContours(img, [box1], -1, (0, 0, 255), 1)

        # rect2 = cv2.minAreaRect(cnt2)
        # box2 = cv2.boxPoints(rect2)
        # box2 = np.int0(box2)
        # cv2.drawContours(img, [box2], -1, (0, 0, 255), 1)

        # box1Average = np.average(box1, axis=0)
        # box2Average = np.average(box2, axis=0)

        # centerX = int((box1Average[0] + box2Average[0]) / 2)
        # centerY = int((box1Average[1] + box2Average[1]) / 2)

        # cv2.circle(img, (centerX, centerY), 2, (0, 0, 255), -1)

        # acrossRat = screenWidth / centerX
        # degreesAcross = fieldOfView / acrossRat - (fieldOfView / 2)

        # print(degreesAcross)
        # print(rect1[2], rect2[2])

        cv2.imshow("IMG", img)
        cv2.imshow("MASK", mask)
        if cv2.waitKey(1) & 0xFF == ord("g"):
            cv2.destroyAllWindows()
            index += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


index = 0


def trackGroundTape(img, screenWidth):
    # files = os.listdir("C:/vision-2019/Sample_Images/Ground Tape/High Exposure/")
    centerArray = []
    # file = files[index]
    # img = cv2.imread(
    #     "C:/vision-2019/Sample_Images/Ground Tape/High Exposure/" + file
    # )
    # ret, img = cap.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None)

    contours = cv2.findContours(mask, 1, 2)[-2]
    # sorted(contours, key=cv2.contourArea)

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        xCoord = rect[0][0]
        distanceFromCenter = abs((screenWidth / 2) - xCoord)

        centerArray.append((distanceFromCenter, rect))

        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(img, [box], -1, (0, 0, 255), 1)

    centerArray.sort(key=lambda a: a[0])
    box = cv2.boxPoints(centerArray[0][1])
    box = np.int0(box)
    # print(box)
    cv2.drawContours(img, [box], -1, (0, 0, 255), 1)

    cv2.imshow("IMG", img)
    cv2.imshow("MASK", mask)


cv2.destroyAllWindows()

while True:
    file = files[index]
    img = cv2.imread("C:/vision-2019/Sample_Images/Ground Tape/High Exposure/" + file)
    trackGroundTape(img, 320)
    if cv2.waitKey(1) & 0xFF == ord("g"):
        cv2.destroyAllWindows()
        index += 1
