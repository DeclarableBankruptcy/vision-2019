import numpy as np
import cv2
import os
import math


lower_green = np.array([60, 177, 177])
upper_green = np.array([160, 255, 255])
lower_white = 100
upper_white = 255
files = os.listdir("C:/vision-2019/Sample_Images/Ground Tape/High Exposure/")

'''
def trackRetroTape(img):
    files = os.listdir("C:/Users/User1/Desktop/Vision Images/Retro Tape/Low Exposure/")
    index = 0

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
'''


def trackGroundTape(img):
    boxPointsOnScreen = []
    # centerArray = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, lower_white, upper_white)

    screenWidth = np.shape(img)[1]
    screenHeight = np.shape(img)[0]

    halfScreenWidth = screenWidth / 2
    halfScreenHeight = screenHeight / 2

    # cv2.morphologyEx(mask, cv2.MORPH_OPEN, None)
    # cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None)

    contours = cv2.findContours(mask, 1, 2)[-2]
    contours = filterNoise(contours)  # Filter small contours from the screen.

    if len(contours) > 0:
        '''
        Center code:
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            xCoord = rect[0][0]
            distanceFromCenter = abs((screenWidth / 2) - xCoord)

            centerArray.append((distanceFromCenter, rect))

        centerArray.sort(key=lambda a: a[0])
        box = cv2.boxPoints(centerArray[0][1])
        box = np.int0(box)
        '''
        contours.sort(key=cv2.contourArea, reverse=True)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.drawContours(img, [box], -1, (0, 0, 255), 1)

        cv2.imshow("IMG", img)
        cv2.imshow("MASK", mask)

        boxPointsOnScreen = np.array(filterPoints(box, screenWidth, screenHeight))  # Get all the boxpoints on the screen.

        # Cases:
        if len(boxPointsOnScreen) == 2:  # If two points are on the screen, return the point between them.
            averageX = np.mean(boxPointsOnScreen, axis=0)[0]
            averageY = np.mean(boxPointsOnScreen, axis=0)[1]

            x_float = -((averageX / halfScreenWidth) - 1)
            y_float = -((averageY / halfScreenHeight) - 1)

            return x_float, y_float, findAngle((boxPointsOnScreen[0], boxPointsOnScreen[1]), oneSide=True)
        elif len(boxPointsOnScreen) == 1:  # If one point is ont the screen, return its coordinates.
            x_float = -((boxPointsOnScreen[0][0] / halfScreenWidth) - 1)
            y_float = -((boxPointsOnScreen[0][1] / halfScreenHeight) - 1)

            return 0, x_float, y_float
        else:  # For any other case, return the average point of the contour.
            averageX = np.mean(box, axis=0)[0]
            averageY = np.mean(box, axis=0)[1]

            x_float = -((averageX / halfScreenWidth) - 1)
            y_float = -((averageY / halfScreenHeight) - 1)

            return x_float, y_float, findAngle(box)

    # Return 0 if no contours found.
    return float("NaN"), float("NaN"), float("NaN")


def findAngle(points, oneSide=False):
    if oneSide:
        side = (points[0], points[1])
        opposite = side[0][0] - side[1][0]
        adjacent = side[0][1] - side[1][1]
        angle = math.atan2(opposite, adjacent)
        angle -= 1.5708  # Subtract 90 degrees in radians.
    else:
        side = findLongestSide((points[0], points[1]), (points[1], points[2]))
        opposite = side[0][0] - side[1][0]
        adjacent = side[0][1] - side[1][1]
        angle = math.atan2(opposite, adjacent)

    return math.degrees(angle)


def filterPoints(box, screenWidth, screenHeight):
    boxPointsOnScreen = []
    for point in box:
        if point[0] > 5 and point[1] > 5 and point[0] < screenWidth - 5 and point[1] < screenHeight - 5:
            boxPointsOnScreen.append(point)
    return boxPointsOnScreen


def filterNoise(contours):
    cnts = []
    for contour in contours:
        if cv2.contourArea(contour) > 10:
            cnts.append(contour)
    return cnts


def findLongestSide(side1, side2):
    longest_side = max((side1, side2), key=lambda a: abs(a[0][0] - a[1][0]) + abs(a[0][1] - a[1][1]))
    return longest_side
    

if __name__ == '__main__':
    index = 0
    while True:
        file = files[index]
        img = cv2.imread("C:/vision-2019/Sample_Images/Ground Tape/High Exposure/" + file)

        print(trackGroundTape(img))
        if cv2.waitKey(0) & 0xFF == ord('g'):
            cv2.destroyAllWindows()
            index += 1
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break