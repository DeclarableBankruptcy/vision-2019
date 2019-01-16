import cv2


img = cv2.imread("C:/vision-2019/Tools/Sample Images/CargoLine36in.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def printHSV(event, x, y, flags, param):
    # Print each item in the pixel array
    print(hsv[y, x], x, y)


while True:
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', printHSV)

    cv2.imshow('image', img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        # Quit the program
        break

cv2.destroyAllWindows
