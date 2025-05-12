import cv2
import numpy as np


def nothing(x):
    pass


while True:
    image = cv2.imread('screenshot_cam0.webp')

    # Set minimum and maximum HSV values to display
    lower_line = np.array([12, 163, 163])
    upper_line = np.array([33, 255, 255])

    # Convert to HSV format and color threshold
    hsv_line = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_line = cv2.inRange(hsv_line, lower_line, upper_line)
    result = cv2.bitwise_and(image, image, mask=mask_line)

    # Display result image
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
