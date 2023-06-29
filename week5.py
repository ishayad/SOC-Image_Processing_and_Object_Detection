import cv2 as cv
import numpy as np


def porosityCal(imgPath):
    img = cv.imread(imgPath)
    # cv.imshow('balls',img)

    grayedImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow('Gray Image', grayedImg)

    blurred = cv.GaussianBlur(grayedImg, (7, 7), cv.BORDER_DEFAULT)
    #cv.imshow('Blur', blurred)

    _, binary_image = cv.threshold(blurred, 150, 255, cv.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)

    contours, hierarchies = cv.findContours(image, 1, 2)  # cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    total_area = 0
    for contour in contours:
        total_area += cv.contourArea(contour)

    # Calculate the porosity percentage
    image_area = img.shape[0] * img.shape[1]
    porosity = (total_area / image_area) * 100
    return porosity

imgPath = 'SampleImage.jpg'
porosity = porosityCal(imgPath)
print('Porosity: ',porosity)

cv.waitKey(0)
cv.destroyAllWindows()