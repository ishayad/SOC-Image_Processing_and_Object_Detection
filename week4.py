import cv2 as cv
import numpy as np

def contourCount(img):
    grayedImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow('Gray Image', grayedImg)

    blurred = cv.GaussianBlur(grayedImg, (7, 7), cv.BORDER_DEFAULT)
    #cv.imshow('Blur', blurred)

    canny = cv.Canny(blurred, 125, 175)
    #cv.imshow('canny', canny)

    kernel = np.ones((3, 3), np.uint8)
    img = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)

    dilated = cv.dilate(img, (1, 1), iterations=0)
    #cv.imshow('Dilated', dilated)

    contours, hierarchies = cv.findContours(dilated, 1, 2)  # cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    return len(contours)

img = cv.imread('ball.jpeg')
cv.imshow('balls',img)

print(f'Number of Objects: {contourCount(img)}')

cv.waitKey(0)
cv.destroyAllWindows()