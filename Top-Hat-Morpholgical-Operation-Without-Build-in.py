import cv2 as cv
import numpy as np
# padding
def padding(originalImg, padSize):
    padImg = np.zeros((rows+2*padSize, columns+2*padSize), dtype=np.uint8)
    # Slicing
    padImg[padSize:rows+padSize, padSize:columns+padSize] = originalImg
    return padImg
def GrayErosion(padImg, kernel, size):
    output = np.zeros((rows, columns), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            # Slicing
            portion = padImg[i:i+size, j:j+size]
            portion1 = portion.flatten()
            portion2 = kernel.flatten()
            # sum of Kernel and window
            p1 = (np.sum(portion1))
            p2 = (np.sum(portion2))*255
            # if Fit Condition Satisfies
            if p1 == p2:
                output[i, j] = 255
            else:
                output[i, j] = np.min(portion1)
    return output
def GrayDilation(padImg, size):
    output = np.zeros((rows, columns), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            # Slicing
            portion = padImg[i:i+size, j:j+size]
            portion1 = portion.flatten()
            # if Hit Condition Satisfies
            if 255 in portion1:
                output[i, j] = 255
            else:
                output[i, j] = np.max(portion1)
    return output
def opening(padImg, kernel, size):
    # First apply Erosion
    erosion = GrayErosion(padImg, kernel, size)
    padImg2 = padding(erosion, size//2)
    # secondly apply Dilation on Eroded
    output = GrayDilation(padImg2, size)
    return output
size = 21
# Structuring Element
kernel = np.ones((size, size), np.uint8)
# padding size
p_size = size//2
# image reading
orginalImg = cv.imread('rice_image_with_intensity_gradient.tif', 0)
orginalImg = cv.medianBlur(orginalImg,5)
# getting size of image
rows = orginalImg.shape[0]
columns = orginalImg.shape[1]
# padding function call
padImg = padding(orginalImg, p_size)
# Opening
open = opening(padImg, kernel, size)
# Top-Hat Morphology
WhiteTopHat = orginalImg - open
cv.imshow('WhiteTopHat', WhiteTopHat)
cv.waitKey(0)
ret, thresh1 = cv.threshold(WhiteTopHat, 10, 255, cv.THRESH_BINARY)
cv.imshow('Thresholding', thresh1)
cv.waitKey(0)
