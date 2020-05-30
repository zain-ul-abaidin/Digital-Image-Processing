import cv2 as cv
import numpy as np
# padding
def padding(originalImg, padSize):
    padImg = np.zeros((rows+2*padSize, columns+2*padSize), dtype=np.uint8)
    # using Slicing
    padImg[padSize:rows+padSize, padSize:columns+padSize] = originalImg
    return padImg
# Morphological Erosion
def Erosion(padImg, kernel, size):
    output = np.zeros((rows, columns), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            # Slicing
            portion = padImg[i:i+size, j:j+size]
            portion1 = portion.flatten()
            portion2 = kernel.flatten()
            # sum of kernel and window
            p1 = (np.sum(portion1))
            p2 = (np.sum(portion2))*255
            # if Fit condition satisfies
            if p1 == p2:
                output[i, j] = 255
            else:
                output[i, j] = np.min(portion1)
    return output
size = 19
# Structuring Element
kernel = np.ones((size, size), np.uint8)
# padding size
p_size = size//2
# image reading
orginalImg = cv.imread('erosion.jpg', 0)
# getting size of image
rows = orginalImg.shape[0]
columns = orginalImg.shape[1]
# padding function call
padImg = padding(orginalImg, p_size)
# Morphological Erosion
Ero = Erosion(padImg, kernel, size)
# erode image show
cv.imshow('output', Ero)
cv.waitKey(0)
