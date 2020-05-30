import cv2 as cv
import numpy as np

def mask(size):
    mask = np.zeros((size, size), dtype=np.float)
    for i in range(size):
        for j in range(size):
            mask[i, j] = int(input('enter values: '))
    return mask

def padding(originalImg, padSize):
    padImg = np.zeros((rows+2*padSize, columns+2*padSize), dtype=np.uint8)
    # Using Slicing
    padImg[padSize:rows+padSize, padSize:columns+padSize] = originalImg
    return padImg

def convolution(padImg, mask):
    convImg = np.zeros((rows, columns), dtype=np.uint32)
    for i in range(0, rows):
        for j in range(0, columns):
            # using Slicing
            prod = np.multiply(padImg[i:i+size, j:j+size], mask)
            # Taking sum of the matrix and then taking absolute of that
            sum1 = np.absolute(np.sum(prod))
            convImg[i][j] = sum1
    return convImg

def normalization(convImg):
    minimum = np.min(convImg)
    maximum = np.max(convImg)
    normImg = np.zeros((rows, columns), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            # Using Normalization Formula
            normImg[i][j] = (255*(convImg[i][j]-minimum))//(maximum-minimum)
    return normImg

# mask size input
size = int(input('enter mask size: '))
# mask function call
mask = mask(size)
# padding size
p_size = size//2
# image reading
orginalImg = cv.imread('lab5Fig1.tif', 0)
# getting size of image
# orginalImg = cv.GaussianBlur(orginalImg, (3, 3), cv.BORDER_DEFAULT)
rows = orginalImg.shape[0]
columns = orginalImg.shape[1]
# padding function call
padImg = padding(orginalImg, p_size)
# convolution function call
convImg = convolution(padImg, mask)
# normalization function call
normImg = normalization(convImg)
cv.imshow('Filtered Image', normImg)
cv.waitKey(0)
cv.imwrite('LAB05Task05.jpg', normImg)
