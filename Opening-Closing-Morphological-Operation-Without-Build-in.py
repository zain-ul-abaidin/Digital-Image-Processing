import cv2 as cv
import numpy as np
# padding
def padding(originalImg, padSize):
    padImg = np.zeros((rows+2*padSize, columns+2*padSize), dtype=np.uint8)
    # Slicing
    padImg[padSize:rows+padSize, padSize:columns+padSize] = originalImg
    return padImg
def Erosion(padImg, kernel, size):
    output = np.zeros((rows, columns), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            # Slicing
            portion = padImg[i:i+size, j:j+size]
            # sum of Kernel and window
            portion1 = portion.flatten()
            portion2 = kernel.flatten()
            p1 = (np.sum(portion1))
            p2 = (np.sum(portion2))*255
            # if Fit Condition Satisfies
            if p1 == p2:
                output[i, j] = 255
            else:
                output[i, j] = np.min(portion1)
    return output
def Dilation(padImg, size):
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
    erosion = Erosion(padImg, kernel , size)
    padImg2 = padding(erosion, size//2)
    # secondly apply Dilation on Eroded
    output = Dilation(padImg2, size)
    return output
def closing(padImg,kernel, size):
    # First apply Dilation
    dilation = Dilation(padImg, size)
    padImg2 = padding(dilation, size//2)
    # secondly apply Erosion on Dilated
    output = Erosion(padImg2, kernel, size)
    return output
size = 3
# Structuring Element
kernel = np.ones((size, size), np.uint8)
print(np.sum(kernel))
# padding size
p_size = size//2
# image reading
orginalImg = cv.imread('noisy_fingerprint.tif', 0)
# getting size of image
rows = orginalImg.shape[0]
columns = orginalImg.shape[1]
# padding function call
padImg = padding(orginalImg, p_size)
# First apply opening to remove small objects
open = opening(padImg, kernel, size)
cv.imshow('output1', open)
cv.waitKey(0)
padImg2 = padding(open, p_size)
# Then applu closing to remove gaps
output = closing(padImg2, kernel, 3)
cv.imshow('output2', output)
cv.waitKey(0)
