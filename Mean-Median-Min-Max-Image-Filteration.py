import cv2 as cv
import numpy as np

def padding(originalImg, padSize):
    padImg = np.zeros((rows+2*padSize, columns+2*padSize), dtype=np.uint8)
    # Using Splicing
    padImg[padSize:rows+padSize, padSize:columns+padSize] = originalImg
    return padImg

def MinFiltering(padImg, size):
    output = np.zeros((rows, columns), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            # using Splicing
            portion = padImg[i:i+size, j:j+size]
            # Converting Matrix to Array
            array1 = portion.flatten()
            Minv = np.min(array1)
            output[i][j] = Minv
    return output

def MaxFiltering(padImg, size):
    output = np.zeros((rows, columns), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            # using Splicing
            portion = padImg[i:i+size, j:j+size]
            # Converting Matrix to Array
            array1 = portion.flatten()
            Maxv = np.max(array1)
            output[i][j] = Maxv
    return output

def MedianFiltering(padImg, size):
    output = np.zeros((rows, columns), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            # using Splicing
            portion = padImg[i:i+size, j:j+size]
            # Converting Matrix to Array
            array1 = portion.flatten()
            medianv = np.lib.median(array1)
            output[i][j] = medianv
    return output

def MeanFiltering(padImg, size):
    output = np.zeros((rows, columns), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            # using Splicing
            portion = padImg[i:i+size, j:j+size]
            # Converting Matrix to Array
            array1 = portion.flatten()
            meanv = np.mean(array1)
            output[i][j] = meanv
    return output

# Taking input of Mask size
size = int(input('enter portion size: '))
# padding size
p_size = size//2
# image reading
orginalImg = cv.imread('home05Fig02.png', 0)
# getting size of image
rows = orginalImg.shape[0]
columns = orginalImg.shape[1]
# padding function call
padImg = padding(orginalImg, p_size)
# Max Function call
maxFilImg = MaxFiltering(padImg, size)
# Min Function call
minFilImg = MinFiltering(padImg, size)
# Median Function call
medianFilImg = MedianFiltering(padImg, size)
# Mean Function call
meanFilImg = MeanFiltering(padImg, size)
# Image Show
cv.imshow('Max Filtered Image', maxFilImg)
cv.waitKey(0)
cv.imshow('Min Filtered Image', minFilImg)
cv.waitKey(0)
cv.imshow('Median Filtered Image', medianFilImg)
cv.waitKey(0)
cv.imshow('Mean Filtered Image', meanFilImg)
cv.waitKey(0)
cv.imwrite('HomeTask02Max.jpg', maxFilImg)
cv.imwrite('HomeTask02Min.jpg', minFilImg)
cv.imwrite('HomeTask02Median.jpg', medianFilImg)
cv.imwrite('HomeTask02Mean.jpg', meanFilImg)
