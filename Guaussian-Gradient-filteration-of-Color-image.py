import cv2
import numpy as np
# image read in RGB
img = cv2.imread('lenna_RGB.tif', cv2.IMREAD_COLOR)
# img spliting in red, green, and blue
r, g, b = cv2.split(img)
# Image size
rows = img.shape[0]
cols = img.shape[1]
# mask size
size = 3
# Guassian Filter
Gmask = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
# SobelX Filter
SXmask = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
# SobelY Filter
SYmask = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
# padSize
p_size = size//2
# padding Function
def padding(originalImg, padSize):
    padImg = np.zeros((rows+2*padSize, cols+2*padSize), dtype=np.uint8)
    # Slicing
    padImg[padSize:rows+padSize, padSize:cols+padSize] = originalImg
    return padImg
# Convolution of mask with image
def convolution(padImg, mask):
    convImg = np.zeros((rows, cols), dtype=np.uint32)
    for i in range(0, rows):
        for j in range(0, cols):
            # using Slicing
            prod = np.multiply(padImg[i:i+size, j:j+size], mask)
            # Taking sum of the matrix and then taking absolute of that
            sum1 = np.absolute(np.sum(prod))
            convImg[i][j] = sum1
    return convImg
# Normailzation  Function
def normalization(convImg):
    minimum = np.min(convImg)
    maximum = np.max(convImg)
    normImg = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, cols):
            # Using Normalization Formula
            normImg[i][j] = (255*(convImg[i][j]-minimum))//(maximum-minimum)
    return normImg
# Function call
outR = normalization(convolution(padding(r, p_size), Gmask))
outG = normalization(convolution(padding(g, p_size), Gmask))
outB = normalization(convolution(padding(b, p_size), Gmask))
SxoutR = normalization(convolution(padding(r, p_size), SXmask))
SxoutG = normalization(convolution(padding(g, p_size), SXmask))
SxoutB = normalization(convolution(padding(b, p_size), SXmask))
SyoutR = normalization(convolution(padding(r, p_size), SYmask))
SyoutG = normalization(convolution(padding(g, p_size), SYmask))
SyoutB = normalization(convolution(padding(b, p_size), SYmask))
# Guassion applied
GuassianOut = cv2.merge((outR, outG, outB))
# Sobel applied
SX = cv2.merge((SxoutR, SxoutG, SxoutB))
SY = cv2.merge((SyoutR, SyoutG, SyoutB))
# Magnitude Image
SobelOut = SX + SY
cv2.imshow('GuassianOutput', GuassianOut)
cv2.waitKey(0)
cv2.imshow('SobelOutput', SobelOut)
cv2.waitKey(0)
