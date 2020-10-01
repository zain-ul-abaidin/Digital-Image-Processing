#importing libraries
import cv2 as cv
import numpy as np
# image read in RGB
img = cv2.imread('RGB-full-color-cube.tif', cv2.IMREAD_COLOR)
# Image dimensions
r = img.shape[0]
c = img.shape[1]
# Function to Convert RGB image into HSV image
def BGR2HSV(img):
    H = np.zeros((r, c), dtype=np.uint32)
    S = np.zeros((r, c), dtype=np.float)
    V = np.zeros((r, c), dtype=np.float)
    for i in range(r):
        for j in range(c):
            # pixel with R, G and B values in array
            pix = img[i, j]
            # The R,G, B values are divided by 255 for the normalization
            B = pix[0] / 255
            G = pix[1] / 255
            R = pix[2] / 255
            array = [B, G, R]
            Cmax = np.max(array)
            Cmin = np.min(array)
            Delta = Cmax - Cmin
            # Hue Calculation
            if Delta == 0:
                H[i, j] = 0
            elif Cmax == R:
                H[i, j] = 60 * np.mod((G - B) / Delta, 6)
            elif Cmax == G:
                H[i, j] = 60 * (((B - R) / Delta) + 2)
            elif Cmax == B:
                H[i, j] = 60 * (((R - G) / Delta) + 4)
            # Saturation Calculation
            if Cmax == 0:
                S[i, j] = 0
            else:
                S[i, j] = (Delta / Cmax)
            # Intensity Value
            V[i, j] = Cmax
    return H, S, V
# Normalization
def normalization(out):
    minimum = np.min(out)
    maximum = np.max(out)
    normImg = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(0, r):
        for j in range(0, c):
            # Using Normalization Formula
            normImg[i][j] = (255*(out[i][j]-minimum))//(maximum-minimum)
    return normImg
H, S, V = BGR2HSV(img)
# Applying Normalization on all the three planes
H = normalization(H)
S = normalization(S)
V = normalization(V)
# Displaying the hue,saturation and value images 
cv.imshow('Hue', H)
cv.waitKey(0)
cv.imshow('Saturation', S)
cv.waitKey(0)
cv.imshow('Intensity Value', V)
cv.waitKey(0)
cv.imshow('HSV', cv2.merge((H, S, V)))
cv.waitKey(0)
