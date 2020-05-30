import cv2
import numpy as np
# image read in RGB
img = cv2.imread('RGB-full-color-cube.tif', cv2.IMREAD_COLOR)
# Image size
rows = img.shape[0]
cols = img.shape[1]
# Function to Convert RGB image to HSV image
def BGR2HSV(img):
    H = np.zeros((rows, cols), dtype=np.uint32)
    S = np.zeros((rows, cols), dtype=np.float)
    V = np.zeros((rows, cols), dtype=np.float)
    for i in range(rows):
        for j in range(cols):
            # pixel with R, G and B values in array
            pix = img[i, j]
            # The R,G, B values are divided by 255 to change the range from 0..255 to 0..1
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
            V[i, j] = Cmax*
    return H, S, V
# Normalization
def normalization(out):
    minimum = np.min(out)
    maximum = np.max(out)
    normImg = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, cols):
            # Using Normalization Formula
            normImg[i][j] = (255*(out[i][j]-minimum))//(maximum-minimum)
    return normImg
H, S, V = BGR2HSV(img)
# Applying Normalization
H = normalization(H)
S = normalization(S)
V = normalization(V)
# image Show
cv2.imshow('Hue', H)
cv2.waitKey(0)
cv2.imshow('Saturation', S)
cv2.waitKey(0)
cv2.imshow('Intensity Value', V)
cv2.waitKey(0)
cv2.imshow('HSV', cv2.merge((H, S, V)))
cv2.waitKey(0)
