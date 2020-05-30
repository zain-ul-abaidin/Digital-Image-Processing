import cv2 as cv
import numpy as np

def padding(originalImg, padSize):
    padImg = np.zeros((rows+2*padSize, columns+2*padSize), dtype=np.uint8)
    # Using Splicing
    padImg[padSize:rows+padSize, padSize:columns+padSize] = originalImg
    return padImg

def convolution(padImg, mask, size):
    convImg = np.zeros((rows, columns), dtype=np.uint32)
    for i in range(0, rows):
        for j in range(0, columns):
            # using Splicing
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
def powerLawTranformation(img, gamma):
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype='uint8')
    return gamma_corrected

# Laplacian Mask
mask = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
size = 3
p_size = size//2
# image reading
orginalImg = cv.imread('orgSkeleton.tif', 0)
# getting size of image
orginalImg = cv.GaussianBlur(orginalImg, (3, 3), cv.BORDER_DEFAULT)
rows = orginalImg.shape[0]
columns = orginalImg.shape[1]
# padding function call
padImg = padding(orginalImg, p_size)
# convolution function call
convImg = convolution(padImg, mask, size)
# normalization function call
lapImg = normalization(convImg)
sharpedImg = cv.addWeighted(lapImg, 1, orginalImg, 1, 0.0)
cv.imwrite('(b)_LAPACIAN_IMAGE.jpg', lapImg)
cv.imwrite('(c)_SHARPED_IMAGE.jpg', sharpedImg)

# Sobel Mask
mask = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# convolution function call
convImg = convolution(padImg, mask, size)
# normalization function call
normImg = normalization(convImg)
sobelImg = cv.addWeighted(normImg, 1, orginalImg, 1, 0.0)
cv.imwrite('(d)_SOBEL_IMAGE.jpg', sobelImg)

# Smoothing mask
mask = [[0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.04, 0.04, 0.04]]
# padding size
size = 5
p_size = size//2
# padding function call
padsobImg = padding(sobelImg, p_size)
# convolution function call
convImg = convolution(padsobImg, mask, size)
# normalization function call
normImg = normalization(convImg)
smoothsobelImg = cv.addWeighted(normImg, 1, orginalImg, 1, 0.0)
cv.imwrite('(e)_SMOOTHED_SOBEL_IMAGE.jpg', smoothsobelImg)

maskImg = cv.bitwise_and(smoothsobelImg, sharpedImg)
cv.imwrite('(f)_MASK_IMAGE_.jpg', maskImg)

SharpestImg = cv.addWeighted(maskImg, 1, orginalImg, 1, 0.0)
cv.imwrite('(g)_SHARPEST_IMAGE.jpg', smoothsobelImg)

powerImg = powerLawTranformation(SharpestImg, 0.4)
cv.imwrite('(h)_POWER_LAW_IMAGE.jpg', powerImg)
