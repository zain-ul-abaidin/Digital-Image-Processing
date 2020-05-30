import cv2
import numpy as np
# Image read
img = cv2.imread(r'C:\Users\ZAIN UL ABAIDIN\Downloads\Fig0417(a)(barbara).tif', 0)
img = cv2.resize(img, (402, 402))
cv2.imshow('img', img)
cv2.waitKey(0)
size = img.shape[0]
# Cut off Frequency
Do = 30
# High pass Filter using Distance Matrix
def FilterDesign(img, size, Do):
    # D is distance Matrix
    D = np.zeros([size, size], dtype=np.uint32)
    # H is Filter
    H = np.zeros([size, size], dtype=np.uint8)
    r = img.shape[0] // 2
    c = img.shape[1] // 2
    # Distance Vector
    for u in range(0, size):
        for v in range(0, size):
            D[u, v] = abs(u - r) + abs(v - c)
    # Using Cut off frequncy applying 0 and 255 in H to make a High Pass Filter and center = 1
    for i in range(size):
        for j in range(size):
            if D[i, j] > Do:
                H[i, j] = 255
            else:
                H[i, j] = 0
    return H
# High Pass Filter
H = FilterDesign(img, size, Do)
cv2.imshow('Rectangulat High Pass Filter', H)
cv2.waitKey(0)
# Applying fft and shift
input = np.fft.fftshift(np.fft.fft2(img))
# Normalizing the absolute of fft of image = Magnitude Spectrum
cv2.imshow('Magnitude Spectrum', cv2.normalize(np.abs(input), None, 0, 255, cv2.NORM_MINMAX, -1))
cv2.waitKey(0)
# Multiplying image with Low Pass Filter
out = input*H
# Taking Inverse Fourier of image
out = np.abs(np.fft.ifft2(np.fft.ifftshift(out)))
out = np.uint8(cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX, -1))
# Gradient image after applying High pass filter
cv2.imshow('Magnitude Gradient', out)
cv2.waitKey(0)
