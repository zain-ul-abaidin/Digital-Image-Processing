import cv2
import numpy as np
# image read
img = cv2.imread(r'C:\Users\ZAIN UL ABAIDIN\Downloads\Fig0424(a)(rectangle).tif', 0)
img = cv2.resize(img, (700, 700))
cv2.imshow('img', img)
cv2.waitKey(0)
# Applying Fourier and Fourier Shift
input = np.fft.fftshift(np.fft.fft2(img))
# Magnitude Spectrum = Normalizing the absolute of fft of image
cv2.imshow('Magnitude Spectrum', cv2.normalize(np.abs(input), None, 0, 255, cv2.NORM_MINMAX, -1))
cv2.waitKey(0)
# Taking inverse and converting it to spatial domain again
out = np.abs(np.fft.ifft2(np.fft.ifftshift(input)))
out = np.uint8(cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX, -1))
cv2.imshow('Spatial Domain', out)
cv2.waitKey(0)
