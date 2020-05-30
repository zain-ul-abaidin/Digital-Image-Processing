import cv2 as cv
import numpy as np
# CCA to find no of objects in image
def ConnectedComponentAnalysis(img, newImage):
    LabelMat = np.array(newImage, dtype=np.uint16)
    # initializing equivalency list
    equivalency_list = [0]
    LabelVal = 0
    for i in range(rows):
        for j in range(columns):
            if img[i, j] == 1:
                top = LabelMat[i, j + 1]
                left = LabelMat[i + 1, j]
                if top == 0 and left == 0:
                    LabelVal = LabelVal + 1
                    equivalency_list.append(LabelVal)
                    LabelMat[i + 1, j + 1] = LabelVal
                elif top == 0 and left != 0:
                    LabelMat[i + 1, j + 1] = left
                elif top != 0 and left == 0:
                    LabelMat[i + 1, j + 1] = top
                elif top == left == 1:
                    LabelMat[i + 1, j + 1] = top
                else:
                    a = min(top, left)
                    b = max(top, left)
                    LabelMat[i + 1, j + 1] = a
                    for val in range(len(equivalency_list)):
                        if equivalency_list[val] == b:
                            equivalency_list[val] = a
    return equivalency_list

orginalImg = cv.imread('erosion.jpg', 0)
# getting size of image
rows = orginalImg.shape[0]
columns = orginalImg.shape[1]
# convertion image to 0's and 1's
for i in range(rows):
    for j in range(columns):
        if orginalImg[i, j] >= 127:
            orginalImg[i, j] = 1
        else:
            orginalImg[i, j] = 0
# padding
newImage = np.pad(orginalImg, pad_width=[(1, 1), (1, 1)], mode='constant', constant_values=0)
# Finding No of Objects by Applying cca
EquivalencyList = ConnectedComponentAnalysis(orginalImg, newImage)
NoOfObjects = np.count_nonzero(np.unique(EquivalencyList))
print('No of Objects: ', NoOfObjects)
