import numpy as np
import cv2

def kMeanClustering(img, color1, color2, iteration):
    # count is how many iterations you want
    count = 3
    while count > 0:
        # As our given image to extract lesion has two regions so i made two clusters
        Clus1 = []
        Clus2 = []
        for i in range(rows):
            for j in range(columns):
                # dividing pixels on the basis of their difference from there center value
                # and appending them into two clusters
                dis1 = img[i, j] - color1
                dis2 = img[i, j] - color2
                if abs(dis2) > abs(dis1):
                    Clus1.append(img[i, j])
                else:
                    Clus2.append(img[i, j])
        # so now here we have done our first iteration
        count = count - 1
        # updating the parameters for the next iteration
        color1 = np.mean(Clus1)
        color2 = np.mean(Clus2)
    # now here i have two colors only on image
    # so i will make it black and white image on basis of its two colors
    # and its background would be black
    if color1 > color2:
        color1 = 0
        color2 = 255
    else:
        color1 = 255
        color2 = 0
    for i in range(rows):
        for j in range(columns):
            # Now here I am implementing results from Kmean clustering
            if img[i, j] in Clus1:
                img[i, j] = color1
            elif img[i, j] in Clus2:
                img[i, j] = color2
            cv2.imshow("progress", img)
            cv2.waitKey(1)
    return img

# This function is to remove the corners white area that are not lesions
def RemoveUnwantedRegions(output):
    new_img = np.zeros_like(output)
    for val in np.unique(output)[1:]:
        # mask
        mask = np.uint8(output == val)
        # Using Connected Component Analysis
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # updating on condition
        new_img[labels == largest_label] = val
    return new_img

# image read
img = cv2.imread("IMD002.bmp", 0)
# image size
rows, columns = np.shape(img)
# calling clustering function
output = kMeanClustering(img, 80, 4)
# removing unwanted regions
out = RemoveUnwantedRegions(output)
# removing noise
opening = cv2.medianBlur(img1, 23)
# image save
cv2.imwrite('output.png', opening)
