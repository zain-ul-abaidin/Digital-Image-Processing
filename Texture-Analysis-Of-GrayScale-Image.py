from skimage.feature import greycomatrix, greycoprops
import cv2
# To present and save data in a formal and proper way
import pandas as pd

# image1
img1 = cv2.imread(r'C:\Users\ZAIN UL ABAIDIN\Downloads\lab12a.tif', 0)
res1 = greycomatrix(img1, [1], [0], levels=256)
GLCM = res1[:, :, 0, 0]
print(GLCM)
# contrast
cont1 = greycoprops(res1, prop='contrast')
# disimilarity
diss1 = greycoprops(res1, prop='dissimilarity')
# homogeneity
homo1 = greycoprops(res1, prop='homogeneity')
# energy
en1 = greycoprops(res1, prop='energy')
# correlation
corr1 = greycoprops(res1, prop='correlation')
# ASM
asm1 = greycoprops(res1, prop='ASM')
# image2
img2 = cv2.imread(r'C:\Users\ZAIN UL ABAIDIN\Downloads\lab12.tif', 0)
res2 = greycomatrix(img2, [1], [0], levels=256)
GLCM = res2[:, :, 0, 0]
print(GLCM)
# contrast
cont2 = greycoprops(res2, prop='contrast')
# disimilarity
diss2 = greycoprops(res2, prop='dissimilarity')
# homogeneity
homo2 = greycoprops(res2, prop='homogeneity')
# energy
en2 = greycoprops(res2, prop='energy')
# correlation
corr2 = greycoprops(res2, prop='correlation')
# ASM
asm2 = greycoprops(res2, prop='ASM')

# image3
img3 = cv2.imread(r'C:\Users\ZAIN UL ABAIDIN\Downloads\lab12(b).tif', 0)
res3 = greycomatrix(img3, [1], [0], levels=256)
GLCM = res3[:, :, 0, 0]
print(GLCM)
# contrast
cont3 = greycoprops(res3, prop='contrast')
# disimilarity
diss3 = greycoprops(res3, prop='dissimilarity')
# homogeneity
homo3 = greycoprops(res3, prop='homogeneity')
# energy
en3 = greycoprops(res3, prop='energy')
# correlation
corr3 = greycoprops(res3, prop='correlation')
# ASM
asm3 = greycoprops(res3, prop='ASM')

# Data Saved to an Excel file using Pandas DataFrame
d = {'Contrast': [cont1, cont2, cont3], 'Dissimilarity': [diss1, diss2, diss3], 'Homogeneity': [homo1, homo2, homo3],
     'Energy': [en1, en2, en3], 'Correlation': [corr1, corr2, corr3], 'ASM': [asm1, asm2, asm3]}
df1 = pd.DataFrame(data=d)
df1.index = ['lab12a', 'lab12', 'lab12b']
df1.to_excel('Lab12.xlsx')
print(df1)
