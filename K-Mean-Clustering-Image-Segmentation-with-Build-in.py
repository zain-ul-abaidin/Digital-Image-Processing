#Importing the libraries
import numpy as np
import cv2 as cv
# To Read Folder of images
from os import listdir
# To  join the path and read the files from that path
from os.path import isfile, join
# To store DiceCoefficient of all images in Excel
import pandas as pd
# Function to remove the smallest objects using cca and only show the biggest one
def RemoveSmallRegions(output):
    new_img = np.zeros_like(output)
    for val in np.unique(output)[1:]:
        mask = np.uint8(output == val)  # step 3
        labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_img[labels == largest_label] = val
    return new_img
# Dice Coefficient to find the performance efficiency of Algorithm including truepositive,falsepositive,falsenegative,truenegative
def DiceCoefficient(output, actualImg):
    TP = 0    #true positive 
    FP = 0    #false positive
    FN = 0   #false negative
    TN = 0   #true negative
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if output[i][j] == 255 and actualImg[i][j] == 255:
                TP = TP + 1
            elif output[i][j] == 0 and actualImg[i][j] == 255:
                FP = FP + 1
            elif output[i][j] == 255 and actualImg[i][j] == 0:
                FN = FN + 1
            elif output[i][j] == 0 and actualImg[i][j] == 0:
                TN = TN + 1
    dice_coefficient = (2 * TP) / (FN + (2 * TP) + FP)
    return dice_coefficient
# DiceCoefficients for every image will append to this array
diceCoeff = []
# Images names in a specific folder are going to append in an array
mypath1 = r'C:\Users\ZAIN UL ABAIDIN\Desktop\NewKM'
onlyfiles1 = [f for f in listdir(mypath1) if isfile(join(mypath1,f))]
images1 = np.empty(len(onlyfiles1), dtype=object)
mypath2 = r'C:\Users\ZAIN UL ABAIDIN\Desktop\actualoutput'
onlyfiles2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]
images2 = np.empty(len(onlyfiles2), dtype=object)
# Writing every output in a folder with different name
strin = ['D002', 'D003', 'D004', 'D006', 'D008', 'D009', 'D010', 'D013', 'D014', 'D015',
         'D016', 'D017', 'D018', 'D019', 'D020', 'D021', 'D022', 'D023', 'D024', 'D025',
         'D027', 'D030', 'D031', 'D032', 'D033', 'D035', 'D036', 'D037', 'D038', 'D039',
         'D040', 'D041', 'D042', 'D043', 'D044', 'D045', 'D047', 'D048', 'D049', 'D050',
         'D057', 'D058', 'D061', 'D063', 'D064', 'D065', 'D075', 'D076', 'D078', 'D080',
         'D085', 'D088', 'D090', 'D091', 'D092', 'D101', 'D103', 'D105', 'D107', 'D108',
         'D112', 'D118', 'D120', 'D125', 'D126', 'D132', 'D133', 'D134', 'D135', 'D137',
         'D138', 'D139', 'D140', 'D142', 'D143', 'D144', 'D146', 'D147', 'D149', 'D150',
         'D152', 'D153', 'D154', 'D155', 'D156', 'D157', 'D159', 'D160', 'D161', 'D162',
         'D164', 'D166', 'D168', 'D169', 'D170', 'D171', 'D173', 'D175', 'D176', 'D177',
         'D182', 'D196', 'D197', 'D198', 'D199', 'D200', 'D203', 'D204', 'D206', 'D207',
         'D208', 'D210', 'D211', 'D219', 'D226', 'D240', 'D242', 'D243', 'D251', 'D254',
         'D256', 'D278', 'D279', 'D280', 'D284', 'D285', 'D304', 'D305', 'D306', 'D312',
         'D328', 'D331', 'D339', 'D347', 'D348', 'D349', 'D356', 'D360', 'D364', 'D365',
         'D367', 'D368', 'D369', 'D370', 'D371', 'D372', 'D374', 'D375', 'D378', 'D379',
         'D380', 'D381', 'D382', 'D383', 'D384', 'D385', 'D386', 'D388', 'D389', 'D390',
         'D392', 'D393', 'D394', 'D395', 'D396', 'D397', 'D398', 'D399', 'D400', 'D402',
         'D403', 'D403', 'D404', 'D405', 'D406', 'D407', 'D408', 'D409', 'D410', 'D411',
         'D413', 'D417', 'D418', 'D419', 'D420', 'D421', 'D423', 'D424', 'D425', 'D426',
         'D427', 'D429', 'D430', 'D431', 'D432', 'D433', 'D434', 'D435', 'D436', 'D437']
# loop to read 200 images one by one
for i in range(0, len(onlyfiles1)):
    # given image to extract lesion
    orginalImg = cv2.imread(join(mypath1, onlyfiles1[i]), 0)
    # given Extracted lesion image
    actualImg = cv2.imread(join(mypath2, onlyfiles2[i]), 0)
    # applying smoothing
    orginalImg = cv2.GaussianBlur(orginalImg, (15, 15), cv2.BORDER_DEFAULT)
    # applying closing to remove hair
    kernel = np.ones((9, 9), np.uint8)
    orginalImg = cv2.morphologyEx(orginalImg, cv2.MORPH_CLOSE, kernel, iterations=2)
    orginalImg = cv2.cvtColor(orginalImg, cv2.COLOR_BGR2RGB)
    pixel_values = orginalImg.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # Applying K means Clustering with K = 2
    K = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    labels = labels.flatten()
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(orginalImg.shape)
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    # Applying Threshold to clustered Image
    ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # Again applying closing to remove small black particles from the lesion part
    kernel = np.ones((9, 9), np.uint8)
    output = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Removing small regions that are not a part of lesion like corners
    output = RemoveSmallRegions(output)
    # finding Dice Coefficient
    df = DiceCoefficient(output, actualImg)
    diceCoeff.append(df)
    # saving every output with different name in a specific folder
    cv.imwrite(r'C:\Users\ZAIN UL ABAIDIN\Desktop\NewKM\IM' + strin[i] + '.bmp', output)

# Saving all DiceCoefficients to an Excel file
df1 = pd.DataFrame([diceCoeff]).transpose()
df1.columns = ['Dice Coefficient']
df1.index = ['img01', 'img02', 'img03', 'img04', 'img05', 'img06', 'img07', 'img08', 'img09', 'img10',
             'img11', 'img12', 'img13', 'img14', 'img15', 'img16', 'img17', 'img18', 'img19', 'img20',
             'img21', 'img22', 'img23', 'img24', 'img25', 'img26', 'img27', 'img28', 'img29', 'img30',
             'img31', 'img32', 'img33', 'img34', 'img35', 'img36', 'img37', 'img38', 'img39', 'img40',
             'img41', 'img42', 'img43', 'img44', 'img45', 'img46', 'img47', 'img48', 'img49', 'img50',
             'img51', 'img52', 'img53', 'img54', 'img55', 'img56', 'img57', 'img58', 'img59', 'img60',
             'img61', 'img62', 'img63', 'img64', 'img65', 'img66', 'img67', 'img68', 'img69', 'img70',
             'img71', 'img72', 'img73', 'img74', 'img75', 'img76', 'img77', 'img78', 'img79', 'img80',
             'img81', 'img82', 'img83', 'img84', 'img85', 'img86', 'img87', 'img88', 'img89', 'img90',
             'img91', 'img92', 'img93', 'img94', 'img95', 'img96', 'img97', 'img98', 'img99', 'img100',
             'img101', 'img102', 'img103', 'img104', 'img105', 'img106', 'img107', 'img108', 'img109', 'img110',
             'img111', 'img112', 'img113', 'img114', 'img115', 'img116', 'img117', 'img118', 'img119', 'img120',
             'img121', 'img122', 'img123', 'img124', 'img125', 'img126', 'img127', 'img128', 'img129', 'img130',
             'img131', 'img132', 'img133', 'img134', 'img135', 'img136', 'img137', 'img138', 'img139', 'img140',
             'img141', 'img142', 'img143', 'img144', 'img145', 'img146', 'img147', 'img148', 'img149', 'img150',
             'img151', 'img152', 'img153', 'img154', 'img155', 'img156', 'img157', 'img158', 'img159', 'img160',
             'img161', 'img162', 'img163', 'img164', 'img165', 'img166', 'img167', 'img168', 'img169', 'img170',
             'img171', 'img172', 'img173', 'img174', 'img175', 'img176', 'img177', 'img178', 'img179', 'img180',
             'img181', 'img182', 'img183', 'img184', 'img185', 'img186', 'img187', 'img188', 'img189', 'img190',
             'img191', 'img192', 'img193', 'img194', 'img195', 'img196', 'img197', 'img198', 'img199', 'img200']
#method to convert into the excel file
df1.to_excel('KMeanClusteringDiceCoeff.xlsx')
