# This is a sample Python script.
import cv2
import numpy as np
import imutils
from imutils.contours import sort_contours
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

img = cv2.imread("/Users/aditya/PycharmProjects/handwrittenCodeGrading/handCode1.png", cv2.IMREAD_GRAYSCALE) #read in image in grayscale

cv2.imshow("original", img) #shows image
cv2.waitKey()


ret, thresh = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY_INV) #175 before
cv2.imshow("image2: ", thresh)
cv2.waitKey()

cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

cv2.drawContours(img, cnts, -1, (0,255,0), 3)
cv2.imshow("countours", img)
cv2.waitKey()

cv2.destroyAllWindows()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
