# This is a sample Python script.
import cv2
import numpy as np
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

img = cv2.imread("path", cv2.IMREAD_GRAYSCALE) #read in image in grayscale

cv2.imshow("original", img) #shows image
cv2.waitKey()
cv2.destroyAllWindows()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
