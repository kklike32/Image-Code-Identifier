import cv2
import numpy as np
import imutils
from imutils.contours import sort_contours
import matplotlib.pyplot as plt

img = cv2.imread("handCode1.png", cv2.IMREAD_GRAYSCALE)  # read image in grayscale

plt.imshow(img, cmap='gray')  # show original image
plt.title('Original')
plt.axis('off')
plt.show()

ret, thresh = cv2.threshold(img, 245, 255, cv2.THRESH_BINARY_INV)  # apply threshold
plt.imshow(thresh, cmap='gray')  # show thresholded image
plt.title('Thresholded')
plt.axis('off')
plt.show()

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

# Create a copy of the original image to draw contours on
img_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_with_contours, cnts, -1, (0, 255, 0), 3)

# Draw convex hulls
hulls = []
for c in cnts:
    hull = cv2.convexHull(c)
    hulls.append(hull)
    cv2.drawContours(img_with_contours, [hull], -1, (0, 0, 255), 2)

plt.imshow(img_with_contours)
plt.title('Contours with Convex Hulls')
plt.axis('off')
plt.show()

# Get the first convex hull (top left)
first_hull = hulls[0]

# Find the bounding rectangle of the first convex hull
x, y, w, h = cv2.boundingRect(first_hull)

# Calculate the dimensions of the grid based on the bounding rectangle
grid_width = w
grid_height = h

print("Grid Dimensions:")
print("Width:", grid_width)
print("Height:", grid_height)
