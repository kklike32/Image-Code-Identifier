import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
orig_img = 'semicolon.png'
image  = cv2.imread(orig_img)


# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise and detail in the image
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Binary thresholding
_, binary = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY_INV)

# Edge detection
edges = cv2.Canny(binary, 50, 150, apertureSize=3)

# Hough line transformation
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Draw the lines on the image
if lines is not None:
    for rho, theta in lines[:,0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

# Perform adaptive thresholding to segment the image into cells
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area to remove noise
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

# Draw the filtered contours on the original image
cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 3)

# Display the image with grid
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.show()

green_img = 'greenLines4.png'
cv2.imwrite(green_img, image_rgb)


def extract_boxes(image_path, orig_path, lower_green=np.array([35, 100, 50]), upper_green=np.array([85, 255, 255]), black_thresh=50):
    # Load the image
    image = cv2.imread(image_path)
    orig = cv2.imread(orig_path)

    # Convert the image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the green color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask= mask)

    # Convert the masked image to grayscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to separate dark characters from the background
    _, threshold = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through the contours and extract individual boxes
    for i, contour in enumerate(contours):
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter out non-square boxes
        aspect_ratio = float(w) / h
        if aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
            # Crop the box from the original image
            box = orig[y:y+h, x:x+w]

            # Convert to gray and threshold to check for black characters
            gray_box = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
            _, thresh_box = cv2.threshold(gray_box, black_thresh, 255, cv2.THRESH_BINARY_INV)
            
            # If there are black pixels, save the box
            if np.any(thresh_box):
                cv2.imwrite(f'/Users/keenan/Documents/GDSC Handwritten/Semi-Colon/SemiColon Images/box_{i}.png', box)
                

# Call the function to extract boxes
extract_boxes(green_img, orig_img)