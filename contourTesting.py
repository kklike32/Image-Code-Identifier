import cv2
import numpy as np

def get_reference_points(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Analyze the histogram to find a suitable threshold value
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    threshold_value = np.argmax(hist)

    # Apply the threshold to the image
    _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area in descending order and keep the largest ones
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

    reference_points = []

    for contour in contours:
        # Get the moments of the contour which can be used to compute the centroid or "center of mass" of the contour
        M = cv2.moments(contour)

        # Calculate x,y coordinate of centroid
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        reference_points.append((cX, cY))

    return reference_points, thresholded

def calculate_box_size(thresholded_img):
    rows, cols = thresholded_img.shape
    for i in range(rows):
        for j in range(cols):
            if thresholded_img[i, j] > 128:  # 0.5 in grayscale, assuming grayscale is from 0 to 255
                box_size = 0
                while thresholded_img[i, j + box_size] > 128:
                    box_size += 1
                return box_size
    return None

def calculate_number_of_boxes(ref_points, box_size):
    # Assuming the reference points are the corners of the rectangle, compute its width and height
    width = max(ref_points, key=lambda point: point[0])[0] - min(ref_points, key=lambda point: point[0])[0]
    height = max(ref_points, key=lambda point: point[1])[1] - min(ref_points, key=lambda point: point[1])[1]

    # Compute the number of boxes in each dimension
    boxes_width = width // box_size
    boxes_height = height // box_size

    return boxes_width * boxes_height


image_path = "handCode1.png"
ref_points, thresholded_img = get_reference_points(image_path)
box_size = calculate_box_size(thresholded_img)

if box_size is not None:
    num_boxes = calculate_number_of_boxes(ref_points, box_size)
    print(num_boxes)
else:
    print("Could not determine box size.")
