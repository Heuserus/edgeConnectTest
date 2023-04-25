import numpy as np
import cv2
from skimage import measure
import math
import os
import random

# Set the path to the folder containing the input images
folder_path = "datasets/hmaps_train"

# Get a list of all the files in the folder
files = os.listdir(folder_path)

# Select 100 random files from the list
random_files = random.sample(files, 100)

# Create an empty list to store the contour images
contour_images = []

# Loop through the selected files
for file in random_files:

    # Load the grayscale heightmap as a 16-bit image file
    heightmap = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_ANYDEPTH)

    # Calculate the contour interval based on the desired height difference
    contour_image = np.zeros_like(heightmap,dtype=np.uint8)
    for contour_interval in range(0,50,1):
        contours = measure.find_contours(heightmap, contour_interval**2)
        for contour in contours:
            contour = np.around(contour).astype(np.int32)
            contour_image[contour[:, 0], contour[:, 1]] = 255

    # Store the contour image in the list
    contour_images.append(contour_image)

# Create an empty numpy array to store the combined image
combined_image = np.zeros((2560, 5120), dtype=np.uint8)

# Loop through the contour images and their corresponding input images
for i, file in enumerate(random_files):

    # Load the input image
    input_image = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)

    # Resize the input image to 256x256
    input_image = cv2.resize(input_image, (256, 256))

    # Stack the contour image and the resized input image horizontally
    combined = np.hstack((contour_images[i], input_image))

    # Calculate the row and column indices of the current image
    row_idx = i // 10
    col_idx = i % 10

    # Insert the combined image into the empty numpy array
    combined_image[row_idx*256:(row_idx+1)*256, col_idx*512:(col_idx+1)*512] = combined

# Save the combined image as a PNG file with OpenCV
cv2.imwrite("combined_image.png", combined_image)