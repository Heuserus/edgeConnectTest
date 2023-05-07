import os
import random
from skimage.feature import canny
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from numpy import asarray
from skimage.color import rgb2gray, gray2rgb
import cv2

# Define the path to the directory containing the images
data_dir = '../datasets/hmaps_train_contrast/'

# Define the size of the combined image
combined_width = 5120
combined_height = 2560

# Select 100 random images from the directory
image_files = os.listdir(data_dir)
selected_files = random.sample(image_files, 100)

# Initialize the combined image
combined_image = np.zeros((combined_height, combined_width), dtype=np.uint8)

edge_images = []
# Loop over the selected files
for i, file_name in enumerate(selected_files):
    # Load the image and convert it to grayscale
    image_path = os.path.join(data_dir, file_name)
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

    # Convert the image to 8-bit and then to RGB
    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    


    edge_image = np.zeros_like(image,dtype=np.uint8)
    # Loop over the canny filter iterations
    for j in range(20):
        # Apply the canny filter
        edge = canny(image_8bit, sigma=j)
        edge_image = edge_image + edge

    edge_images.append(edge_image)

for i, file in enumerate(selected_files):

    # Load the input image
    input_image = cv2.imread(os.path.join(data_dir, file), cv2.IMREAD_GRAYSCALE)

    # Resize the input image to 256x256
    input_image = cv2.resize(input_image, (256, 256))

    # Stack the contour image and the resized input image horizontally
    combined = np.hstack((edge_images[i], input_image))

    # Calculate the row and column indices of the current image
    row_idx = i // 10
    col_idx = i % 10

    # Insert the combined image into the empty numpy array
    combined_image[row_idx*256:(row_idx+1)*256, col_idx*512:(col_idx+1)*512] = combined

# Save the combined image as a PNG file with OpenCV
cv2.imwrite("combined_image.png", combined_image)