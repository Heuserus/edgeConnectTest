import os
import random

# Set the path to the directory containing the files
directory = "../datasets/hmaps_train"

# Get a list of all the files in the directory
files = os.listdir(directory)

# Shuffle the list of files
random.shuffle(files)

# Select the first 100 files from the shuffled list
selected_files = files[:100]

# Write the file names to a text file
with open("random_files.txt", "w") as f:
    for file in selected_files:
        f.write(file + "\n")