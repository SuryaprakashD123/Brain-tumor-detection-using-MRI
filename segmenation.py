
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

def ShowImage(title, img, ctype):
    plt.figure(figsize=(10, 10))
    if ctype == 'bgr':
        b, g, r = cv2.split(img)
        rgb_img = cv2.merge([r, g, b])
        plt.imshow(rgb_img)
    elif ctype == 'hsv':
        rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        plt.imshow(rgb)
    elif ctype == 'gray':
        plt.imshow(img, cmap='gray')
    elif ctype == 'rgb':
        plt.imshow(img)
    else:
        raise Exception("Unknown colour type")
    plt.axis('off')
    plt.title(title)
    plt.show()

# Specify the path to the image
image_path = r'C:\Users\prasa\OneDrive\Pictures\pro\Brain-tumor-segmentation\Figure-A-Axial-T1-MRI-with-contrast-shows-no-evidence-of-a-brain-tumor.png'

# Debugging: Print the file path
print(f"Checking for file at: {image_path}")

# Debugging: List contents of the directory
directory = os.path.dirname(image_path)
print(f"Contents of the directory {directory}:")
print(os.listdir(directory))

# Verify if the file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The file at path {image_path} does not exist.")

# Read the image
img = cv2.imread(image_path)
if img is None:
    raise IOError(f"Cannot open image file at path {image_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ShowImage('Brain MRI', gray, 'gray')

# Thresholding
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
ShowImage('Thresholding image', thresh, 'gray')

# Connected Components
ret, markers = cv2.connectedComponents(thresh)

# Get the area taken by each component. Ignore label 0 since this is the background.
marker_area = [np.sum(markers == m) for m in range(1, np.max(markers) + 1)]

# Get label of largest component by area
largest_component_label = np.argmax(marker_area) + 1

# Get pixels which correspond to the brain
brain_mask = markers == largest_component_label

# Display the segmented brain region
brain_out = img.copy()
brain_out[brain_mask == False] = (0, 0, 0)
ShowImage('Segmented Brain Region', brain_out, 'bgr')

# Thresholding for sure foreground and background
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

# Convert image to RGB for display
im1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ShowImage('Watershed segmented image', im1, 'rgb')

# Calculate tumor area
tumor_area_pixels = np.sum(markers == 1)
pixel_to_mm_conversion_factor = 0.25  # Assuming each pixel represents 0.25 mm (adjust as per image scale)
tumor_area_mm2 = tumor_area_pixels * pixel_to_mm_conversion_factor**2

# Determine if tumor is present based on area threshold
tumor_threshold_mm2 = 50  # Example threshold value in mm²
tumor_present = tumor_area_mm2 > tumor_threshold_mm2

# Define tumor types based on area (hypothetical criteria)
tumor_type = "Unknown"
if tumor_present:
    if tumor_area_mm2 <= 100:
        tumor_type = "Benign"
    elif 100 < tumor_area_mm2 <= 200:
        tumor_type = "Precancerous"
    elif tumor_area_mm2 > 200:
        tumor_type = "Malignant"

# Display the tumor area and whether a tumor is present
print("Tumor area:", tumor_area_mm2, "mm²")
if tumor_present:
    print("Tumor is present.")
    print("Tumor type:", tumor_type)
else:
    print("No tumor detected.")
