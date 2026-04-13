# ==========================================
# Name: Your Name
# Roll No: Your Roll No
# Course: Image Processing & Computer Vision
# Assignment: Medical Image Compression & Segmentation
# Date:
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("Medical Image Compression & Segmentation System Started")

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# ==============================
# Task 1: Load Image
# ==============================

image = cv2.imread("medical.jpg", 0)

if image is None:
    print("Error: Image not found!")
    exit()

# ==============================
# Task 1: RLE Compression
# ==============================

def rle_encode(img):
    pixels = img.flatten()
    encoded = []
    
    prev = pixels[0]
    count = 1
    
    for pixel in pixels[1:]:
        if pixel == prev:
            count += 1
        else:
            encoded.append((prev, count))
            prev = pixel
            count = 1
    
    encoded.append((prev, count))
    return encoded

encoded = rle_encode(image)

# Compression stats
original_size = image.size
compressed_size = len(encoded)

compression_ratio = original_size / compressed_size
storage_saving = (1 - compressed_size/original_size) * 100

# ==============================
# Task 2: Segmentation
# ==============================

# Global Threshold
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Otsu Threshold
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ==============================
# Task 3: Morphology
# ==============================

kernel = np.ones((3,3), np.uint8)

dilation = cv2.dilate(otsu_thresh, kernel, iterations=1)
erosion = cv2.erode(otsu_thresh, kernel, iterations=1)

# ==============================
# Display Results
# ==============================

titles = [
    "Original",
    "Global Threshold",
    "Otsu Threshold",
    "Dilation",
    "Erosion"
]

images = [
    image,
    global_thresh,
    otsu_thresh,
    dilation,
    erosion
]

plt.figure(figsize=(10,6))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# Save outputs
cv2.imwrite("outputs/original.jpg", image)
cv2.imwrite("outputs/global.jpg", global_thresh)
cv2.imwrite("outputs/otsu.jpg", otsu_thresh)
cv2.imwrite("outputs/dilation.jpg", dilation)
cv2.imwrite("outputs/erosion.jpg", erosion)

# ==============================
# Print Results
# ==============================

print("\n--- Compression Results ---")
print(f"Original Size: {original_size}")
print(f"Compressed Size: {compressed_size}")
print(f"Compression Ratio: {compression_ratio:.2f}")
print(f"Storage Saving: {storage_saving:.2f}%")