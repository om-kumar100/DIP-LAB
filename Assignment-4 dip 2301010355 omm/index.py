# ==========================================
# Name: om
# Roll No: 2301010355
# Course: Image Processing & Computer Vision
# Assignment: Traffic Monitoring System
# Date: 13-04-2026
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("Traffic Monitoring System Started")

# Create output folder
os.makedirs("outputs", exist_ok=True)

# ==============================
# Load Image
# ==============================

image = cv2.imread("traffic.jpg")

if image is None:
    print("Error: Image not found!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==============================
# Task 1: Edge Detection
# ==============================

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)

# Canny
canny = cv2.Canny(gray, 100, 200)

# ==============================
# Task 2: Contours
# ==============================

contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = image.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area > 500:  # filter small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w,y+h), (0,255,0), 2)

        print(f"Object -> Area: {area:.2f}, Perimeter: {perimeter:.2f}")

# ==============================
# Task 3: Feature Extraction
# ==============================

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)

feature_img = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0))

# ==============================
# Display Results
# ==============================

titles = [
    "Original", "Sobel Edge", "Canny Edge",
    "Contours", "ORB Features"
]

images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    sobel,
    canny,
    cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB),
    cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(10,8))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# Save outputs
cv2.imwrite("outputs/sobel.jpg", sobel)
cv2.imwrite("outputs/canny.jpg", canny)
cv2.imwrite("outputs/contours.jpg", contour_img)
cv2.imwrite("outputs/features.jpg", feature_img)

print("Processing Completed!")
