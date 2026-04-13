# ==========================================
# Name: om
# Roll No: 2301010355
# Course: Image Processing & Computer Vision
# Assignment: Intelligent Image Processing System
# Date:
# ==========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

print("Welcome to Intelligent Image Processing System")

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# ==============================
# Task 2: Load Image
# ==============================

image = cv2.imread("input.jpg")

if image is None:
    print("Error: Image not found!")
    exit()

image = cv2.resize(image, (512, 512))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==============================
# Task 3: Noise + Restoration
# ==============================

# Gaussian Noise
gaussian_noise = np.random.normal(0, 25, gray.shape)
noisy_gaussian = np.clip(gray + gaussian_noise, 0, 255).astype(np.uint8)

# Salt & Pepper Noise
sp_noisy = gray.copy()
prob = 0.02
salt = np.random.rand(*gray.shape) < prob
pepper = np.random.rand(*gray.shape) < prob
sp_noisy[salt] = 255
sp_noisy[pepper] = 0

# Filters
mean = cv2.blur(noisy_gaussian, (5,5))
median = cv2.medianBlur(sp_noisy, 5)
gaussian = cv2.GaussianBlur(noisy_gaussian, (5,5), 0)

# Contrast Enhancement
enhanced = cv2.equalizeHist(gray)

# ==============================
# Task 4: Segmentation
# ==============================

_, thresh_global = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
_, thresh_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3,3), np.uint8)
dilation = cv2.dilate(thresh_otsu, kernel, iterations=1)
erosion = cv2.erode(thresh_otsu, kernel, iterations=1)

# ==============================
# Task 5: Features
# ==============================

# Edges
sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
canny = cv2.Canny(gray, 100, 200)

# Contours
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = image.copy()

for cnt in contours:
    if cv2.contourArea(cnt) > 500:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w,y+h), (0,255,0), 2)

# ORB Features
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)
feature_img = cv2.drawKeypoints(image, kp, None, color=(0,255,0))

# ==============================
# Task 6: Metrics
# ==============================

def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return 100
    return 10 * np.log10((255**2)/m)

def calc_ssim(a, b):
    return ssim(a, b)

print("\n--- Metrics ---")
print("MSE:", mse(gray, enhanced))
print("PSNR:", psnr(gray, enhanced))
print("SSIM:", calc_ssim(gray, enhanced))

# ==============================
# Task 7: Visualization
# ==============================

titles = [
    "Original", "Noisy", "Restored",
    "Enhanced", "Segmented", "Features"
]

images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    noisy_gaussian,
    gaussian,
    enhanced,
    thresh_otsu,
    cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)
]

plt.figure(figsize=(12,8))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

print("System Completed Successfully!")
