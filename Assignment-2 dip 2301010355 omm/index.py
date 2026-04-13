
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("Image Restoration System Started")

# Create output folder
os.makedirs("outputs", exist_ok=True)

# ==============================
# Task 1: Load Image
# ==============================

image = cv2.imread("image.jpg")

if image is None:
    print("Error: Image not found!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==============================
# Task 2: Noise Modeling
# ==============================

# Gaussian Noise
gaussian_noise = np.random.normal(0, 25, gray.shape)
gaussian_noisy = gray + gaussian_noise
gaussian_noisy = np.clip(gaussian_noisy, 0, 255).astype(np.uint8)

# Salt & Pepper Noise
sp_noisy = gray.copy()
prob = 0.02

# Salt
salt = np.random.rand(*gray.shape) < prob
sp_noisy[salt] = 255

# Pepper
pepper = np.random.rand(*gray.shape) < prob
sp_noisy[pepper] = 0

# ==============================
# Task 3: Filtering
# ==============================

# Mean Filter
mean_filtered = cv2.blur(gaussian_noisy, (5,5))

# Median Filter
median_filtered = cv2.medianBlur(sp_noisy, 5)

# Gaussian Filter
gaussian_filtered = cv2.GaussianBlur(gaussian_noisy, (5,5), 0)

# ==============================
# Task 4: Metrics
# ==============================

def mse(original, processed):
    return np.mean((original - processed) ** 2)

def psnr(original, processed):
    m = mse(original, processed)
    if m == 0:
        return 100
    return 10 * np.log10((255 ** 2) / m)

# Calculate metrics
mse_mean = mse(gray, mean_filtered)
psnr_mean = psnr(gray, mean_filtered)

mse_median = mse(gray, median_filtered)
psnr_median = psnr(gray, median_filtered)

mse_gaussian = mse(gray, gaussian_filtered)
psnr_gaussian = psnr(gray, gaussian_filtered)

# ==============================
# Task 5: Display
# ==============================

titles = [
    "Original", "Gaussian Noise", "Salt & Pepper",
    "Mean Filter", "Median Filter", "Gaussian Filter"
]

images = [
    gray, gaussian_noisy, sp_noisy,
    mean_filtered, median_filtered, gaussian_filtered
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
cv2.imwrite("outputs/original.jpg", gray)
cv2.imwrite("outputs/gaussian_noise.jpg", gaussian_noisy)
cv2.imwrite("outputs/sp_noise.jpg", sp_noisy)
cv2.imwrite("outputs/mean.jpg", mean_filtered)
cv2.imwrite("outputs/median.jpg", median_filtered)
cv2.imwrite("outputs/gaussian.jpg", gaussian_filtered)

# Print results
print("\n--- Performance ---")
print(f"Mean Filter -> MSE: {mse_mean:.2f}, PSNR: {psnr_mean:.2f}")
print(f"Median Filter -> MSE: {mse_median:.2f}, PSNR: {psnr_median:.2f}")
print(f"Gaussian Filter -> MSE: {mse_gaussian:.2f}, PSNR: {psnr_gaussian:.2f}")