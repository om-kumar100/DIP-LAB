
import cv2
import numpy as np
import matplotlib.pyplot as plt

print("Welcome to Smart Document Scanner & Quality Analysis System")

# ==============================
# Task 2: Image Acquisition
# ==============================

# Load image (change path)
image = cv2.imread("document.jpg")

# Resize to 512x512
image = cv2.resize(image, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ==============================
# Task 3: Sampling
# ==============================

# Downsample
img_256 = cv2.resize(gray, (256, 256))
img_128 = cv2.resize(gray, (128, 128))

# Upscale back to 512 for comparison
img_256_up = cv2.resize(img_256, (512, 512))
img_128_up = cv2.resize(img_128, (512, 512))

# ==============================
# Task 4: Quantization
# ==============================

def quantize(img, levels):
    return np.floor(img / (256 / levels)) * (256 / levels)

quant_256 = quantize(gray, 256)
quant_16 = quantize(gray, 16)
quant_4 = quantize(gray, 4)

# Convert to uint8
quant_256 = quant_256.astype(np.uint8)
quant_16 = quant_16.astype(np.uint8)
quant_4 = quant_4.astype(np.uint8)

# ==============================
# Task 5: Display Results
# ==============================

titles = [
    "Original", "Grayscale",
    "512x512", "256x256", "128x128",
    "8-bit", "4-bit", "2-bit"
]

images = [
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB), gray,
    gray, img_256_up, img_128_up,
    quant_256, quant_16, quant_4
]

plt.figure(figsize=(12, 8))

for i in range(len(images)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# Save outputs
cv2.imwrite("outputs/original.jpg", image)
cv2.imwrite("outputs/gray.jpg", gray)
cv2.imwrite("outputs/256.jpg", img_256_up)
cv2.imwrite("outputs/128.jpg", img_128_up)
cv2.imwrite("outputs/quant_8bit.jpg", quant_256)
cv2.imwrite("outputs/quant_4bit.jpg", quant_16)
cv2.imwrite("outputs/quant_2bit.jpg", quant_4)

print("Processing Completed! Check outputs folder.")