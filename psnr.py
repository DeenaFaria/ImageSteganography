import cv2
import os
import numpy as np

def calculate_psnr(original_image_path, stego_image_path):
    # Read images
    original_img = cv2.imread(original_image_path)
    stego_img = cv2.imread(stego_image_path)

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    stego_gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)

    # Calculate PSNR
    mse = np.mean((original_gray - stego_gray) ** 2)
    if mse == 0:
        return float('inf')  # PSNR is infinite if images are identical
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def estimate_payload(original_image_path, stego_image_path):
    # Get file sizes in bytes
    original_size = os.path.getsize(original_image_path)
    stego_size = os.path.getsize(stego_image_path)

    # Calculate payload assuming 1 bit per pixel difference
    payload = (stego_size - original_size) * 8
    return payload

# Paths to the original and stego images
original_image_path = 'Bird.png'  # Replace with your original image path
stego_image_path = 'StegoBird.png'  # Replace with your stego image path

# Calculate PSNR
psnr_value = calculate_psnr(original_image_path, stego_image_path)
print(f"PSNR: {psnr_value} dB")

# Estimate Payload
payload_value = estimate_payload(original_image_path, stego_image_path)
print(f"Estimated Payload: {payload_value} bits")
