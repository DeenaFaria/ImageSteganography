import cv2
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(original_image_path, stego_image_path):
    # Read the original and stego images
    original_img = cv2.imread(original_image_path)
    stego_img = cv2.imread(stego_image_path)

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    stego_gray = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM between the images
    ssim_index, _ = ssim(original_gray, stego_gray, full=True)

    return ssim_index

# Paths to the original and stego images
original_image_path = 'Bird.png'  # Replace with your original image path
stego_image_path = 'StegoBird.png'  # Replace with your stego image path

# Calculate SSIM
ssim_value = calculate_ssim(original_image_path, stego_image_path)
print(f"SSIM between original and stego images: {ssim_value}")
