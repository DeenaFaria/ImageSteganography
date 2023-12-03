import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy, ttest_ind, chi2_contingency
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import chi2_contingency

def calculate_standard_deviation(image_path):
    img = cv2.imread(image_path, 0)  # Read the image in grayscale
    std_dev = np.std(img)
    return std_dev

def calculate_cosine_similarity(original_image_path, stego_image_path):
    original_img = cv2.imread(original_image_path, 0)
    stego_img = cv2.imread(stego_image_path, 0)
    
    # Reshape images to 1D arrays
    original_flat = original_img.flatten().reshape(1, -1)
    stego_flat = stego_img.flatten().reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(original_flat, stego_flat)
    return similarity[0][0]

def calculate_t_test(original_image_path, stego_image_path):
    original_img = cv2.imread(original_image_path, 0)
    stego_img = cv2.imread(stego_image_path, 0)
    
    _, p_value = ttest_ind(original_img.flatten(), stego_img.flatten())
    return p_value



def calculate_entropy(image_path):
    img = cv2.imread(image_path, 0)
    ent = entropy(img.ravel(), base=2)
    return np.sum(ent)

def calculate_chi_squared_test(original_image_path, stego_image_path):
    original_img = cv2.imread(original_image_path, 0)
    stego_img = cv2.imread(stego_image_path, 0)
    
    # Calculate histograms for the original and stego images
    hist_original = cv2.calcHist([original_img], [0], None, [256], [0, 256])
    hist_stego = cv2.calcHist([stego_img], [0], None, [256], [0, 256])
    
    # Flatten histograms and create the contingency table
    hist_original = hist_original.flatten().astype(int)
    hist_stego = hist_stego.flatten().astype(int)
    contingency_table = np.array([hist_original, hist_stego])
    
    # Perform the chi-squared test
    chi_stat, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value





# Paths to the original and stego images
original_image_path = 'Bird.png'  # Replace with your original image path
stego_image_path = 'StegoBird.png'  # Replace with your stego image path

# Calculate Standard Deviation
std_dev = calculate_standard_deviation(stego_image_path)
print(f"Standard Deviation: {std_dev}")

# Calculate Cosine Similarity
cos_similarity = calculate_cosine_similarity(original_image_path, stego_image_path)
print(f"Cosine Similarity: {cos_similarity}")

# Calculate T-test
t_test_p_value = calculate_t_test(original_image_path, stego_image_path)
print(f"T-test p-value: {t_test_p_value}")

# Calculate Chi-Squared Test
chi_test_p_value = calculate_chi_squared_test(original_image_path, stego_image_path)
print(f"Chi-Squared Test p-value: {chi_test_p_value}")

# Calculate Entropy
entropy_value = calculate_entropy(stego_image_path)
print(f"Entropy: {entropy_value}")
