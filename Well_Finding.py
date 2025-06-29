import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, feature, exposure, measure
import cv2

def remove_large_objects(ar, max_size, connectivity=1):
    """Remove connected components larger than max_size."""
    out = np.copy(ar)
    ccs = measure.label(ar, connectivity=connectivity)
    component_sizes = np.bincount(ccs.ravel())
    too_large = component_sizes > max_size
    too_large_mask = too_large[ccs]
    out[too_large_mask] = 0
    return out

def preprocess_with_skimage_no_mask(image_path):
    """
    Preprocess the calibration image using scikit-image, without any masking.
    """
    # Create figure for all plots
    plt.figure(figsize=(20, 10))
    current_plot = 1
    total_plots = 7  # Total number of processing steps
    
    # Load image
    print("Loading image...")
    image = io.imread(image_path)
    # Keep image in RGB format
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV functions
    
    plt.subplot(2, 4, current_plot)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    current_plot += 1
    plt.draw()
    plt.pause(0.1)

    # 2. Denoising using Gaussian filter
    print("Applying Gaussian blur...")
    image_denoised = cv2.GaussianBlur(image, (11,11), 0)
    plt.subplot(2, 4, current_plot)
    plt.imshow(cv2.cvtColor(image_denoised, cv2.COLOR_BGR2RGB))
    plt.title('After Gaussian Blur')
    plt.axis('off')
    current_plot += 1
    plt.draw()
    plt.pause(0.1)

    # Apply mean shift filtering
    print("Applying mean shift filtering...")
    image_ms = cv2.pyrMeanShiftFiltering(image_denoised, 30, 50, 4)  # spatial radius, color radius, max iterations
    plt.subplot(2, 4, current_plot)
    plt.imshow(cv2.cvtColor(image_ms, cv2.COLOR_BGR2RGB))
    plt.title('After Mean Shift Filtering')
    plt.axis('off')
    current_plot += 1
    plt.draw()
    plt.pause(0.1)

    # 1. Contrast enhancement
    print("Applying contrast enhancement...")
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(image_ms, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl,a,b))
    image_rescale = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    plt.subplot(2, 4, current_plot)
    plt.imshow(cv2.cvtColor(image_rescale, cv2.COLOR_BGR2RGB))
    plt.title('After Contrast Enhancement')
    plt.axis('off')
    current_plot += 1
    plt.draw()
    plt.pause(0.1)

    # 3. Convert to grayscale for thresholding and corner detection
    print("Converting to grayscale and thresholding...")
    gray = cv2.cvtColor(image_rescale, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 4, current_plot)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    current_plot += 1
    plt.draw()
    plt.pause(0.1)

    # Local adaptive thresholding
    block_size = 131
    threshold_value = 0.8  # Threshold value between 0 and 1
    local_thresh = filters.threshold_local(gray, block_size=block_size)
    image_binary = (gray > local_thresh * threshold_value).astype(np.uint8) * 255
    plt.subplot(2, 4, current_plot)
    plt.imshow(image_binary, cmap='gray')
    plt.title(f'Binary Image (block={block_size}, thresh={threshold_value:.2f})')
    plt.axis('off')
    current_plot += 1
    plt.draw()
    plt.pause(0.1)

    # 4. Corner detection using Harris
    print("Detecting corners...")
    coords_resp = feature.corner_harris(image_binary, method='k', k=0.1, sigma=12.0, eps=1e-7)
    thresh = 0.35
    mask_harris = coords_resp > thresh
    mask_harris = remove_large_objects(mask_harris, 550)
    coords_peak = feature.corner_peaks(mask_harris * coords_resp,
                                       min_distance=50,
                                       threshold_rel=0.02)

    # Show corners on the denoised color image
    corners_img = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2RGB).copy()
    for y, x in coords_peak:
        cv2.circle(corners_img, (int(x), int(y)), 5, (255, 0, 0), -1)
    plt.subplot(2, 4, current_plot)
    plt.imshow(corners_img)
    plt.title(f'Detected Corners ({len(coords_peak)} points)')
    plt.axis('off')
    plt.draw()
    plt.pause(0.1)

    plt.tight_layout()
    plt.show()

    return image_binary

def main(image_path):
    preprocess_with_skimage_no_mask(image_path)

if __name__ == "__main__":
    image_path = r"C:\Users\NoahB\Documents\HebrewU Bioengineering\Equipment\Camera\Full_Plate_Edited.jpg"
    main(image_path) 