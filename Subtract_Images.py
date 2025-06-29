import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

def select_image_file(title):
    """Open file browser to select an image file"""
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Bring dialog to front
    
    # Define supported file types
    filetypes = [
        ('All Image Files', '*.jpg *.jpeg *.png *.tiff *.tif *.bmp'),
        ('JPEG Files', '*.jpg *.jpeg'),
        ('PNG Files', '*.png'),
        ('TIFF Files', '*.tiff *.tif'),
        ('All Files', '*.*')
    ]
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes,
        initialdir=os.getcwd()
    )
    
    root.destroy()
    
    if not file_path:
        print("No file selected. Exiting.")
        exit()
    
    return file_path

def get_processing_options():
    """Get user preferences for image processing options"""
    root = tk.Tk()
    root.title("Image Processing Options")
    root.geometry("400x350")
    root.attributes('-topmost', True)
    
    # Variables to store user choices
    apply_dilation = tk.BooleanVar(value=False)
    apply_island_filter = tk.BooleanVar(value=False)
    invert_mask = tk.BooleanVar(value=False)
    dilation_kernel_size = tk.IntVar(value=5)
    dilation_iterations = tk.IntVar(value=1)
    min_island_size = tk.IntVar(value=200)
    
    # Create GUI elements
    tk.Label(root, text="Optional Processing for Image 2", font=("Arial", 14, "bold")).pack(pady=10)
    
    # Mask inversion option
    mask_frame = tk.LabelFrame(root, text="Mask Options", padx=10, pady=5)
    mask_frame.pack(fill="x", padx=10, pady=5)
    
    tk.Checkbutton(mask_frame, text="Invert Mask (swap black/white)", variable=invert_mask).pack(anchor="w")
    tk.Label(mask_frame, text="• Checked: Black areas in mask become white (show through)", font=("Arial", 8)).pack(anchor="w", padx=20)
    tk.Label(mask_frame, text="• Unchecked: White areas in mask stay white (show through)", font=("Arial", 8)).pack(anchor="w", padx=20)
    
    # Dilation options
    dilation_frame = tk.LabelFrame(root, text="Morphological Dilation", padx=10, pady=5)
    dilation_frame.pack(fill="x", padx=10, pady=5)
    
    tk.Checkbutton(dilation_frame, text="Apply Dilation", variable=apply_dilation).pack(anchor="w")
    
    tk.Label(dilation_frame, text="Kernel Size:").pack(anchor="w")
    tk.Scale(dilation_frame, from_=3, to=15, orient="horizontal", variable=dilation_kernel_size).pack(fill="x")
    
    tk.Label(dilation_frame, text="Iterations:").pack(anchor="w")
    tk.Scale(dilation_frame, from_=1, to=5, orient="horizontal", variable=dilation_iterations).pack(fill="x")
    
    # Island filter options
    island_frame = tk.LabelFrame(root, text="Island Size Filter", padx=10, pady=5)
    island_frame.pack(fill="x", padx=10, pady=5)
    
    tk.Checkbutton(island_frame, text="Remove Small Black Objects", variable=apply_island_filter).pack(anchor="w")
    
    tk.Label(island_frame, text="Minimum Island Size (pixels):").pack(anchor="w")
    tk.Scale(island_frame, from_=50, to=1000, orient="horizontal", variable=min_island_size).pack(fill="x")
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    result = {}
    
    def on_ok():
        result['apply_dilation'] = apply_dilation.get()
        result['apply_island_filter'] = apply_island_filter.get()
        result['invert_mask'] = invert_mask.get()
        result['dilation_kernel_size'] = dilation_kernel_size.get()
        result['dilation_iterations'] = dilation_iterations.get()
        result['min_island_size'] = min_island_size.get()
        root.quit()
        root.destroy()
    
    def on_skip():
        result['apply_dilation'] = False
        result['apply_island_filter'] = False
        result['invert_mask'] = False
        result['dilation_kernel_size'] = 5
        result['dilation_iterations'] = 1
        result['min_island_size'] = 200
        root.quit()
        root.destroy()
    
    tk.Button(button_frame, text="Apply Processing", command=on_ok, bg="lightgreen").pack(side="left", padx=5)
    tk.Button(button_frame, text="Skip Processing", command=on_skip, bg="lightgray").pack(side="left", padx=5)
    
    root.mainloop()
    
    return result

def apply_morphological_dilation(img, kernel_size=5, iterations=1):
    """Apply morphological dilation to the image"""
    print(f"Applying morphological dilation (kernel size: {kernel_size}, iterations: {iterations})...")
    
    # Create structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply dilation
    if len(img.shape) == 3:
        # For color images, apply to each channel
        dilated = cv2.dilate(img, kernel, iterations=iterations)
    else:
        # For grayscale images
        dilated = cv2.dilate(img, kernel, iterations=iterations)
    
    return dilated

def remove_small_islands(img, min_area=200):
    """Remove small black objects (islands) by making them white"""
    print(f"Removing black objects smaller than {min_area} pixels (making them white)...")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Create binary mask - invert so black objects become white for analysis
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find connected components (now black objects are white in the binary mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create mask for black objects to remove (make white)
    removal_mask = np.zeros_like(gray)
    
    objects_removed = 0
    total_objects = num_labels - 1  # Subtract 1 for background
    
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:  # Small black objects to remove
            removal_mask[labels == i] = 255
            objects_removed += 1
    
    print(f"Removed {objects_removed} small black objects out of {total_objects} total black objects")
    
    # Apply mask to original image - make small black areas white
    if len(img.shape) == 3:
        # For color images
        result = img.copy()
        for c in range(img.shape[2]):
            # Where removal_mask is white (255), set the pixel to white (255)
            result[:, :, c] = np.where(removal_mask == 255, 255, img[:, :, c])
    else:
        # For grayscale images
        result = np.where(removal_mask == 255, 255, img)
    
    return result

def process_image_2(img, options):
    """Apply selected processing options to image 2"""
    processed_img = img.copy()
    processing_applied = []
    
    if options['apply_dilation']:
        processed_img = apply_morphological_dilation(
            processed_img, 
            options['dilation_kernel_size'], 
            options['dilation_iterations']
        )
        processing_applied.append(f"Dilation (kernel:{options['dilation_kernel_size']}, iter:{options['dilation_iterations']})")
    
    if options['apply_island_filter']:
        processed_img = remove_small_islands(processed_img, options['min_island_size'])
        processing_applied.append(f"Island filter (min size: {options['min_island_size']}px)")
    
    if options['invert_mask']:
        processing_applied.append("Mask inversion enabled")
    
    if processing_applied:
        print(f"Applied processing: {', '.join(processing_applied)}")
    else:
        print("No processing applied to Image 2")
    
    return processed_img, processing_applied

# --- Step 1: Get file paths from user via file browser ---
print("=== IMAGE MASK APPLICATION TOOL ===")
print("Supports: JPG, JPEG, PNG, TIFF, TIF formats")
print("Image 1 will be filtered using Image 2 as an inverted mask")
print("Only areas where the inverted mask is WHITE will be shown")
print("Output will be saved as PNG in the same directory as the first image.")
print()
print("Please select the images using the file browser dialogs...")

img1_path = select_image_file("Select Image 1 (Image to be FILTERED)")
img2_path = select_image_file("Select Image 2 (MASK - will be inverted)")

print(f"\nLoading images...")
print(f"Image 1: {img1_path}")
print(f"Image 2: {img2_path}")

# Load images with support for different formats
img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

if img1 is None:
    raise ValueError(f"Could not load image 1: {img1_path}")
if img2 is None:
    raise ValueError(f"Could not load image 2: {img2_path}")

print(f"Image 1 shape: {img1.shape}")
print(f"Image 2 shape: {img2.shape}")

# Ensure both images have the same size
if img1.shape != img2.shape:
    print(f"Warning: Image shapes don't match!")
    print(f"Image 1: {img1.shape}")
    print(f"Image 2: {img2.shape}")
    
    # Resize the second image to match the first
    print("Resizing second image to match the first...")
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    print(f"Resized Image 2 shape: {img2.shape}")

# --- Step 2: Get processing options for Image 2 ---
print("\nConfiguring image processing options...")
processing_options = get_processing_options()

# --- Step 3: Apply processing to Image 2 if requested ---
img2_processed, processing_steps = process_image_2(img2, processing_options)

# Handle different channel configurations
def prepare_for_display(img):
    """Convert image to RGB format for display"""
    if len(img.shape) == 2:  # Grayscale
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:  # BGR
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:  # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        return img

# --- Step 4: Convert to RGB for visualization ---
img1_rgb = prepare_for_display(img1)
img2_rgb = prepare_for_display(img2)
img2_processed_rgb = prepare_for_display(img2_processed)

# --- Step 5: Apply inverted mask (AND operation) ---
print("\nApplying inverted mask...")

# Convert processed image 2 to grayscale for mask creation
if len(img2_processed.shape) == 3:
    mask_gray = cv2.cvtColor(img2_processed, cv2.COLOR_BGR2GRAY)
else:
    mask_gray = img2_processed.copy()

# Invert the mask (black becomes white, white becomes black)
if processing_options['invert_mask']:
    inverted_mask = cv2.bitwise_not(mask_gray)
    print("Mask inverted: Black areas will show through from Image 1")
else:
    inverted_mask = mask_gray.copy()
    print("Mask not inverted: White areas will show through from Image 1")

# Create 3-channel mask for color images
if len(img1.shape) == 3:
    inverted_mask_3ch = cv2.cvtColor(inverted_mask, cv2.COLOR_GRAY2BGR)
else:
    inverted_mask_3ch = inverted_mask

# Apply mask using AND operation
# Only areas where inverted mask is white (255) will show the original image
if len(img1.shape) == 3:
    masked_result = cv2.bitwise_and(img1, inverted_mask_3ch)
else:
    masked_result = cv2.bitwise_and(img1, inverted_mask)

# Calculate statistics
total_pixels = mask_gray.size
white_pixels_original = np.count_nonzero(mask_gray)
white_pixels_inverted = np.count_nonzero(inverted_mask)
visible_pixels = np.count_nonzero(cv2.cvtColor(masked_result, cv2.COLOR_BGR2GRAY) if len(masked_result.shape) == 3 else masked_result)

print(f"Original mask white pixels: {white_pixels_original:,} ({(white_pixels_original/total_pixels)*100:.1f}%)")
print(f"Inverted mask white pixels: {white_pixels_inverted:,} ({(white_pixels_inverted/total_pixels)*100:.1f}%)")
print(f"Visible pixels in result: {visible_pixels:,} ({(visible_pixels/total_pixels)*100:.1f}%)")

# --- Step 6: Display the result ---
# Determine number of subplots based on whether processing was applied
if processing_steps:
    plt.figure(figsize=(20, 12))
    
    # Top row: Original images and processing
    plt.subplot(3, 3, 1)
    plt.title("Image 1 (To be filtered)")
    plt.imshow(img1_rgb)
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.title("Image 2 (Original Mask)")
    plt.imshow(img2_rgb)
    plt.axis('off')
    
    plt.subplot(3, 3, 3)
    plt.title(f"Image 2 (Processed)\n{', '.join(processing_steps)}")
    plt.imshow(img2_processed_rgb)
    plt.axis('off')
    
    # Middle row: Mask operations
    plt.subplot(3, 3, 4)
    plt.title("Processed Mask (Grayscale)")
    plt.imshow(mask_gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    mask_status = "Inverted" if processing_options['invert_mask'] else "Original"
    show_areas = "Black" if processing_options['invert_mask'] else "White"
    plt.title(f"{mask_status} Mask\n({show_areas} areas will be kept)")
    plt.imshow(inverted_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(3, 3, 6)
    plt.title("Final Masked Result")
    masked_result_rgb = prepare_for_display(masked_result)
    plt.imshow(masked_result_rgb)
    plt.axis('off')
    
    # Bottom row: Comparisons
    plt.subplot(3, 3, 7)
    # Original mask application without processing
    if len(img2.shape) == 3:
        original_mask_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        original_mask_gray = img2.copy()
    original_inverted = cv2.bitwise_not(original_mask_gray)
    if len(img1.shape) == 3:
        original_inverted_3ch = cv2.cvtColor(original_inverted, cv2.COLOR_GRAY2BGR)
        original_masked = cv2.bitwise_and(img1, original_inverted_3ch)
    else:
        original_masked = cv2.bitwise_and(img1, original_inverted)
    original_masked_rgb = prepare_for_display(original_masked)
    plt.title("Result without processing")
    plt.imshow(original_masked_rgb)
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    processing_diff = cv2.absdiff(img2_rgb, img2_processed_rgb)
    plt.title("Processing Effect\n(Original vs Processed)")
    plt.imshow(processing_diff)
    plt.axis('off')
    
    plt.subplot(3, 3, 9)
    mask_comparison = cv2.absdiff(original_masked_rgb, masked_result_rgb)
    plt.title("Mask Effect Comparison\n(Original vs Processed)")
    plt.imshow(mask_comparison)
    plt.axis('off')
    
else:
    plt.figure(figsize=(20, 8))
    
    plt.subplot(2, 3, 1)
    plt.title("Image 1 (To be filtered)")
    plt.imshow(img1_rgb)
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("Image 2 (Mask)")
    plt.imshow(img2_rgb)
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    mask_status = "Inverted" if processing_options['invert_mask'] else "Original"
    show_areas = "Black" if processing_options['invert_mask'] else "White"
    plt.title(f"{mask_status} Mask\n({show_areas} areas will be kept)")
    plt.imshow(inverted_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title("Final Masked Result")
    masked_result_rgb = prepare_for_display(masked_result)
    plt.imshow(masked_result_rgb)
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title("Original Mask (Grayscale)")
    plt.imshow(mask_gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.title("Mask Statistics")
    plt.text(0.1, 0.7, f"Total pixels: {total_pixels:,}", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Original white: {white_pixels_original:,} ({(white_pixels_original/total_pixels)*100:.1f}%)", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Inverted white: {white_pixels_inverted:,} ({(white_pixels_inverted/total_pixels)*100:.1f}%)", fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Visible result: {visible_pixels:,} ({(visible_pixels/total_pixels)*100:.1f}%)", fontsize=12, transform=plt.gca().transAxes)
    plt.axis('off')

plt.tight_layout()
plt.show()

# --- Step 7: Save the result ---
print(f"\nPreparing to save result...")

# Generate default filename
img1_path_obj = Path(img1_path)
img1_name = img1_path_obj.stem
img2_name = Path(img2_path).stem

# Include processing info in filename
if processing_steps:
    processing_suffix = "_processed"
    for step in processing_steps:
        if "Dilation" in step:
            processing_suffix += "_dil"
        if "Island" in step:
            processing_suffix += "_filt"
else:
    processing_suffix = ""

default_filename = f"{img1_name}_MASKED_BY_{img2_name}{processing_suffix}_result.png"

# Let user choose save location
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

output_path = filedialog.asksaveasfilename(
    title="Save Masked Image As",
    defaultextension=".png",
    filetypes=[
        ('PNG Files', '*.png'),
        ('JPEG Files', '*.jpg'),
        ('All Files', '*.*')
    ],
    initialfile=default_filename,
    initialdir=str(img1_path_obj.parent)
)

root.destroy()

if not output_path:
    print("❌ No save location selected. Image not saved.")
    print("Processing complete - image was displayed but not saved.")
    exit()

print(f"Selected save location: {output_path}")

# Verify directory exists and is writable
output_path_obj = Path(output_path)
output_dir = output_path_obj.parent

if not output_dir.exists():
    print(f"❌ Directory does not exist: {output_dir}")
    exit()

if not os.access(output_dir, os.W_OK):
    print(f"❌ No write permission for directory: {output_dir}")
    exit()

# Convert result to BGR for OpenCV saving (if it's RGB)
print("Converting image format for saving...")
if len(masked_result.shape) == 3 and masked_result.shape[2] == 3:
    # Assume it's already in BGR format from original processing
    result_bgr = masked_result
else:
    result_bgr = masked_result

# Attempt to save with detailed error reporting
print(f"Saving image...")
try:
    success = cv2.imwrite(str(output_path), result_bgr)
    
    if success:
        # Verify file was actually created and has size > 0
        if output_path_obj.exists() and output_path_obj.stat().st_size > 0:
            file_size_mb = output_path_obj.stat().st_size / (1024 * 1024)
            print(f"✅ Successfully saved masked image!")
            print(f"📁 Location: {output_path}")
            print(f"📊 File size: {file_size_mb:.2f} MB")
            
            # Try to open the folder containing the file
            try:
                os.startfile(output_dir)  # Windows
                print(f"📂 Opened folder: {output_dir}")
            except:
                try:
                    os.system(f'open "{output_dir}"')  # macOS
                except:
                    try:
                        os.system(f'xdg-open "{output_dir}"')  # Linux
                    except:
                        print(f"💡 Manual navigation needed to: {output_dir}")
        else:
            print(f"❌ File was not created or is empty: {output_path}")
    else:
        print(f"❌ OpenCV failed to save image to: {output_path}")
        
except Exception as e:
    print(f"❌ Error occurred while saving: {str(e)}")

print(f"\n=== MASK APPLICATION COMPLETE ===")
print(f"Source image: {Path(img1_path).name}")
print(f"Mask image: {Path(img2_path).name}")
if processing_steps:
    print(f"Processing applied: {', '.join(processing_steps)}")
else:
    print(f"Processing applied: None")
mask_operation = "Inverted mask AND filtering" if processing_options['invert_mask'] else "Direct mask AND filtering"
print(f"Operation: {mask_operation}")
show_areas = "black" if processing_options['invert_mask'] else "white"
print(f"Areas shown: Where mask is {show_areas}")
print(f"Visible pixels: {visible_pixels:,} ({(visible_pixels/total_pixels)*100:.1f}% of total)")
if 'output_path' in locals() and output_path:
    print(f"Output: {Path(output_path).name}")
    print(f"Location: {Path(output_path).parent}")
else:
    print(f"Output: Not saved")