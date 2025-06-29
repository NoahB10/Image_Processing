import cv2
import numpy as np
from tkinter import filedialog
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import csv
import glob

def auto_filter_from_circle(image_path, radius=45, tolerance=30, save_visualizations=True):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    print(f"Image loaded: {w}x{h}, center sample at {center}, radius {radius}")

    # Create circular mask for sampling
    y_indices, x_indices = np.ogrid[:h, :w]
    mask = (x_indices - center[0])**2 + (y_indices - center[1])**2 <= radius**2

    # Convert to HSV and sample pixels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sample_pixels = hsv[mask]

    if sample_pixels.size == 0:
        print("No pixels found in circular sample area.")
        return None

    # Compute mean and standard deviation of HSV values
    mean_hsv = np.mean(sample_pixels, axis=0)
    std_hsv = np.std(sample_pixels, axis=0)
    
    print(f"HSV Statistics:")
    print(f"  Mean HSV: {mean_hsv}")
    print(f"  Std Dev HSV: {std_hsv}")
    
    # Check for bimodal distribution (high standard deviation indicates potential bimodal)
    hue_std_threshold = 30  # Threshold for detecting bimodal hue distribution
    sat_std_threshold = 50  # Threshold for saturation
    val_std_threshold = 50  # Threshold for value
    
    is_bimodal_hue = std_hsv[0] > hue_std_threshold
    is_bimodal_sat = std_hsv[1] > sat_std_threshold  
    is_bimodal_val = std_hsv[2] > val_std_threshold
    
    print(f"Bimodal Analysis:")
    print(f"  Hue std dev: {std_hsv[0]:.1f} ({'BIMODAL' if is_bimodal_hue else 'normal'}, threshold: {hue_std_threshold})")
    print(f"  Saturation std dev: {std_hsv[1]:.1f} ({'BIMODAL' if is_bimodal_sat else 'normal'}, threshold: {sat_std_threshold})")
    print(f"  Value std dev: {std_hsv[2]:.1f} ({'BIMODAL' if is_bimodal_val else 'normal'}, threshold: {val_std_threshold})")
    
    # Handle bimodal distributions by splitting data
    if is_bimodal_hue or is_bimodal_sat or is_bimodal_val:
        print(f"\n🔄 BIMODAL DETECTION - Splitting sample data into two groups...")
        
        # Cluster the sample pixels into 2 groups
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(sample_pixels)
        
        # Split into two groups
        group1_pixels = sample_pixels[cluster_labels == 0]
        group2_pixels = sample_pixels[cluster_labels == 1]
        
        # Calculate means for each group
        mean1_hsv = np.mean(group1_pixels, axis=0)
        mean2_hsv = np.mean(group2_pixels, axis=0)
        
        print(f"Group 1: {len(group1_pixels)} pixels, Mean HSV: {mean1_hsv}")
        print(f"Group 2: {len(group2_pixels)} pixels, Mean HSV: {mean2_hsv}")
        
        # Create two separate color ranges
        lower1 = np.array([
            max(0, mean1_hsv[0] - tolerance),
            max(0, mean1_hsv[1] - tolerance),
            max(0, mean1_hsv[2] - tolerance)
        ])
        upper1 = np.array([
            min(179, mean1_hsv[0] + tolerance),
            min(255, mean1_hsv[1] + tolerance),
            min(255, mean1_hsv[2] + tolerance)
        ])
        
        lower2 = np.array([
            max(0, mean2_hsv[0] - tolerance),
            max(0, mean2_hsv[1] - tolerance),
            max(0, mean2_hsv[2] - tolerance)
        ])
        upper2 = np.array([
            min(179, mean2_hsv[0] + tolerance),
            min(255, mean2_hsv[1] + tolerance),
            min(255, mean2_hsv[2] + tolerance)
        ])
        
        print(f"Bimodal HSV ranges:")
        print(f"  Range 1 - Lower: {lower1}, Upper: {upper1}")
        print(f"  Range 2 - Lower: {lower2}, Upper: {upper2}")
        
        # Create combined mask from both ranges
        color_mask1 = cv2.inRange(hsv, lower1, upper1)
        color_mask2 = cv2.inRange(hsv, lower2, upper2)
        color_mask = cv2.bitwise_or(color_mask1, color_mask2)
        
        bimodal_processing = True
        
    else:
        print(f"\n✓ Normal distribution detected - using single range")
        
        # Define single HSV range with tolerance (original method)
        lower = np.array([
            max(0, mean_hsv[0] - tolerance),
            max(0, mean_hsv[1] - tolerance),
            max(0, mean_hsv[2] - tolerance)
        ])
        upper = np.array([
            min(179, mean_hsv[0] + tolerance),
            min(255, mean_hsv[1] + tolerance),
            min(255, mean_hsv[2] + tolerance)
        ])

        print(f"Single HSV range for filtering:\n  Lower: {lower}\n  Upper: {upper}")

        # Create initial mask to analyze color distribution
        color_mask = cv2.inRange(hsv, lower, upper)
        
        bimodal_processing = False
    
    # Filter out pure white pixels from analysis (they don't count as meaningful content)
    # Pure white: Saturation < 10 AND Value > 245
    white_pixel_mask = (hsv[:,:,1] < 10) & (hsv[:,:,2] > 245)
    non_white_pixels = np.sum(~white_pixel_mask)
    total_white_pixels = np.sum(white_pixel_mask)
    
    print(f"White pixel filtering:")
    print(f"  Total image pixels: {h * w:,}")
    print(f"  Pure white pixels: {total_white_pixels:,}")
    print(f"  Non-white pixels: {non_white_pixels:,}")
    
    # Analyze color distribution across entire image (excluding white pixels)
    color_mask_no_white = color_mask & (~white_pixel_mask)
    matching_pixels = np.sum(color_mask_no_white > 0)
    color_percentage = (matching_pixels / non_white_pixels) * 100 if non_white_pixels > 0 else 0
    min_percentage_threshold = 0.2  # 0.2% minimum
    
    print(f"\nColor Distribution Analysis (excluding white pixels):")
    print(f"  Non-white pixels in image: {non_white_pixels:,}")
    print(f"  Non-white pixels matching sampled colors: {matching_pixels:,}")
    print(f"  Percentage of non-white content: {color_percentage:.2f}%")
    print(f"  Minimum threshold: {min_percentage_threshold}%")
    
    # Only proceed if colors represent significant portion of image
    if color_percentage < min_percentage_threshold:
        print(f"✗ Color represents only {color_percentage:.2f}% of image (< {min_percentage_threshold}%)")
        print(f"  Sampled colors are too rare - filtering cancelled")
        
        if save_visualizations:
            # Show results with cancellation message
            cancelled_image = image.copy()
            cv2.putText(cancelled_image, f"FILTERING CANCELLED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(cancelled_image, f"Colors only {color_percentage:.2f}% of image", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(cancelled_image, f"Minimum required: {min_percentage_threshold}%", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Original", image)
            cv2.imshow("Filtering Cancelled", cancelled_image)
            cv2.imshow("Circular Sample Mask", mask.astype(np.uint8)*255)
            cv2.imshow("Color Distribution", color_mask)
            print("Press any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return None
    
    print(f"✓ Color distribution passed ({color_percentage:.2f}% ≥ {min_percentage_threshold}%)")
    print(f"  Proceeding with filtering...")
    
    # Create histogram comparison of all pixels vs selected pixels
    def create_pixel_histogram():
        if not save_visualizations:
            return
            
        print("\nCreating pixel distribution histograms (excluding white pixels)...")
        
        # Get all pixels in image and filter out white pixels
        all_pixels_hsv = hsv.reshape(-1, 3)
        white_pixel_mask_flat = white_pixel_mask.reshape(-1)
        
        # Filter out white pixels from all pixels
        all_pixels_no_white = all_pixels_hsv[~white_pixel_mask_flat]
        total_non_white_pixels = len(all_pixels_no_white)
        
        # Filter out white pixels from selected sample pixels
        # Check which sample pixels are white
        sample_white_mask = (sample_pixels[:,1] < 10) & (sample_pixels[:,2] > 245)
        sample_pixels_no_white = sample_pixels[~sample_white_mask]
        selected_pixels_count = len(sample_pixels_no_white)
        
        # Calculate percentage breakdown (non-white pixels only)
        selected_percentage = (selected_pixels_count / total_non_white_pixels) * 100 if total_non_white_pixels > 0 else 0
        
        print(f"  Total non-white pixels in image: {total_non_white_pixels:,}")
        print(f"  White pixels filtered out: {len(all_pixels_hsv) - total_non_white_pixels:,}")
        print(f"  Selected non-white sample pixels: {selected_pixels_count:,}")
        print(f"  Sample represents: {selected_percentage:.3f}% of non-white content")
        
        # Create subplot for HSV histograms
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        bimodal_text = "BIMODAL PROCESSING" if bimodal_processing else "Single Range"
        fig.suptitle(f'Pixel Distribution Analysis (White Pixels Excluded) - {bimodal_text}\nSample: {selected_pixels_count:,} pixels ({selected_percentage:.3f}% of {total_non_white_pixels:,} non-white)', 
                     fontsize=14, fontweight='bold')
        
        # H (Hue) channel
        axes[0,0].hist(all_pixels_no_white[:, 0], bins=180, alpha=0.7, color='lightblue', 
                      label=f'All non-white pixels ({total_non_white_pixels:,})', density=True)
        axes[0,0].hist(sample_pixels_no_white[:, 0], bins=180, alpha=0.8, color='red', 
                      label=f'Selected non-white pixels ({selected_pixels_count:,})', density=True)
        # Calculate std dev of non-white sample pixels for display
        sample_std_h = np.std(sample_pixels_no_white[:, 0]) if len(sample_pixels_no_white) > 0 else 0
        sample_std_s = np.std(sample_pixels_no_white[:, 1]) if len(sample_pixels_no_white) > 0 else 0
        sample_std_v = np.std(sample_pixels_no_white[:, 2]) if len(sample_pixels_no_white) > 0 else 0
        
        axes[0,0].set_title(f'Hue (H) Distribution (σ={sample_std_h:.1f})')
        axes[0,0].set_xlabel('Hue Value (0-179)')
        axes[0,0].set_ylabel('Density')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # S (Saturation) channel
        axes[0,1].hist(all_pixels_no_white[:, 1], bins=256, alpha=0.7, color='lightblue', 
                      label=f'All non-white pixels ({total_non_white_pixels:,})', density=True)
        axes[0,1].hist(sample_pixels_no_white[:, 1], bins=256, alpha=0.8, color='red', 
                      label=f'Selected non-white pixels ({selected_pixels_count:,})', density=True)
        axes[0,1].set_title(f'Saturation (S) Distribution (σ={sample_std_s:.1f})')
        axes[0,1].set_xlabel('Saturation Value (0-255)')
        axes[0,1].set_ylabel('Density')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # V (Value/Brightness) channel
        axes[1,0].hist(all_pixels_no_white[:, 2], bins=256, alpha=0.7, color='lightblue', 
                      label=f'All non-white pixels ({total_non_white_pixels:,})', density=True)
        axes[1,0].hist(sample_pixels_no_white[:, 2], bins=256, alpha=0.8, color='red', 
                      label=f'Selected non-white pixels ({selected_pixels_count:,})', density=True)
        axes[1,0].set_title(f'Value (V) Distribution (σ={sample_std_v:.1f})')
        axes[1,0].set_xlabel('Value/Brightness (0-255)')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Combined 3D scatter plot (sampled)
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        
        # Sample a subset for 3D plot (too many points slow down rendering)
        sample_size = min(5000, len(all_pixels_no_white))
        if len(all_pixels_no_white) > 0:
            idx_all = np.random.choice(len(all_pixels_no_white), sample_size, replace=False)
            sample_all = all_pixels_no_white[idx_all]
        else:
            sample_all = np.array([]).reshape(0, 3)
        
        sample_size_selected = min(1000, len(sample_pixels_no_white))
        if len(sample_pixels_no_white) > 0:
            idx_selected = np.random.choice(len(sample_pixels_no_white), sample_size_selected, replace=False)
            sample_selected = sample_pixels_no_white[idx_selected]
        else:
            sample_selected = np.array([]).reshape(0, 3)
        
        if len(sample_all) > 0:
            ax_3d.scatter(sample_all[:, 0], sample_all[:, 1], sample_all[:, 2], 
                         c='lightblue', alpha=0.3, s=1, label=f'All non-white pixels (sample)')
        if len(sample_selected) > 0:
            ax_3d.scatter(sample_selected[:, 0], sample_selected[:, 1], sample_selected[:, 2], 
                         c='red', alpha=0.8, s=3, label=f'Selected non-white pixels')
        
        ax_3d.set_xlabel('Hue')
        ax_3d.set_ylabel('Saturation')
        ax_3d.set_zlabel('Value')
        ax_3d.set_title('3D HSV Distribution')
        ax_3d.legend()
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"pixel_distribution_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"  Histogram saved as: {plot_filename}")
        
        plt.show()
        
        return selected_percentage
    
    # Create the histogram analysis
    if save_visualizations:
        sample_percentage = create_pixel_histogram()
    
    # Apply slight erosion to remove lone pixels
    erode_kernel = np.ones((3,3), np.uint8)
    eroded_mask = cv2.erode(color_mask, erode_kernel, iterations=1)
    print(f"Applied erosion to remove lone pixels")
    
    # Get all white pixels from eroded mask
    white_pixels = np.where(eroded_mask == 255)
    
    # Calculate minimum area threshold: π * 45² pixels
    min_area_threshold = int(np.pi * 45 * 45)  # ≈ 6,361 pixels
    
    # Initialize variables for bounding box
    x_min = y_min = x_max = y_max = None
    bounding_box_created = False
    
    if len(white_pixels[0]) > 0:
        y_coords = white_pixels[0]
        x_coords = white_pixels[1]
        total_pixels = len(y_coords)
        
        print(f"Found {total_pixels:,} white pixels after erosion")
        print(f"Minimum area threshold: {min_area_threshold:,} pixels (π * 45²)")
        
        # Only proceed with bounding box if we have enough area
        if total_pixels >= min_area_threshold:
            # Use percentiles to ignore extreme outlier points
            # Calculate percentile-based bounds to ignore extreme outliers
            y_min = int(np.percentile(y_coords, .7))   # Ignore bottom 0.7% of y coordinates
            y_max = int(np.percentile(y_coords, 99.3))  # Ignore top 0.7% of y coordinates
            x_min = int(np.percentile(x_coords, .7))   # Ignore leftmost 0.7% of x coordinates  
            x_max = int(np.percentile(x_coords, 99.3))  # Ignore rightmost 0.7% of x coordinates
            
            # Create boxed mask - fill the percentile-based bounding rectangle
            final_mask = np.zeros_like(eroded_mask)
            final_mask[y_min:y_max+1, x_min:x_max+1] = 255
            
            bounding_box_created = True
            print(f"✓ Area threshold met - creating percentile-based bounding box:")
            print(f"  Box coordinates: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            print(f"  Box dimensions: {x_max-x_min+1} x {y_max-y_min+1}")
        else:
            final_mask = eroded_mask
            print(f"✗ Insufficient area ({total_pixels:,} < {min_area_threshold:,}) - using eroded mask as-is")
    else:
        final_mask = eroded_mask
        print("No white pixels found, using eroded mask")
    
    # Apply final mask
    filtered = cv2.bitwise_and(image, image, mask=final_mask)
    
    # Create overlay showing bounding box border on original image
    if save_visualizations:
        overlay_image = image.copy()
        if bounding_box_created:
            # Draw bounding box border on original image
            cv2.rectangle(overlay_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            # Add corner markers for better visibility
            marker_size = 10
            cv2.line(overlay_image, (x_min, y_min), (x_min + marker_size, y_min), (0, 255, 0), 5)
            cv2.line(overlay_image, (x_min, y_min), (x_min, y_min + marker_size), (0, 255, 0), 5)
            cv2.line(overlay_image, (x_max, y_max), (x_max - marker_size, y_max), (0, 255, 0), 5)
            cv2.line(overlay_image, (x_max, y_max), (x_max, y_max - marker_size), (0, 255, 0), 5)
            
            # Add text showing both color distribution and area status
            processing_method = "BIMODAL" if bimodal_processing else "Single Range"
            cv2.putText(overlay_image, f"Processing: {processing_method}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(overlay_image, f"Color Distribution: {color_percentage:.2f}% (PASSED)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(overlay_image, f"Area: {len(white_pixels[0]):,} pixels (PASSED)", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Add text showing area status
            processing_method = "BIMODAL" if bimodal_processing else "Single Range"
            cv2.putText(overlay_image, f"Processing: {processing_method}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(overlay_image, f"Color Distribution: {color_percentage:.2f}% (PASSED)", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(white_pixels[0]) > 0:
                cv2.putText(overlay_image, f"Area: {len(white_pixels[0]):,} pixels (FAILED - too small)", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(overlay_image, f"Min area required: {min_area_threshold:,} pixels", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Show results
        cv2.imshow("Original", image)
        cv2.imshow("Bounding Box Overlay", overlay_image)
        cv2.imshow("Circular Sample Mask", mask.astype(np.uint8)*255)
        cv2.imshow("Initial Color Mask", color_mask)
        cv2.imshow("After Erosion", eroded_mask)
        cv2.imshow("Percentile Bounding Box", final_mask)
        cv2.imshow("Filtered Output", filtered)
        
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"filtered_result_{timestamp}.png", filtered)
        print(f"Filtered result saved as filtered_result_{timestamp}.png")
    
    # Return results for batch processing
    if bounding_box_created:
        # Calculate centroid
        centroid_x = (x_min + x_max) / 2
        centroid_y = (y_min + y_max) / 2
        
        # Calculate width and height
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        # Calculate area
        area = width * height
        
        return {
            'success': True,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'bbox_x1': x_min,
            'bbox_y1': y_min,
            'bbox_x2': x_max,
            'bbox_y2': y_max,
            'width': width,
            'height': height,
            'area': area,
            'color_percentage': color_percentage,
            'bimodal_processing': bimodal_processing,
            'total_pixels_after_erosion': len(white_pixels[0]) if len(white_pixels[0]) > 0 else 0
        }
    else:
        return {
            'success': False,
            'reason': 'insufficient_area' if len(white_pixels[0]) > 0 else 'no_pixels_found',
            'color_percentage': color_percentage,
            'bimodal_processing': bimodal_processing,
            'total_pixels_after_erosion': len(white_pixels[0]) if len(white_pixels[0]) > 0 else 0
        }

def batch_process_folder(folder_path, radius=45, tolerance=30, output_csv_name="circle_dropper_results.csv"):
    """
    Process all images in a folder and save results to CSV
    """
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, extension)))
        image_files.extend(glob.glob(os.path.join(folder_path, extension.upper())))
    
    if not image_files:
        print(f"No image files found in folder: {folder_path}")
        return
    
    print(f"Found {len(image_files)} image files to process")
    
    # Prepare CSV output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"circle_dropper_batch_{timestamp}.csv"
    csv_path = os.path.join(folder_path, csv_filename)
    
    # CSV headers similar to the Well_objects.csv format
    headers = [
        'Object Number',
        'File Name',
        'Processing Status',
        'Area (px²)',
        'Centroid X',
        'Centroid Y',
        'Bounding Box X1',
        'Bounding Box Y1', 
        'Bounding Box X2',
        'Bounding Box Y2',
        'Bounding Box Width',
        'Bounding Box Height',
        'Color Percentage',
        'Processing Method',
        'Pixels After Erosion',
        'Failure Reason'
    ]
    
    results = []
    processed_count = 0
    successful_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        print(f"\n{'='*60}")
        print(f"Processing image {i}/{len(image_files)}: {filename}")
        print(f"{'='*60}")
        
        # Process image without showing visualizations
        result = auto_filter_from_circle(image_path, radius, tolerance, save_visualizations=False)
        
        if result is None:
            # Image couldn't be loaded or processed
            row = [
                i, filename, 'FAILED - Could not load image', '', '', '', '', '', '', '', '', '', '', '', '', 'image_load_error'
            ]
        elif result['success']:
            # Successful processing
            processing_method = 'BIMODAL' if result['bimodal_processing'] else 'Single Range'
            row = [
                i,
                filename,
                'SUCCESS',
                result['area'],
                f"{result['centroid_x']:.2f}",
                f"{result['centroid_y']:.2f}",
                result['bbox_x1'],
                result['bbox_y1'],
                result['bbox_x2'],
                result['bbox_y2'],
                result['width'],
                result['height'],
                f"{result['color_percentage']:.2f}",
                processing_method,
                result['total_pixels_after_erosion'],
                ''
            ]
            successful_count += 1
        else:
            # Failed processing
            processing_method = 'BIMODAL' if result['bimodal_processing'] else 'Single Range'
            row = [
                i,
                filename,
                'FAILED',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                f"{result['color_percentage']:.2f}",
                processing_method,
                result['total_pixels_after_erosion'],
                result['reason']
            ]
        
        results.append(row)
        processed_count += 1
        
        # Print summary for this image
        if result and result['success']:
            print(f"✓ SUCCESS: Bounding box created")
            print(f"  Centroid: ({result['centroid_x']:.1f}, {result['centroid_y']:.1f})")
            print(f"  Bounding box: ({result['bbox_x1']}, {result['bbox_y1']}) to ({result['bbox_x2']}, {result['bbox_y2']})")
            print(f"  Dimensions: {result['width']} x {result['height']} (Area: {result['area']:,} px²)")
        elif result:
            print(f"✗ FAILED: {result['reason']}")
            print(f"  Color percentage: {result['color_percentage']:.2f}%")
        else:
            print(f"✗ FAILED: Could not load or process image")
    
    # Write results to CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(results)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total images processed: {processed_count}")
    print(f"Successful detections: {successful_count}")
    print(f"Failed detections: {processed_count - successful_count}")
    print(f"Success rate: {(successful_count/processed_count)*100:.1f}%")
    print(f"\nResults saved to: {csv_path}")
    
    return csv_path

def main():
    """
    Main function that prompts user to choose between single image or batch processing
    """
    print("Circle Dropper Filter - Choose processing mode:")
    print("1. Single image")
    print("2. Batch process folder")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice == '1':
        # Single image processing
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("All Files", "*.*")]
        )
        if not image_path:
            print("No image selected. Exiting.")
            return
        
        auto_filter_from_circle(image_path)
        
    elif choice == '2':
        # Batch processing
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        if not folder_path:
            print("No folder selected. Exiting.")
            return
        
        # Get processing parameters
        try:
            radius = int(input("Enter sampling radius (default 45): ") or "45")
            tolerance = int(input("Enter color tolerance (default 30): ") or "30")
        except ValueError:
            print("Invalid input, using defaults: radius=45, tolerance=30")
            radius = 45
            tolerance = 30
        
        csv_path = batch_process_folder(folder_path, radius, tolerance)
        print(f"\nBatch processing complete! Results saved to:\n{csv_path}")
        
    else:
        print("Invalid choice. Please run again and select 1 or 2.")

if __name__ == "__main__":
    main()
