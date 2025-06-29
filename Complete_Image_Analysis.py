#!/usr/bin/env python3
"""
Well Organoid Analyzer

This script combines dual detection (binary + color) with well cropping and individual well analysis.
It finds organoid centroids, breaks the image into individual wells, correlates centroids to wells,
detects well boundaries, and creates comprehensive visualizations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import os
import json
import csv
from datetime import datetime
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from pathlib import Path
import shutil

# Import functionality from existing scripts
from mask_cropper import MaskCropper
from Cirlce_Dropper_Filter import auto_filter_from_circle

class WellOrganoidAnalyzer:
    """Comprehensive analyzer for well-plate organoid detection and analysis"""
    
    def __init__(self):
        # Image data
        self.original_image = None
        self.image_path = None
        self.mask_path = None
        self.image_with_organoids = None
        self.image_with_organoids_path = None
        self.height = 0
        self.width = 0
        
        # Detection parameters
        self.sample_masks = []
        self.circle_radius = 50
        self.mouse_x = 0
        self.mouse_y = 0
        self.color_tolerance = 30
        
        # Results storage
        self.binary_centroids = []
        self.color_centroids = []
        self.well_crops = []
        self.well_boundary_results = {}
        
        # GUI state
        self.zoom_factor = 1.0
        self.zoom_center_x = 0
        self.zoom_center_y = 0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.display_img = None
        self.window_name = "Well Organoid Analyzer"
        
        # Color palette
        self.palette_file = "color_palette_save.json"
    
    def load_images(self):
        """Load main image and mask for processing"""
        print("=== WELL ORGANOID ANALYZER ===")
        print("Select the main image to analyze...")
        
        # Load main image
        root = Tk()
        root.withdraw()
        self.image_path = filedialog.askopenfilename(
            title="Select Main Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        
        if not self.image_path:
            print("No image selected.")
            return False
            
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            print(f"Error: Could not load image from {self.image_path}")
            return False
            
        self.height, self.width = self.original_image.shape[:2]
        self.zoom_center_x = self.width // 2
        self.zoom_center_y = self.height // 2
        
        print(f"Loaded main image: {self.width}x{self.height}")
        
        # Load mask
        print("Select the well mask image...")
        self.mask_path = filedialog.askopenfilename(
            title="Select Well Mask",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        
        if not self.mask_path:
            print("No mask selected.")
            return False
            
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error: Could not load mask from {self.mask_path}")
            return False
            
        print(f"Loaded mask: {mask.shape[1]}x{mask.shape[0]}")
        return True
    
    def merge_close_centroids(self, centroids, threshold=24):
        """Merge centroids that are close together to avoid duplicates"""
        if len(centroids) == 0:
            return []
        if len(centroids) == 1:
            return [tuple(map(int, centroids[0]))]
        
        centroids = np.array(centroids, dtype=float)
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        
        changed = True
        while changed:
            changed = False
            if len(centroids) < 2:
                break
                
            dists = distance_matrix(centroids, centroids)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)

            if min_dist < threshold:
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                merged = (centroids[i] + centroids[j]) / 2
                centroids = np.delete(centroids, [i, j], axis=0)
                centroids = np.vstack([centroids, merged])
                changed = True

        return [tuple(map(int, c)) for c in centroids]
    
    def run_dual_detection(self, min_diameter=15, max_diameter=50, circularity_threshold=0.75):
        """Run both binary and color detection to find organoid centroids"""
        print("\n--- DUAL DETECTION PHASE ---")
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # === BINARY DETECTION PROCESSING ===
        print("\n--- BINARY DETECTION PHASE ---")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Create mask with threshold 122 (inverted for inpainting)
        dark_mask = (gray < 122).astype(np.uint8) * 255
        
        # Apply inpainting
        inpainted = cv2.inpaint(self.original_image, dark_mask, inpaintRadius=20, flags=cv2.INPAINT_TELEA)
        
        # Convert inpainted image to grayscale for binarization
        inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        
        # Apply binarization with threshold 163
        _, binary_plate = cv2.threshold(inpainted_gray, 163, 255, cv2.THRESH_BINARY)
        
        # Find centroids from binary detection
        binary_centroids = self.find_binary_centroids(binary_plate, min_diameter, max_diameter, circularity_threshold)
        self.binary_centroids = binary_centroids
        
        print(f"Binary detection found {len(binary_centroids)} centroids")
        
        # === COLOR DETECTION PROCESSING ===
        print("\n--- COLOR DETECTION PHASE ---")
        
        # Check for saved palette first
        sample_pixels = None
        if self.ask_use_saved_palette():
            sample_pixels = self.load_color_palette()
        
        if sample_pixels is None:
            # Interactive sample selection
            print("Instructions for color sampling:")
            print("1. RIGHT CLICK to place circular sample areas")
            print("2. LEFT CLICK + DRAG to pan around the image")
            print("3. Mouse Wheel: Zoom in/out")
            print("4. [ ] keys: Decrease/Increase circle size")
            print("5. Press SPACE to process samples")
            print("6. Press ESC to skip color detection")
            
            # Setup window for color sampling
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, min(1200, self.width), min(800, self.height))
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
            # Reset samples
            self.sample_masks = []
            self.display_img = self.original_image.copy()
            
            # Sample selection loop
            color_detection_enabled = True
            while True:
                self.update_color_filter_display()
                display_to_show = self.get_zoomed_display_image()
                cv2.imshow(self.window_name, display_to_show)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # Escape - skip color detection
                    color_detection_enabled = False
                    break
                elif cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    color_detection_enabled = False
                    break
                elif key == ord(' ') and len(self.sample_masks) > 0:
                    # Process samples
                    break
                elif key == ord('r'):
                    self.sample_masks = []
                    print("Sample areas reset!")
                elif key == ord('['):
                    self.circle_radius = max(5, self.circle_radius - 5)
                    print(f"Circle radius decreased to {self.circle_radius}")
                elif key == ord(']'):
                    self.circle_radius = min(200, self.circle_radius + 5)
                    print(f"Circle radius increased to {self.circle_radius}")
                elif key == ord('x'):
                    self.zoom_factor = 1.0
                    self.zoom_center_x = self.width // 2
                    self.zoom_center_y = self.height // 2
                    print(f"Zoom reset: {self.zoom_factor:.2f}x")
            
            cv2.destroyAllWindows()
            
            if not color_detection_enabled or len(self.sample_masks) == 0:
                print("Color detection skipped or no samples selected.")
                color_centroids = []
                color_filtered_image = np.zeros_like(self.original_image)
            else:
                # Process color samples and get centroids
                color_centroids, color_filtered_image = self.process_color_samples_for_centroids()
        else:
            # Use saved palette
            print("Using saved color palette...")
            color_centroids, color_filtered_image = self.process_saved_palette_for_centroids(sample_pixels)
        
        self.color_centroids = color_centroids
        print(f"Color detection found {len(color_centroids)} centroids")
        
        # Create image with organoids plotted for well sectioning
        self.create_image_with_organoids()
        
        print(f"Total centroids found: {len(self.binary_centroids)} binary + {len(self.color_centroids)} color")

    def find_binary_centroids(self, image, min_diameter=15, max_diameter=50, circularity_threshold=0.75):
        """Find centroids from binary detection (inverted image)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Invert the image for binary detection
        gray = cv2.bitwise_not(gray)
        
        kernel = np.ones((3, 3), np.uint8)
        all_centroids = []

        for i in range(4):  # 0 to 3 erosions
            eroded = gray.copy() if i == 0 else cv2.erode(gray.copy(), kernel, iterations=i)
            contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            iteration_threshold = circularity_threshold * (1 - 0.05 * i)
            iter_min_diameter = min_diameter - 1

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area == 0:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                (_, _), radius = cv2.minEnclosingCircle(cnt)
                diameter = 2 * radius

                if circularity >= iteration_threshold and iter_min_diameter < diameter <= max_diameter:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        all_centroids.append((cx, cy))

        # Merge close centroids
        print(f"Binary detection - Before merging: {len(all_centroids)} centroids")
        merged_centroids = self.merge_close_centroids(all_centroids, threshold=24)
        print(f"Binary detection - After merging: {len(merged_centroids)} centroids")

        return merged_centroids

    def create_image_with_organoids(self):
        """Create a copy of the original image with organoids plotted on it"""
        print("Creating image with plotted organoids...")
        
        # Create a copy of the original image
        self.image_with_organoids = self.original_image.copy()
        
        # Plot binary centroids (red circles)
        for (cx, cy) in self.binary_centroids:
            cv2.circle(self.image_with_organoids, (cx, cy), 8, (0, 0, 255), 2)  # Red
            cv2.circle(self.image_with_organoids, (cx, cy), 3, (0, 0, 255), -1)  # Red center
        
        # Plot color centroids (green squares)
        for (cx, cy) in self.color_centroids:
            # Draw square by drawing rectangle
            cv2.rectangle(self.image_with_organoids, (cx-6, cy-6), (cx+6, cy+6), (0, 255, 0), 2)  # Green
            cv2.rectangle(self.image_with_organoids, (cx-2, cy-2), (cx+2, cy+2), (0, 255, 0), -1)  # Green center
        
        # Save the image with organoids
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_with_organoids_path = f"image_with_organoids_{timestamp}.png"
        cv2.imwrite(self.image_with_organoids_path, self.image_with_organoids)
        print(f"Saved image with organoids: {self.image_with_organoids_path}")
    
    def create_well_crops(self):
        """Use MaskCropper to break image into individual well images"""
        print("\n--- CREATING WELL CROPS ---")
        
        output_dir = "well_analysis_output"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        # Create MaskCropper instance - use image with organoids plotted and NO target_size
        cropper = MaskCropper(
            input_image_path=self.image_with_organoids_path,  # Use image with organoids
            mask_path=self.mask_path,
            output_folder=output_dir,
            target_size=None  # Keep original crop sizes - no white borders
        )
        
        # Get well regions
        regions, labels = cropper.find_connected_regions()
        print(f"Found {len(regions)} wells")
        
        # Store well information
        self.well_crops = []
        for region in regions:
            well_info = {
                'grid_label': region['grid_label'],
                'bbox': region['bbox'],  # (x, y, w, h)
                'centroid': region['centroid'],
                'area': region['area'],
                'organoid_centroids': []
            }
            self.well_crops.append(well_info)
        
        # Save the crops
        cropper.save_crops()
        return output_dir
    
    def correlate_centroids_to_wells(self):
        """Map organoid centroids to their respective wells"""
        print("\n--- CORRELATING CENTROIDS TO WELLS ---")
        
        all_centroids = []
        for centroid in self.binary_centroids:
            all_centroids.append((centroid, 'binary'))
        for centroid in self.color_centroids:
            all_centroids.append((centroid, 'color'))
        
        print(f"Correlating {len(all_centroids)} centroids to wells...")
        
        # Map each centroid to its well
        for centroid, detection_type in all_centroids:
            cx, cy = centroid
            
            for well in self.well_crops:
                x, y, w, h = well['bbox']
                
                if x <= cx <= x + w and y <= cy <= y + h:
                    relative_x = cx - x
                    relative_y = cy - y
                    
                    well['organoid_centroids'].append({
                        'absolute_pos': (cx, cy),
                        'relative_pos': (relative_x, relative_y),
                        'detection_type': detection_type
                    })
                    break
        
        # Print results
        wells_with_organoids = sum(1 for well in self.well_crops if len(well['organoid_centroids']) > 0)
        total_organoids = sum(len(well['organoid_centroids']) for well in self.well_crops)
        
        print(f"Wells with organoids: {wells_with_organoids}/{len(self.well_crops)}")
        print(f"Total organoids placed: {total_organoids}/{len(all_centroids)}")
    
    def analyze_individual_wells(self, output_dir):
        """Run Circle_Dropper_Filter on each well to detect boundaries"""
        print("\n--- ANALYZING INDIVIDUAL WELLS ---")
        
        well_files = list(Path(output_dir).glob("Color_well_*.png"))
        print(f"Analyzing {len(well_files)} well images...")
        
        for well_file in well_files:
            grid_label = well_file.name.replace("Color_well_", "").replace(".png", "")
            
            try:
                result = auto_filter_from_circle(
                    str(well_file), 
                    radius=45, 
                    tolerance=30, 
                    save_visualizations=False
                )
                
                self.well_boundary_results[grid_label] = result
                
                if result and result['success']:
                    print(f"  Well {grid_label}: Boundary detected")
                else:
                    print(f"  Well {grid_label}: No boundary")
                    
            except Exception as e:
                print(f"  Well {grid_label}: Error - {e}")
                self.well_boundary_results[grid_label] = None
    
    def save_results_csv(self):
        """Save comprehensive results to CSV"""
        print("\n--- SAVING RESULTS ---")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"well_organoid_analysis_{timestamp}.csv"
        
        headers = [
            'Well_ID', 'Grid_Label', 'Well_Detected', 'Well_Center_X', 'Well_Center_Y',
            'Organoid_Count', 'Organoid_Centroids', 'Detection_Types'
        ]
        
        results = []
        for i, well in enumerate(self.well_crops, 1):
            grid_label = well['grid_label']
            
            # Well boundary info
            well_result = self.well_boundary_results.get(grid_label)
            if well_result and well_result['success']:
                well_detected = 'Yes'
                well_center_x = well_result['centroid_x']
                well_center_y = well_result['centroid_y']
            else:
                well_detected = 'No'
                well_center_x = well_center_y = ''
            
            # Organoid info
            organoid_count = len(well['organoid_centroids'])
            organoid_centroids = ';'.join([f"({o['absolute_pos'][0]},{o['absolute_pos'][1]})" 
                                         for o in well['organoid_centroids']])
            detection_types = ';'.join([o['detection_type'] for o in well['organoid_centroids']])
            
            results.append([
                i, grid_label, well_detected, well_center_x, well_center_y,
                organoid_count, organoid_centroids, detection_types
            ])
        
        # Write CSV
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(results)
        
        print(f"Results saved to: {csv_filename}")
        return csv_filename
    
    def create_final_visualization(self, output_dir):
        """Create visualization showing wells with organoids and boundaries"""
        print("\n--- CREATING VISUALIZATION ---")
        
        # Filter wells with data
        wells_with_data = [w for w in self.well_crops 
                          if len(w['organoid_centroids']) > 0 or 
                             w['grid_label'] in self.well_boundary_results]
        
        if len(wells_with_data) == 0:
            print("No wells with data to visualize")
            return
        
        # Create subplot grid
        n_wells = min(16, len(wells_with_data))  # Limit to 16 wells for display
        cols = min(4, n_wells)
        rows = (n_wells + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'Well Analysis Results - Showing {n_wells} Wells', fontsize=16)
        
        for i, well in enumerate(wells_with_data[:n_wells]):
            ax = axes[i]
            grid_label = well['grid_label']
            
            # Load well image
            well_image_path = Path(output_dir) / f"Color_well_{grid_label}.png"
            if well_image_path.exists():
                well_image = cv2.imread(str(well_image_path))
                well_image = cv2.cvtColor(well_image, cv2.COLOR_BGR2RGB)
                
                ax.imshow(well_image)
                
                # Draw well boundary if detected
                if grid_label in self.well_boundary_results:
                    result = self.well_boundary_results[grid_label]
                    if result and result['success']:
                        # Draw boundary rectangle
                        x1, y1 = result['bbox_x1'], result['bbox_y1']
                        x2, y2 = result['bbox_x2'], result['bbox_y2']
                        
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           fill=False, edgecolor='blue', linewidth=2)
                        ax.add_patch(rect)
                        
                        # Mark well center
                        cx, cy = result['centroid_x'], result['centroid_y']
                        ax.plot(cx, cy, 'bo', markersize=6)
                
                # Draw organoid centroids - no scaling needed since crops are original size
                for j, organoid in enumerate(well['organoid_centroids']):
                    rel_x, rel_y = organoid['relative_pos']
                    detection_type = organoid['detection_type']
                    
                    # No scaling needed - use relative coordinates directly
                    color = 'red' if detection_type == 'binary' else 'green'
                    marker = 'o' if detection_type == 'binary' else 's'
                    
                    ax.plot(rel_x, rel_y, color=color, marker=marker, markersize=8)
                
                # Set title
                organoid_count = len(well['organoid_centroids'])
                well_detected = grid_label in self.well_boundary_results and \
                               self.well_boundary_results[grid_label] and \
                               self.well_boundary_results[grid_label]['success']
                
                title = f"Well {grid_label}\n{organoid_count} Organoids"
                if well_detected:
                    title += ", Well Found"
                
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f'Well {grid_label}\nImage not found', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_wells, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_filename = f"well_analysis_visualization_{timestamp}.png"
        plt.savefig(viz_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {viz_filename}")
        
        plt.show()
        return viz_filename
    
    def run_complete_analysis(self):
        """Execute the complete analysis pipeline"""
        print("Starting complete well-organoid analysis...")
        
        # Step 1: Load images
        if not self.load_images():
            return
        
        # Step 2: Run dual detection
        self.run_dual_detection()
        
        # Step 3: Create well crops
        output_dir = self.create_well_crops()
        
        # Step 4: Correlate centroids to wells
        self.correlate_centroids_to_wells()
        
        # Step 5: Analyze individual wells
        self.analyze_individual_wells(output_dir)
        
        # Step 6: Create visualization
        viz_file = self.create_final_visualization(output_dir)
        
        # Step 7: Save results
        csv_file = self.save_results_csv()
        
        # Cleanup temporary image file
        if self.image_with_organoids_path and os.path.exists(self.image_with_organoids_path):
            os.remove(self.image_with_organoids_path)
            print(f"Cleaned up temporary file: {self.image_with_organoids_path}")
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Binary centroids: {len(self.binary_centroids)}")
        print(f"Color centroids: {len(self.color_centroids)}")
        print(f"Wells analyzed: {len(self.well_crops)}")
        print(f"Results: {csv_file}")
        print(f"Well images: {output_dir}")

    def save_color_palette(self, sample_pixels, sample_masks):
        """Save color palette to file for future use"""
        palette_data = {
            "timestamp": datetime.now().isoformat(),
            "image_path": self.image_path,
            "sample_pixels": sample_pixels.tolist(),
            "sample_areas": []
        }
        
        # Save sample area information
        for i, mask_info in enumerate(sample_masks):
            palette_data["sample_areas"].append({
                "center": mask_info['center'],
                "radius": mask_info['radius']
            })
        
        try:
            with open(self.palette_file, 'w') as f:
                json.dump(palette_data, f, indent=2)
            print(f"✅ Color palette saved to {self.palette_file}")
            return True
        except Exception as e:
            print(f"❌ Error saving palette: {e}")
            return False
    
    def load_color_palette(self):
        """Load previously saved color palette"""
        if not os.path.exists(self.palette_file):
            print(f"No saved palette found at {self.palette_file}")
            return None
            
        try:
            with open(self.palette_file, 'r') as f:
                palette_data = json.load(f)
            
            sample_pixels = np.array(palette_data["sample_pixels"])
            print(f"✅ Loaded palette with {len(sample_pixels)} pixels from {palette_data['timestamp']}")
            print(f"   Original image: {os.path.basename(palette_data.get('image_path', 'Unknown'))}")
            print(f"   Sample areas: {len(palette_data.get('sample_areas', []))}")
            
            return sample_pixels
        except Exception as e:
            print(f"❌ Error loading palette: {e}")
            return None
    
    def ask_use_saved_palette(self):
        """Ask user if they want to use saved palette"""
        if not os.path.exists(self.palette_file):
            return False
            
        while True:
            try:
                choice = input("\nFound saved color palette. Use it? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                else:
                    print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                return False

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for color filter mode"""
        # Update mouse position for preview circle
        self.mouse_x = x
        self.mouse_y = y
        
        # Convert screen coordinates to image coordinates based on zoom
        img_x, img_y = self.screen_to_image_coords(x, y)
        
        # LEFT CLICK - Panning
        if event == cv2.EVENT_LBUTTONDOWN:
            self.panning = True
            self.pan_start_x = x
            self.pan_start_y = y
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.panning = False
            
        elif event == cv2.EVENT_MOUSEMOVE and self.panning:
            # Calculate pan delta
            dx = x - self.pan_start_x
            dy = y - self.pan_start_y
            
            # Update zoom center (inverse direction for natural feel)
            self.zoom_center_x -= int(dx / self.zoom_factor)
            self.zoom_center_y -= int(dy / self.zoom_factor)
            
            # Clamp to image bounds
            self.zoom_center_x = max(0, min(self.width - 1, self.zoom_center_x))
            self.zoom_center_y = max(0, min(self.height - 1, self.zoom_center_y))
            
            # Update pan start for continuous dragging
            self.pan_start_x = x
            self.pan_start_y = y
            
        # RIGHT CLICK - Place circular sample area
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.add_circular_sample(img_x, img_y, self.circle_radius)
            print(f"Circular sample area added at ({img_x}, {img_y}) with radius {self.circle_radius}")
            print(f"Total samples: {len(self.sample_masks)}. Press SPACE to process.")
            
        # Handle zoom with mouse wheel
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Set zoom center to current mouse position
            self.zoom_center_x = img_x
            self.zoom_center_y = img_y
            
            if flags > 0:  # Scroll up - zoom in
                self.zoom_factor = min(10.0, self.zoom_factor * 1.2)
            else:  # Scroll down - zoom out
                self.zoom_factor = max(0.1, self.zoom_factor / 1.2)
            
            print(f"Zoom: {self.zoom_factor:.2f}x at ({img_x}, {img_y})")

    def add_circular_sample(self, center_x, center_y, radius):
        """Add a circular sample area to the list"""
        h, w = self.height, self.width
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
        
        self.sample_masks.append({
            'mask': mask,
            'center': (center_x, center_y),
            'radius': radius
        })

    def screen_to_image_coords(self, screen_x, screen_y):
        """Convert screen coordinates to image coordinates based on zoom"""
        if self.zoom_factor == 1.0:
            return screen_x, screen_y

        if not hasattr(self, '_current_zoom_box'):
            return screen_x, screen_y
            
        x1, y1, zoom_w, zoom_h = self._current_zoom_box
        img_x = int(x1 + (screen_x / self.width) * zoom_w)
        img_y = int(y1 + (screen_y / self.height) * zoom_h)
        
        # Clamp to image bounds
        img_x = max(0, min(img_x, self.width - 1))
        img_y = max(0, min(img_y, self.height - 1))
        
        return img_x, img_y

    def get_zoomed_display_image(self):
        """Return a zoomed-in view of the image around the zoom center"""
        if self.zoom_factor == 1.0:
            self._current_zoom_box = (0, 0, self.width, self.height)
            return self.display_img

        zoom_w = int(self.width / self.zoom_factor)
        zoom_h = int(self.height / self.zoom_factor)

        x1 = max(0, self.zoom_center_x - zoom_w // 2)
        y1 = max(0, self.zoom_center_y - zoom_h // 2)
        x2 = min(self.width, x1 + zoom_w)
        y2 = min(self.height, y1 + zoom_h)

        # Adjust x1, y1 again if x2 or y2 were clipped
        x1 = max(0, x2 - zoom_w)
        y1 = max(0, y2 - zoom_h)

        zoomed_crop = self.display_img[y1:y2, x1:x2]
        resized = cv2.resize(zoomed_crop, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        self._current_zoom_box = (x1, y1, zoom_w, zoom_h)
        return resized

    def update_color_filter_display(self):
        """Update the display image for color filter mode"""
        self.display_img = self.original_image.copy()
        
        # Add info text
        info_text = f"Color Filter Mode (Tolerance: {self.color_tolerance}, Zoom: {self.zoom_factor:.1f}x)"
        cv2.putText(self.display_img, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw existing circular sample areas
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, mask_info in enumerate(self.sample_masks):
            color = colors[i % len(colors)]
            center = mask_info['center']
            radius = mask_info['radius']
            cv2.circle(self.display_img, center, radius, color, 2)
            cv2.putText(self.display_img, f"S{i+1}", 
                       (center[0] - 15, center[1] - radius - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw preview circle at mouse position
        if hasattr(self, '_current_zoom_box'):
            # Convert mouse screen coordinates to image coordinates
            img_mouse_x, img_mouse_y = self.screen_to_image_coords(self.mouse_x, self.mouse_y)
            
            # Draw preview circle in white with dashed effect
            cv2.circle(self.display_img, (img_mouse_x, img_mouse_y), self.circle_radius, (255, 255, 255), 2)
            cv2.circle(self.display_img, (img_mouse_x, img_mouse_y), self.circle_radius, (0, 0, 0), 1)
            
            # Draw center crosshair
            cv2.line(self.display_img, (img_mouse_x - 5, img_mouse_y), (img_mouse_x + 5, img_mouse_y), (255, 255, 255), 1)
            cv2.line(self.display_img, (img_mouse_x, img_mouse_y - 5), (img_mouse_x, img_mouse_y + 5), (255, 255, 255), 1)
        
        # Add circle size and sample count info
        circle_info = f"Circle radius: {self.circle_radius} | Samples: {len(self.sample_masks)}"
        cv2.putText(self.display_img, circle_info, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add instructions
        if len(self.sample_masks) > 0:
            instruction = "Press SPACE to process samples"
        else:
            instruction = "Right-click to place circular sample areas"
        
        cv2.putText(self.display_img, instruction, (10, self.height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def apply_frequency_filtering(self, sample_pixels):
        """Apply frequency-based filtering to remove 15% outliers from each edge of the distribution"""
        print("📊 Analyzing pixel frequency distribution...")
        
        # Convert sample pixels to strings for frequency counting
        pixel_strings = [f"{p[0]},{p[1]},{p[2]}" for p in sample_pixels]
        
        # Count frequency of each unique pixel
        from collections import Counter
        pixel_counts = Counter(pixel_strings)
        
        print(f"  Found {len(pixel_counts)} unique pixel values")
        print(f"  Original sample size: {len(sample_pixels)}")
        
        # Sort by frequency (ascending)
        sorted_pixels = sorted(pixel_counts.items(), key=lambda x: x[1])
        
        # Calculate cumulative frequencies
        total_count = sum(pixel_counts.values())
        cumulative_freq = 0
        frequency_data = []
        
        for pixel_str, count in sorted_pixels:
            cumulative_freq += count
            frequency_data.append({
                'pixel': pixel_str,
                'count': count,
                'cumulative_freq': cumulative_freq,
                'percentile': (cumulative_freq / total_count) * 100
            })
        
        # Filter out bottom 25% (keep upper 75%)
        filtered_pixels = []
        for data in frequency_data:
            percentile = data['percentile']
            if 25.0 <= percentile:  # Keep upper 75%
                pixel_str = data['pixel']
                count = data['count']
                # Add this pixel 'count' times to filtered list
                h, s, v = map(int, pixel_str.split(','))
                for _ in range(count):
                    filtered_pixels.append([h, s, v])
        
        filtered_sample_pixels = np.array(filtered_pixels)
        
        print(f"  Filtered sample size: {len(filtered_sample_pixels)}")
        print(f"  Removed {len(sample_pixels) - len(filtered_sample_pixels)} pixels ({((len(sample_pixels) - len(filtered_sample_pixels)) / len(sample_pixels) * 100):.1f}%)")
        print(f"  Kept upper 75% of frequency distribution")
        
        return filtered_sample_pixels

    def find_circular_contours_with_centroids(self, image, min_diameter=30, max_diameter=110, circularity_threshold=0.6):
        """Find circular contours and return centroids"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        kernel = np.ones((3, 3), np.uint8)
        all_centroids = []

        for i in range(4):  # 0 to 3 erosions
            eroded = gray.copy() if i == 0 else cv2.erode(gray.copy(), kernel, iterations=i)
            blurred = cv2.GaussianBlur(eroded, (3, 3), 0)
            edged = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            iteration_threshold = circularity_threshold * (1 - 0.05 * i)
            iter_min_diameter = min_diameter - 1

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area == 0:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                (_, _), radius = cv2.minEnclosingCircle(cnt)
                diameter = 2 * radius

                if circularity >= iteration_threshold and iter_min_diameter < diameter <= max_diameter:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        all_centroids.append((cx, cy))

        # Merge close centroids
        print(f"Before merging: {len(all_centroids)} centroids")
        merged_centroids = self.merge_close_centroids(all_centroids, threshold=24)
        print(f"After merging: {len(merged_centroids)} centroids")

        return merged_centroids

    def process_color_samples_for_centroids(self):
        """Process color samples and return centroids and filtered image"""
        if len(self.sample_masks) == 0:
            return [], np.zeros_like(self.original_image)
        
        # Create combined mask from all circular samples
        combined_mask = np.zeros((self.height, self.width), dtype=bool)
        for mask_info in self.sample_masks:
            combined_mask = combined_mask | mask_info['mask']
        
        # Convert to HSV and sample pixels
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        sample_pixels = hsv[combined_mask]
        
        if len(sample_pixels) == 0:
            return [], np.zeros_like(self.original_image)
        
        # Apply frequency filtering
        filtered_sample_pixels = self.apply_frequency_filtering(sample_pixels)
        
        # Save color palette
        self.save_color_palette(filtered_sample_pixels, self.sample_masks)
        
        # Create K-means clustered filtering
        n_clusters = min(max(2, len(sample_pixels) // 1000), 8)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(sample_pixels)
        
        # Create bounding box mask
        bounding_box_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for i in range(n_clusters):
            cluster_pixels = filtered_sample_pixels[cluster_labels == i]
            if len(cluster_pixels) == 0:
                continue
                
            h_min, h_max = np.min(cluster_pixels[:, 0]), np.max(cluster_pixels[:, 0])
            s_min, s_max = np.min(cluster_pixels[:, 1]), np.max(cluster_pixels[:, 1])
            v_min, v_max = np.min(cluster_pixels[:, 2]), np.max(cluster_pixels[:, 2])
            
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])
            cluster_mask = cv2.inRange(hsv, lower_bound, upper_bound)
            bounding_box_mask = cv2.bitwise_or(bounding_box_mask, cluster_mask)
        
        # Apply filtering
        filtered_image = cv2.bitwise_and(self.original_image, self.original_image, mask=bounding_box_mask)
        
        # Find centroids from filtered image
        color_centroids = self.find_circular_contours_with_centroids(filtered_image)
        
        return color_centroids, filtered_image

    def process_saved_palette_for_centroids(self, sample_pixels):
        """Process saved palette and return centroids and filtered image"""
        # Convert to HSV
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        
        # Apply frequency filtering
        filtered_sample_pixels = self.apply_frequency_filtering(sample_pixels)
        
        # K-means clustering
        n_clusters = min(max(2, len(filtered_sample_pixels) // 1000), 8)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(filtered_sample_pixels)
        
        bounding_box_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for i in range(n_clusters):
            cluster_pixels = filtered_sample_pixels[cluster_labels == i]
            if len(cluster_pixels) == 0:
                continue
                
            h_min, h_max = np.min(cluster_pixels[:, 0]), np.max(cluster_pixels[:, 0])
            s_min, s_max = np.min(cluster_pixels[:, 1]), np.max(cluster_pixels[:, 1])
            v_min, v_max = np.min(cluster_pixels[:, 2]), np.max(cluster_pixels[:, 2])
            
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])
            cluster_mask = cv2.inRange(hsv, lower_bound, upper_bound)
            bounding_box_mask = cv2.bitwise_or(bounding_box_mask, cluster_mask)
        
        # Apply filtering
        filtered_image = cv2.bitwise_and(self.original_image, self.original_image, mask=bounding_box_mask)
        
        # Find centroids
        color_centroids = self.find_circular_contours_with_centroids(filtered_image)
        
        return color_centroids, filtered_image

def main():
    """Main function to run the well organoid analyzer"""
    analyzer = WellOrganoidAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 