#!/usr/bin/env python3
"""
Comprehensive Well Organoid Analysis Pipeline
Simple version without debugging GUI
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
import csv
import os
from datetime import datetime
from tkinter import filedialog, messagebox, Tk
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist, squareform
import glob

class AnalysisParameters:
    """Centralized parameter definitions for the Well Organoid Analyzer"""
    
    # === QUADRANT WELL LABELING PARAMETERS ===
    QUADRANT_ROW_RANGE = ['m', 'l', 'k', 'j', 'i', 'h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']
    QUADRANT_COL_RANGE = list(range(10, 25))
    ENABLE_QUADRANT_LABELING = True
    
    # === BINARY DETECTION PARAMETERS ===
    BINARY_DARK_THRESHOLD = 80
    BINARY_INPAINT_RADIUS = 20
    BINARY_THRESHOLD = 110
    BINARY_MIN_DIAMETER = 15
    BINARY_MAX_DIAMETER = 50
    BINARY_CIRCULARITY_THRESHOLD = 0.75
    BINARY_EROSION_STAGES = 4
    
    # === COLOR DETECTION PARAMETERS ===
    COLOR_MIN_DIAMETER = 30
    COLOR_MAX_DIAMETER = 110
    COLOR_CIRCULARITY_THRESHOLD = 0.6
    COLOR_EROSION_STAGES = 9
    COLOR_GAUSSIAN_BLUR_SIZE = 3
    COLOR_CANNY_LOW_THRESHOLD = 50
    COLOR_CANNY_HIGH_THRESHOLD = 150
    
    # === CENTROID MERGING PARAMETERS ===
    CENTROID_MERGE_THRESHOLD = 24
    
    # === FREQUENCY FILTERING PARAMETERS ===
    FREQUENCY_FILTER_PERCENTAGE = 15
    
    # === SAMPLE COLLECTION PARAMETERS ===
    SAMPLE_RADIUS = 15
    MIN_SAMPLES_REQUIRED = 3
    
    # === WELL BOUNDARY DETECTION PARAMETERS ===
    WELL_BOUNDARY_RADIUS = 45
    WELL_BOUNDARY_TOLERANCE = 30
    
    # === VISUALIZATION PARAMETERS ===
    VISUALIZATION_MAX_WELLS = 16
    VISUALIZATION_COLS = 4
    ORGANOID_MARKER_SIZE = 8
    WELL_BBOX_EXPANSION = 0.07
    SHOW_DISPLAY_GRAPHS = True
    
    # === FILE PARAMETERS ===
    COLOR_PALETTE_FILENAME = "color_palette_save.json"
    WELL_CROPS_FOLDER = "well_crops"
    BLANK_LOCATIONS_FILE = "Locations_Blank.txt"
    
    # === DISPLAY PARAMETERS ===
    DISPLAY_ZOOM_FACTOR = 0.3
    DISPLAY_WINDOW_SIZE = (1200, 800)

    # === DEBUG AND CONTROL PARAMETERS ===
    ENABLE_INTERACTIVE_FINAL_CHECK = True
    DEBUG_MODE = False  # Disabled in simple version
    ENABLE_WELL_ANALYSIS = True
    SHOW_FINAL_OVERVIEW = True

class QuadrantMaskCropperIntegrated:
    """Integrated mask cropper for quadrant analysis"""
    
    def __init__(self, input_image_path, mask_path, row_range, col_range, output_folder="Quadrant_Crops", target_size=None):
        self.input_image_path = input_image_path
        self.mask_path = mask_path
        self.row_range = row_range
        self.col_range = col_range
        self.output_folder = output_folder
        self.target_size = target_size
        self.input_image = None
        self.mask_image = None
        self.regions = []
        
        os.makedirs(output_folder, exist_ok=True)

    def load_images(self):
        """Load input image and mask"""
        self.input_image = cv2.imread(self.input_image_path)
        if self.input_image is None:
            raise ValueError(f"Could not load input image: {self.input_image_path}")
        
        self.mask_image = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask_image is None:
            raise ValueError(f"Could not load mask image: {self.mask_path}")
        
        print(f"Input image shape: {self.input_image.shape}")
        print(f"Mask image shape: {self.mask_image.shape}")
        return True

    def find_connected_regions(self):
        """Find connected regions in the mask"""
        _, binary_mask = cv2.threshold(self.mask_image, 127, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                regions.append({
                    'id': i,
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area
                })
        
        self.regions = sorted(regions, key=lambda r: (r['center'][1], r['center'][0]))
        print(f"Found {len(self.regions)} regions")
        return self.regions

    def assign_quadrant_grid_positions(self, regions):
        """Assign grid positions to regions based on quadrant layout"""
        if not regions:
            return []
        
        centers = [r['center'] for r in regions]
        centers_array = np.array(centers)
        
        y_coords = centers_array[:, 1]
        unique_rows = len(set(np.round(y_coords / 50).astype(int)))
        
        x_coords = centers_array[:, 0]
        unique_cols = len(set(np.round(x_coords / 50).astype(int)))
        
        print(f"Detected grid: {unique_rows} rows x {unique_cols} cols")
        
        sorted_regions = sorted(regions, key=lambda r: (r['center'][1], r['center'][0]))
        
        assigned_regions = []
        for i, region in enumerate(sorted_regions):
            if i < len(self.row_range) * len(self.col_range):
                row_idx = i // len(self.col_range)
                col_idx = i % len(self.col_range)
                
                if row_idx < len(self.row_range) and col_idx < len(self.col_range):
                    row_label = self.row_range[row_idx].upper()
                    col_label = self.col_range[col_idx]
                    well_id = f"{row_label}{col_label}"
                    
                    region['well_id'] = well_id
                    region['row'] = row_label
                    region['col'] = col_label
                    assigned_regions.append(region)
        
        return assigned_regions

class WellOrganoidAnalyzer:
    """Comprehensive analyzer for well-plate organoid detection and analysis"""
    
    def __init__(self, show_visualizations=False, debug_mode=None):
        self.params = AnalysisParameters()
        self.show_visualizations = show_visualizations
        self.debug_mode = debug_mode if debug_mode is not None else self.params.DEBUG_MODE
        
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
        self.circle_radius = self.params.SAMPLE_RADIUS
        self.mouse_x = 0
        self.mouse_y = 0
        self.color_tolerance = 30
        
        # Results storage
        self.binary_centroids = []
        self.color_centroids = []
        self.well_crops = []
        self.well_boundary_results = {}
        
        # GUI state
        self.zoom_factor = self.params.DISPLAY_ZOOM_FACTOR
        self.zoom_center_x = 0
        self.zoom_center_y = 0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.display_img = None
        self.window_name = "Well Organoid Analyzer"
        
        # Color palette
        self.palette_file = self.params.COLOR_PALETTE_FILENAME

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
                
            dists = squareform(pdist(centroids))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)

            if min_dist < threshold:
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                merged = (centroids[i] + centroids[j]) / 2
                centroids = np.delete(centroids, [i, j], axis=0)
                centroids = np.vstack([centroids, merged])
                changed = True

        return [tuple(map(int, c)) for c in centroids]

    def run_dual_detection(self):
        """Run both binary and color detection methods"""
        print("\n=== DUAL DETECTION PIPELINE ===")
        
        # Binary detection
        print("\n1. Binary Detection:")
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Create dark mask for inpainting
        dark_mask = cv2.threshold(gray, self.params.BINARY_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)[1]
        
        # Inpaint dark regions
        inpainted = cv2.inpaint(self.original_image, dark_mask, self.params.BINARY_INPAINT_RADIUS, cv2.INPAINT_TELEA)
        
        # Convert to grayscale and apply binary threshold
        inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        binary_plate = cv2.threshold(inpainted_gray, self.params.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        
        # Find binary centroids
        self.binary_centroids = self.find_binary_centroids(binary_plate)
        print(f"   Found {len(self.binary_centroids)} binary centroids")
        
        # Color detection
        print("\n2. Color Detection:")
        if self.ask_use_saved_palette():
            sample_pixels = self.load_color_palette()
            if sample_pixels:
                self.color_centroids = self.process_saved_palette_for_centroids(sample_pixels)
                print(f"   Found {len(self.color_centroids)} color centroids using saved palette")
            else:
                print("   No saved palette found, running color sampling...")
                self.run_color_sampling()
        else:
            self.run_color_sampling()
        
        # Merge close centroids
        if self.binary_centroids:
            self.binary_centroids = self.merge_close_centroids(self.binary_centroids, self.params.CENTROID_MERGE_THRESHOLD)
        if self.color_centroids:
            self.color_centroids = self.merge_close_centroids(self.color_centroids, self.params.CENTROID_MERGE_THRESHOLD)
        
        print(f"\nFinal results after merging:")
        print(f"   Binary centroids: {len(self.binary_centroids)}")
        print(f"   Color centroids: {len(self.color_centroids)}")

    def find_binary_centroids(self, image, min_diameter=None, max_diameter=None, circularity_threshold=None):
        """Find organoid centroids using binary detection with erosion stages"""
        min_diameter = min_diameter or self.params.BINARY_MIN_DIAMETER
        max_diameter = max_diameter or self.params.BINARY_MAX_DIAMETER
        circularity_threshold = circularity_threshold or self.params.BINARY_CIRCULARITY_THRESHOLD
        
        # Invert binary image (organoids should be white)
        inverted_binary = cv2.bitwise_not(image)
        
        # Apply mask filtering if mask is available
        if hasattr(self, 'mask_path') and self.mask_path and self.mask_path != "dummy_mask.png":
            try:
                mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Resize mask to match image if needed
                    if mask.shape != inverted_binary.shape:
                        mask = cv2.resize(mask, (inverted_binary.shape[1], inverted_binary.shape[0]))
                    
                    # Apply mask filter - keep only white areas of mask
                    inverted_binary = cv2.bitwise_and(inverted_binary, mask)
                    print(f"   Applied mask filter to binary detection")
            except Exception as e:
                print(f"   Warning: Could not apply mask filter: {e}")
        
        # Apply erosion stages
        kernel = np.ones((3, 3), np.uint8)
        if self.params.BINARY_EROSION_STAGES > 0:
            inverted_binary = cv2.erode(inverted_binary, kernel, iterations=self.params.BINARY_EROSION_STAGES)
        
        # Find contours
        contours, _ = cv2.findContours(inverted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_centroids = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue
            
            # Calculate equivalent diameter
            diameter = 2 * np.sqrt(area / np.pi)
            if diameter < min_diameter or diameter > max_diameter:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < circularity_threshold:
                continue
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                valid_centroids.append((cx, cy))
        
        return valid_centroids

    def find_circular_contours_with_centroids(self, image, min_diameter=None, max_diameter=None, circularity_threshold=None):
        """Find circular contours and return their centroids"""
        min_diameter = min_diameter or self.params.COLOR_MIN_DIAMETER
        max_diameter = max_diameter or self.params.COLOR_MAX_DIAMETER
        circularity_threshold = circularity_threshold or self.params.COLOR_CIRCULARITY_THRESHOLD
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (self.params.COLOR_GAUSSIAN_BLUR_SIZE, self.params.COLOR_GAUSSIAN_BLUR_SIZE), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.params.COLOR_CANNY_LOW_THRESHOLD, self.params.COLOR_CANNY_HIGH_THRESHOLD)
        
        # Apply erosion stages
        kernel = np.ones((3, 3), np.uint8)
        if self.params.COLOR_EROSION_STAGES > 0:
            edges = cv2.erode(edges, kernel, iterations=self.params.COLOR_EROSION_STAGES)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_centroids = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue
            
            # Calculate equivalent diameter
            diameter = 2 * np.sqrt(area / np.pi)
            if diameter < min_diameter or diameter > max_diameter:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < circularity_threshold:
                continue
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                valid_centroids.append((cx, cy))
        
        return valid_centroids

    def create_image_with_organoids(self):
        """Create visualization image with detected organoids"""
        if self.original_image is None:
            return None
            
        result_image = self.original_image.copy()
        
        # Draw binary centroids in blue
        for cx, cy in self.binary_centroids:
            cv2.circle(result_image, (cx, cy), 8, (255, 0, 0), 2)  # Blue
            cv2.circle(result_image, (cx, cy), 2, (255, 0, 0), -1)
        
        # Draw color centroids in red
        for cx, cy in self.color_centroids:
            cv2.circle(result_image, (cx, cy), 8, (0, 0, 255), 2)  # Red
            cv2.circle(result_image, (cx, cy), 2, (0, 0, 255), -1)
        
        # Save result image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_with_organoids_path = f"image_with_organoids_{timestamp}.png"
        cv2.imwrite(self.image_with_organoids_path, result_image)
        
        self.image_with_organoids = result_image
        return result_image

    def create_well_crops(self):
        """Create well crops using integrated mask cropper"""
        if not self.mask_path or not self.image_path:
            print("Error: Missing image or mask path")
            return []
        
        print("\n=== CREATING WELL CROPS ===")
        
        # Create cropper instance
        cropper = QuadrantMaskCropperIntegrated(
            self.image_path,
            self.mask_path,
            self.params.QUADRANT_ROW_RANGE,
            self.params.QUADRANT_COL_RANGE,
            "well_analysis_output"
        )
        
        # Load images and find regions
        cropper.load_images()
        regions = cropper.find_connected_regions()
        assigned_regions = cropper.assign_quadrant_grid_positions(regions)
        
        # Create crops
        crops = []
        for region in assigned_regions:
            x, y, w, h = region['bbox']
            
            # Expand bounding box
            expansion = int(min(w, h) * self.params.WELL_BBOX_EXPANSION)
            x = max(0, x - expansion)
            y = max(0, y - expansion)
            w = min(cropper.input_image.shape[1] - x, w + 2 * expansion)
            h = min(cropper.input_image.shape[0] - y, h + 2 * expansion)
            
            # Extract crop
            crop = cropper.input_image[y:y+h, x:x+w]
            
            # Save crop
            crop_filename = f"Color_well_{region['well_id']}.png"
            crop_path = os.path.join("well_analysis_output", crop_filename)
            cv2.imwrite(crop_path, crop)
            
            crop_info = {
                'well_id': region['well_id'],
                'bbox': (x, y, w, h),
                'center': region['center'],
                'crop': crop,
                'crop_path': crop_path
            }
            crops.append(crop_info)
        
        self.well_crops = crops
        print(f"Created {len(crops)} well crops")
        return crops

    def correlate_centroids_to_wells(self):
        """Correlate detected centroids with well crops"""
        if not self.well_crops:
            print("No well crops available for correlation")
            return
        
        print("\n=== CORRELATING CENTROIDS TO WELLS ===")
        
        for crop_info in self.well_crops:
            well_id = crop_info['well_id']
            x, y, w, h = crop_info['bbox']
            well_center = crop_info['center']
            
            # Find centroids within this well
            binary_in_well = []
            color_in_well = []
            
            for cx, cy in self.binary_centroids:
                if x <= cx <= x + w and y <= cy <= y + h:
                    # Convert to well-relative coordinates
                    rel_x = cx - x
                    rel_y = cy - y
                    binary_in_well.append((rel_x, rel_y))
            
            for cx, cy in self.color_centroids:
                if x <= cx <= x + w and y <= cy <= y + h:
                    # Convert to well-relative coordinates
                    rel_x = cx - x
                    rel_y = cy - y
                    color_in_well.append((rel_x, rel_y))
            
            crop_info['binary_centroids'] = binary_in_well
            crop_info['color_centroids'] = color_in_well
            
            print(f"  {well_id}: {len(binary_in_well)} binary, {len(color_in_well)} color")

    def save_results_csv(self):
        """Save analysis results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save organoid locations
        locations_file = f"organoid_locations_formatted_quadrant_{timestamp}.csv"
        with open(locations_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['X', 'Y', 'Detection_Type'])
            
            for x, y in self.binary_centroids:
                writer.writerow([x, y, 'Binary'])
            
            for x, y in self.color_centroids:
                writer.writerow([x, y, 'Color'])
        
        print(f"Saved organoid locations to: {locations_file}")
        
        # Save well analysis
        if self.well_crops:
            well_analysis_file = f"well_organoid_analysis_quadrant_{timestamp}.csv"
            with open(well_analysis_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Well_ID', 'Binary_Count', 'Color_Count', 'Total_Count', 'Well_Center_X', 'Well_Center_Y'])
                
                for crop_info in self.well_crops:
                    well_id = crop_info['well_id']
                    binary_count = len(crop_info.get('binary_centroids', []))
                    color_count = len(crop_info.get('color_centroids', []))
                    total_count = binary_count + color_count
                    center_x, center_y = crop_info['center']
                    
                    writer.writerow([well_id, binary_count, color_count, total_count, center_x, center_y])
            
            print(f"Saved well analysis to: {well_analysis_file}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🧬 STARTING COMPLETE ORGANOID ANALYSIS")
        print("=" * 50)
        
        # Load images
        if not self.load_images():
            print("❌ Failed to load images")
            return False
        
        # Run dual detection
        self.run_dual_detection()
        
        # Create visualization
        self.create_image_with_organoids()
        print(f"✅ Created organoid visualization: {self.image_with_organoids_path}")
        
        # Create well crops and correlate
        if self.params.ENABLE_WELL_ANALYSIS:
            self.create_well_crops()
            self.correlate_centroids_to_wells()
        
        # Save results
        self.save_results_csv()
        
        # Show final summary
        print(f"\n📊 ANALYSIS COMPLETE!")
        print(f"   Binary detections: {len(self.binary_centroids)}")
        print(f"   Color detections: {len(self.color_centroids)}")
        print(f"   Total organoids: {len(self.binary_centroids) + len(self.color_centroids)}")
        if self.well_crops:
            print(f"   Wells analyzed: {len(self.well_crops)}")
        
        return True

    def save_color_palette(self, sample_pixels, sample_masks):
        """Save collected color samples to JSON file"""
        palette_data = {
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(sample_pixels),
            'samples': []
        }
        
        for i, pixels in enumerate(sample_pixels):
            sample_data = {
                'id': i,
                'pixels': pixels.tolist(),
                'mean_color': np.mean(pixels, axis=0).tolist(),
                'std_color': np.std(pixels, axis=0).tolist()
            }
            palette_data['samples'].append(sample_data)
        
        with open(self.palette_file, 'w') as f:
            json.dump(palette_data, f, indent=2)
        
        print(f"✅ Saved color palette with {len(sample_pixels)} samples to {self.palette_file}")

    def load_color_palette(self):
        """Load saved color palette from JSON file"""
        if not os.path.exists(self.palette_file):
            return None
        
        try:
            with open(self.palette_file, 'r') as f:
                palette_data = json.load(f)
            
            sample_pixels = []
            for sample in palette_data['samples']:
                pixels = np.array(sample['pixels'])
                sample_pixels.append(pixels)
            
            print(f"✅ Loaded color palette with {len(sample_pixels)} samples")
            return sample_pixels
            
        except Exception as e:
            print(f"❌ Error loading color palette: {e}")
            return None

    def ask_use_saved_palette(self):
        """Ask user if they want to use saved color palette"""
        if not os.path.exists(self.palette_file):
            return False
        
        root = Tk()
        root.withdraw()
        
        result = messagebox.askyesno(
            "Saved Color Palette Found",
            f"Found saved color palette: {self.palette_file}\n\n"
            "Would you like to use the saved palette for color detection?\n\n"
            "• YES: Use saved palette (faster)\n"
            "• NO: Collect new color samples"
        )
        
        root.destroy()
        return result

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for color sampling"""
        self.mouse_x, self.mouse_y = self.screen_to_image_coords(x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_circular_sample(self.mouse_x, self.mouse_y, self.circle_radius)
            self.update_color_filter_display()
        elif event == cv2.EVENT_RBUTTONDOWN and self.sample_masks:
            self.sample_masks.pop()
            self.update_color_filter_display()
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.zoom_factor = min(2.0, self.zoom_factor * 1.1)
            else:
                self.zoom_factor = max(0.1, self.zoom_factor * 0.9)
            self.update_color_filter_display()
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start_x, self.pan_start_y = x, y
        elif event == cv2.EVENT_MBUTTONUP:
            self.panning = False
        elif event == cv2.EVENT_MOUSEMOVE and self.panning:
            dx = x - self.pan_start_x
            dy = y - self.pan_start_y
            self.zoom_center_x -= dx / self.zoom_factor
            self.zoom_center_y -= dy / self.zoom_factor
            self.pan_start_x, self.pan_start_y = x, y
            self.update_color_filter_display()

    def add_circular_sample(self, center_x, center_y, radius):
        """Add a circular sample area"""
        mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        self.sample_masks.append(mask)
        print(f"Added sample {len(self.sample_masks)} at ({center_x}, {center_y})")

    def screen_to_image_coords(self, screen_x, screen_y):
        """Convert screen coordinates to image coordinates"""
        if self.display_img is None:
            return screen_x, screen_y
        
        display_h, display_w = self.display_img.shape[:2]
        
        # Calculate the visible area in the original image
        visible_w = int(display_w / self.zoom_factor)
        visible_h = int(display_h / self.zoom_factor)
        
        # Calculate the top-left corner of the visible area
        start_x = max(0, self.zoom_center_x - visible_w // 2)
        start_y = max(0, self.zoom_center_y - visible_h // 2)
        
        # Convert screen coordinates to image coordinates
        image_x = int(start_x + screen_x / self.zoom_factor)
        image_y = int(start_y + screen_y / self.zoom_factor)
        
        # Clamp to image bounds
        image_x = max(0, min(self.width - 1, image_x))
        image_y = max(0, min(self.height - 1, image_y))
        
        return image_x, image_y

    def get_zoomed_display_image(self):
        """Get the zoomed and panned display image"""
        if self.original_image is None:
            return None
        
        h, w = self.original_image.shape[:2]
        
        # Calculate visible area
        visible_w = int(w / self.zoom_factor)
        visible_h = int(h / self.zoom_factor)
        
        # Calculate crop bounds
        start_x = max(0, min(w - visible_w, self.zoom_center_x - visible_w // 2))
        start_y = max(0, min(h - visible_h, self.zoom_center_y - visible_h // 2))
        end_x = start_x + visible_w
        end_y = start_y + visible_h
        
        # Crop and resize
        cropped = self.original_image[start_y:end_y, start_x:end_x]
        display_img = cv2.resize(cropped, (w, h))
        
        return display_img

    def update_color_filter_display(self):
        """Update the color sampling display"""
        if self.original_image is None:
            return
        
        self.display_img = self.get_zoomed_display_image()
        display = self.display_img.copy()
        
        # Draw sample circles
        for mask in self.sample_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)
        
        # Draw current mouse position
        screen_x = int((self.mouse_x - (self.zoom_center_x - self.width // (2 * self.zoom_factor))) * self.zoom_factor)
        screen_y = int((self.mouse_y - (self.zoom_center_y - self.height // (2 * self.zoom_factor))) * self.zoom_factor)
        
        if 0 <= screen_x < display.shape[1] and 0 <= screen_y < display.shape[0]:
            cv2.circle(display, (screen_x, screen_y), int(self.circle_radius * self.zoom_factor), (255, 255, 0), 2)
        
        cv2.imshow(self.window_name, display)

    def apply_frequency_filtering(self, sample_pixels):
        """Apply frequency-based filtering to remove outliers"""
        if not sample_pixels or self.params.FREQUENCY_FILTER_PERCENTAGE <= 0:
            return sample_pixels
        
        print(f"Applying frequency filtering ({self.params.FREQUENCY_FILTER_PERCENTAGE}% outlier removal)...")
        
        filtered_samples = []
        for pixels in sample_pixels:
            if len(pixels) == 0:
                continue
            
            # Calculate distances from mean
            mean_color = np.mean(pixels, axis=0)
            distances = np.linalg.norm(pixels - mean_color, axis=1)
            
            # Remove outliers
            percentile_low = self.params.FREQUENCY_FILTER_PERCENTAGE
            percentile_high = 100 - self.params.FREQUENCY_FILTER_PERCENTAGE
            
            low_thresh = np.percentile(distances, percentile_low)
            high_thresh = np.percentile(distances, percentile_high)
            
            filtered_pixels = pixels[(distances >= low_thresh) & (distances <= high_thresh)]
            
            if len(filtered_pixels) > 0:
                filtered_samples.append(filtered_pixels)
        
        print(f"Frequency filtering: {len(sample_pixels)} → {len(filtered_samples)} samples")
        return filtered_samples

    def process_color_samples_for_centroids(self):
        """Process collected color samples to find centroids"""
        if len(self.sample_masks) < self.params.MIN_SAMPLES_REQUIRED:
            print(f"Need at least {self.params.MIN_SAMPLES_REQUIRED} samples for color detection")
            return []
        
        print("Processing color samples...")
        
        # Collect sample pixels
        sample_pixels = []
        for mask in self.sample_masks:
            pixels = self.original_image[mask > 0]
            if len(pixels) > 0:
                sample_pixels.append(pixels)
        
        # Apply frequency filtering
        sample_pixels = self.apply_frequency_filtering(sample_pixels)
        
        if not sample_pixels:
            print("No valid samples after filtering")
            return []
        
        # Save color palette
        self.save_color_palette(sample_pixels, self.sample_masks)
        
        # Process samples for centroid detection
        return self.process_saved_palette_for_centroids(sample_pixels)

    def process_saved_palette_for_centroids(self, sample_pixels):
        """Process saved color palette to find centroids"""
        print("Processing color palette for organoid detection...")
        
        # Create color filter
        combined_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
        
        for pixels in sample_pixels:
            mean_color = np.mean(pixels, axis=0)
            std_color = np.std(pixels, axis=0)
            
            # Create color range (mean ± 2*std)
            lower_bound = np.maximum(0, mean_color - 2 * std_color).astype(np.uint8)
            upper_bound = np.minimum(255, mean_color + 2 * std_color).astype(np.uint8)
            
            # Create mask for this color range
            mask = cv2.inRange(self.original_image, lower_bound, upper_bound)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Find centroids in the filtered image
        centroids = self.find_circular_contours_with_centroids(combined_mask)
        print(f"Found {len(centroids)} color centroids")
        
        return centroids

    def run_color_sampling(self):
        """Run interactive color sampling"""
        print("\n=== COLOR SAMPLING MODE ===")
        print("Controls:")
        print("  Left click: Add color sample")
        print("  Right click: Remove last sample")
        print("  Mouse wheel: Zoom in/out")
        print("  Middle drag: Pan image")
        print("  SPACE: Process samples")
        print("  ESC: Skip color detection")
        
        self.sample_masks = []
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *self.params.DISPLAY_WINDOW_SIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_color_filter_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC - skip color detection
                print("Skipping color detection")
                self.color_centroids = []
                break
            elif key == ord(' '):  # SPACE - process samples
                if len(self.sample_masks) >= self.params.MIN_SAMPLES_REQUIRED:
                    self.color_centroids = self.process_color_samples_for_centroids()
                    break
                else:
                    print(f"Need at least {self.params.MIN_SAMPLES_REQUIRED} samples (current: {len(self.sample_masks)})")
        
        cv2.destroyAllWindows()

def main():
    """Main function to run the analysis"""
    analyzer = WellOrganoidAnalyzer(show_visualizations=True)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 