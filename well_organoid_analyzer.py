#!/usr/bin/env python3
"""
=== WELL ORGANOID ANALYZER ===
Comprehensive pipeline for detecting organoids in well plate images using dual detection methods
and correlating them to individual wells.

Author: AI Assistant
Created: 2025
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

# ============================================================================
# PARAMETER DEFINITIONS
# ============================================================================

class AnalysisParameters:
    """Centralized parameter definitions for the Well Organoid Analyzer"""
    
    # === BINARY DETECTION PARAMETERS ===
    BINARY_DARK_THRESHOLD = 122         # Threshold for creating dark mask for inpainting (0-255)
    BINARY_INPAINT_RADIUS = 20          # Radius for inpainting dark regions (pixels)
    BINARY_THRESHOLD = 163              # Binarization threshold after inpainting (0-255)
    BINARY_MIN_DIAMETER = 15            # Minimum organoid diameter for binary detection (pixels)
    BINARY_MAX_DIAMETER = 50            # Maximum organoid diameter for binary detection (pixels)
    BINARY_CIRCULARITY_THRESHOLD = 0.75 # Minimum circularity for binary detection (0.0-1.0)
    BINARY_EROSION_STAGES = 4           # Number of erosion stages to apply (0-8)
    
    # === COLOR DETECTION PARAMETERS ===
    COLOR_MIN_DIAMETER = 30             # Minimum organoid diameter for color detection (pixels)
    COLOR_MAX_DIAMETER = 110            # Maximum organoid diameter for color detection (pixels)
    COLOR_CIRCULARITY_THRESHOLD = 0.6   # Minimum circularity for color detection (0.0-1.0)
    COLOR_EROSION_STAGES = 9            # Number of erosion stages for color detection (0-9)
    COLOR_GAUSSIAN_BLUR_SIZE = 3        # Gaussian blur kernel size (odd numbers: 3, 5, 7, etc.)
    COLOR_CANNY_LOW_THRESHOLD = 50      # Canny edge detection low threshold (0-255)
    COLOR_CANNY_HIGH_THRESHOLD = 150    # Canny edge detection high threshold (0-255)
    
    # === CENTROID MERGING PARAMETERS ===
    CENTROID_MERGE_THRESHOLD = 24       # Distance threshold for merging close centroids (pixels)
    
    # === FREQUENCY FILTERING PARAMETERS ===
    FREQUENCY_FILTER_PERCENTAGE = 15    # Percentage of outliers to remove from each edge (0-50)
    
    # === SAMPLE COLLECTION PARAMETERS ===
    SAMPLE_RADIUS = 15                  # Default radius for color sample areas (pixels)
    MIN_SAMPLES_REQUIRED = 3            # Minimum number of samples needed for color detection
    
    # === WELL BOUNDARY DETECTION PARAMETERS ===
    WELL_BOUNDARY_RADIUS = 45           # Expected well radius for boundary detection (pixels)
    WELL_BOUNDARY_TOLERANCE = 30        # Tolerance for well boundary detection (pixels)
    
    # === VISUALIZATION PARAMETERS ===
    VISUALIZATION_MAX_WELLS = 16        # Maximum number of wells to show in visualization
    VISUALIZATION_COLS = 4              # Number of columns in visualization grid
    ORGANOID_MARKER_SIZE = 8            # Size of organoid markers in visualization
    WELL_BBOX_EXPANSION = 0.07           # Expansion factor for well bounding boxes (10% = 0.1)
    SHOW_DISPLAY_GRAPHS = True          # Whether to show matplotlib display graphs
    
    # === FILE PARAMETERS ===
    COLOR_PALETTE_FILENAME = "color_palette_save.json"  # Filename for saved color palettes
    WELL_CROPS_FOLDER = "well_crops"    # Folder name for well crop outputs
    
    # === DISPLAY PARAMETERS ===
    DISPLAY_ZOOM_FACTOR = 0.3           # Zoom factor for display images (0.1-1.0)
    DISPLAY_WINDOW_SIZE = (1200, 800)   # Display window size (width, height)

    ENABLE_INTERACTIVE_FINAL_CHECK = True # Whether to enable interactive final check   
    
    @classmethod
    def get_parameter_info(cls):
        """Return a dictionary with parameter descriptions"""
        return {
            # Binary Detection
            'BINARY_DARK_THRESHOLD': 'Threshold for identifying dark areas to inpaint (lower = more sensitive)',
            'BINARY_INPAINT_RADIUS': 'Radius for filling dark spots (larger = smoother inpainting)',
            'BINARY_THRESHOLD': 'Final binarization threshold (higher = stricter binary detection)',
            'BINARY_MIN_DIAMETER': 'Smallest organoid size to detect in binary mode',
            'BINARY_MAX_DIAMETER': 'Largest organoid size to detect in binary mode',
            'BINARY_CIRCULARITY_THRESHOLD': 'How circular organoids must be (1.0 = perfect circle)',
            
            # Color Detection
            'COLOR_MIN_DIAMETER': 'Smallest organoid size to detect in color mode',
            'COLOR_MAX_DIAMETER': 'Largest organoid size to detect in color mode', 
            'COLOR_CIRCULARITY_THRESHOLD': 'Circularity requirement for color detection',
            'COLOR_GAUSSIAN_BLUR_SIZE': 'Blur amount before edge detection (reduces noise)',
            'COLOR_CANNY_LOW_THRESHOLD': 'Lower edge detection sensitivity',
            'COLOR_CANNY_HIGH_THRESHOLD': 'Upper edge detection sensitivity',
            
            # Processing
            'CENTROID_MERGE_THRESHOLD': 'Distance to merge nearby detections (avoids duplicates)',
            'FREQUENCY_FILTER_PERCENTAGE': 'Percentage of color outliers to remove',
            'SAMPLE_RADIUS': 'Size of color sampling areas',
            'WELL_BBOX_EXPANSION': 'How much to expand well bounding boxes for visualization',
            'SHOW_DISPLAY_GRAPHS': 'Whether to show matplotlib display graphs'
        }

# ============================================================================
# INTERACTIVE WELL EDITOR CLASS
# ============================================================================

class InteractiveWellEditor:
    """Interactive editor for adjusting organoid positions in individual wells"""
    
    def __init__(self, well_image, well_centroid, well_bbox, organoid_centroids, detection_types, well_label):
        """
        Initialize the interactive editor
        
        Args:
            well_image: The well crop image
            well_centroid: (x, y) coordinates of well center in crop coordinates
            well_bbox: (x1, y1, x2, y2) bounding box of well in crop coordinates
            organoid_centroids: List of (x, y) organoid coordinates in crop coordinates
            detection_types: List of detection types ('binary' or 'color')
            well_label: Label of the well (e.g., 'A1')
        """
        self.well_image = well_image.copy()
        self.original_image = well_image.copy()
        self.well_centroid = well_centroid
        self.well_bbox = well_bbox
        self.organoid_centroids = organoid_centroids.copy()
        self.original_centroids = organoid_centroids.copy()
        self.detection_types = detection_types.copy()
        self.well_label = well_label
        
        # Interactive state
        self.selected_organoid = None
        self.dragging = False
        self.drag_start = None
        self.adding_mode = False
        
        # Display parameters
        self.marker_size = 8
        self.well_centroid_color = (255, 0, 255)  # Magenta
        self.binary_color = (0, 0, 255)  # Red
        self.color_color = (0, 255, 0)  # Green
        self.bbox_color = (255, 255, 0)  # Yellow
        
        # Setup matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Track if changes were made
        self.changes_made = False
    
    def draw_well_components(self):
        """Draw well components on the image"""
        display_img = self.well_image.copy()
        
        # Draw well bounding box if available
        if self.well_bbox and all(coord is not None for coord in self.well_bbox):
            x1, y1, x2, y2 = self.well_bbox
            cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), self.bbox_color, 2)
        
        # Draw well centroid
        if self.well_centroid and all(coord is not None for coord in self.well_centroid):
            cx, cy = int(self.well_centroid[0]), int(self.well_centroid[1])
            cv2.circle(display_img, (cx, cy), 3, self.well_centroid_color, -1)
        
        # Draw organoid centroids
        for i, (ox, oy) in enumerate(self.organoid_centroids):
            ox, oy = int(ox), int(oy)
            color = self.binary_color if i < len(self.detection_types) and self.detection_types[i] == 'binary' else self.color_color
            marker_type = 'o' if i < len(self.detection_types) and self.detection_types[i] == 'binary' else 's'
            
            # Draw marker
            if marker_type == 'o':
                cv2.circle(display_img, (ox, oy), self.marker_size, color, 2)
                cv2.circle(display_img, (ox, oy), 3, color, -1)
            else:
                cv2.rectangle(display_img, (ox-6, oy-6), (ox+6, oy+6), color, 2)
                cv2.rectangle(display_img, (ox-2, oy-2), (ox+2, oy+2), color, -1)
            
            # Highlight selected organoid
            if i == self.selected_organoid:
                cv2.circle(display_img, (ox, oy), self.marker_size + 5, (255, 255, 255), 3)
        
        return display_img
    
    def find_nearest_organoid(self, x, y, threshold=20):
        """Find the nearest organoid to the clicked position"""
        if not self.organoid_centroids:
            return None
            
        distances = [np.sqrt((x - ox)**2 + (y - oy)**2) for ox, oy in self.organoid_centroids]
        min_dist_idx = np.argmin(distances)
        
        if distances[min_dist_idx] <= threshold:
            return min_dist_idx
        return None
    
    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
            
        if event.button == 1:  # Left click
            if self.adding_mode:
                # Add new organoid
                self.organoid_centroids.append((int(event.xdata), int(event.ydata)))
                self.detection_types.append('manual')  # Mark as manually added
                self.changes_made = True
                self.adding_mode = False
                print(f"Added organoid at ({int(event.xdata)}, {int(event.ydata)})")
                self.update_display()
            else:
                # Select existing organoid for dragging
                self.selected_organoid = self.find_nearest_organoid(event.xdata, event.ydata)
                if self.selected_organoid is not None:
                    self.dragging = True
                    self.drag_start = (event.xdata, event.ydata)
                    print(f"Selected organoid {self.selected_organoid}")
                    self.update_display()
        
        elif event.button == 3:  # Right click - delete organoid
            organoid_to_delete = self.find_nearest_organoid(event.xdata, event.ydata)
            if organoid_to_delete is not None:
                print(f"Deleted organoid {organoid_to_delete}")
                del self.organoid_centroids[organoid_to_delete]
                if organoid_to_delete < len(self.detection_types):
                    del self.detection_types[organoid_to_delete]
                self.changes_made = True
                self.selected_organoid = None
                self.update_display()
    
    def on_release(self, event):
        """Handle mouse release events"""
        if event.button == 1:  # Left click
            self.dragging = False
            self.drag_start = None
    
    def on_motion(self, event):
        """Handle mouse motion events"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
            
        if self.dragging and self.selected_organoid is not None:
            # Update organoid position
            old_pos = self.organoid_centroids[self.selected_organoid]
            new_pos = (int(event.xdata), int(event.ydata))
            self.organoid_centroids[self.selected_organoid] = new_pos
            self.changes_made = True
            self.update_display()
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'r' or event.key == 'R':
            # Reset to original positions
            self.organoid_centroids = self.original_centroids.copy()
            self.detection_types = self.detection_types[:len(self.original_centroids)]
            self.selected_organoid = None
            self.dragging = False
            self.adding_mode = False
            self.changes_made = False
            print("Reset to original positions")
            self.update_display()
        elif event.key == 'a' or event.key == 'A':
            # Enter adding mode
            self.adding_mode = True
            self.selected_organoid = None
            print("Adding mode: Click to add new organoid")
            self.update_display()
        elif event.key == 'escape':
            # Skip this well without saving
            print("Skipping well...")
            plt.close(self.fig)
        elif event.key == 'enter':
            # Save changes and close
            print("Saving changes...")
            plt.close(self.fig)
        elif event.key == 'c' or event.key == 'C':
            # Cancel entire operation
            print("Canceling entire analysis...")
            self.cancel_operation = True
            plt.close(self.fig)
        elif event.key == 'd' or event.key == 'D':
            # Delete selected organoid
            if self.selected_organoid is not None:
                print(f"Deleted organoid {self.selected_organoid}")
                del self.organoid_centroids[self.selected_organoid]
                if self.selected_organoid < len(self.detection_types):
                    del self.detection_types[self.selected_organoid]
                self.changes_made = True
                self.selected_organoid = None
                self.update_display()
    
    def update_display(self):
        """Update the display with current organoid positions"""
        display_img = self.draw_well_components()
        self.ax.clear()
        self.ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        
        # Update title with current status
        mode_text = " [ADDING MODE]" if self.adding_mode else ""
        changes_text = " [MODIFIED]" if self.changes_made else ""
        title = f'Well {self.well_label} - {len(self.organoid_centroids)} Organoids{mode_text}{changes_text}'
        self.ax.set_title(title, fontsize=14)
        self.ax.axis('off')
        
        # Add instructions
        instructions = [
            'CONTROLS:',
            '• Left Click: Select/Move organoid',
            '• Right Click: Delete organoid',
            '• A: Add new organoid mode',
            '• D: Delete selected organoid',
            '• R: Reset changes',
            '• ENTER: Save and continue',
            '• ESC: Skip well',
            '• C: Cancel entire analysis'
        ]
        
        instruction_text = '\n'.join(instructions)
        self.ax.text(0.02, 0.98, instruction_text, 
                    transform=self.ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                    fontsize=10, fontfamily='monospace')
        
        # Add centroid info if available
        if self.well_centroid and all(coord is not None for coord in self.well_centroid):
            centroid_text = f'Well Center: ({int(self.well_centroid[0])}, {int(self.well_centroid[1])})'
            self.ax.text(0.98, 0.02, centroid_text,
                        transform=self.ax.transAxes, verticalalignment='bottom',
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                        fontsize=10)
        
        self.fig.canvas.draw()
    
    def run_editor(self):
        """Run the interactive editor and return updated centroids"""
        print(f"\n=== Editing Well {self.well_label} ===")
        print("Instructions:")
        print("  - Click and drag to move organoids")
        print("  - Press 'A' then click to add new organoids")
        print("  - Right-click or press 'D' to delete organoids")
        print("  - Press 'R' to reset changes")
        print("  - Press ENTER to save and continue")
        print("  - Press ESC to skip this well")
        print("  - Press 'C' to cancel entire analysis")
        
        # Initialize cancel flag
        self.cancel_operation = False
        
        self.update_display()
        plt.show()
        
        # Check if operation was cancelled
        if self.cancel_operation:
            return None
        
        # Calculate relative positions to well center
        updated_centroids = []
        
        if self.well_centroid and all(coord is not None for coord in self.well_centroid):
            well_cx, well_cy = self.well_centroid
            
            for i, (ox, oy) in enumerate(self.organoid_centroids):
                # Calculate relative position to well center
                rel_x = ox - well_cx
                rel_y = oy - well_cy
                detection_type = self.detection_types[i] if i < len(self.detection_types) else 'manual'
                
                updated_centroids.append({
                    'relative_pos': (rel_x, rel_y),
                    'absolute_pos': (ox, oy),
                    'detection_type': detection_type,
                    'distance_from_center': np.sqrt(rel_x**2 + rel_y**2)
                })
        else:
            # If no well center, use absolute coordinates
            for i, (ox, oy) in enumerate(self.organoid_centroids):
                detection_type = self.detection_types[i] if i < len(self.detection_types) else 'manual'
                updated_centroids.append({
                    'relative_pos': (ox, oy),
                    'absolute_pos': (ox, oy),
                    'detection_type': detection_type,
                    'distance_from_center': 0
                })
        
        return updated_centroids

# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class WellOrganoidAnalyzer:
    """Comprehensive analyzer for well-plate organoid detection and analysis"""
    
    def __init__(self, show_visualizations=False):
        """
        Initialize the Well Organoid Analyzer
        
        Args:
            show_visualizations (bool): Whether to show intermediate visualization windows
        """
        # Analysis parameters
        self.params = AnalysisParameters()
        self.show_visualizations = show_visualizations
        
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
    
    def run_dual_detection(self):
        """Run both binary and color detection to find organoid centroids using parameter definitions"""
        print("\n--- DUAL DETECTION PHASE ---")
        print(f"📊 Using parameters:")
        print(f"   Binary: diameter {self.params.BINARY_MIN_DIAMETER}-{self.params.BINARY_MAX_DIAMETER}px, circularity ≥{self.params.BINARY_CIRCULARITY_THRESHOLD}")
        print(f"   Color: diameter {self.params.COLOR_MIN_DIAMETER}-{self.params.COLOR_MAX_DIAMETER}px, circularity ≥{self.params.COLOR_CIRCULARITY_THRESHOLD}")
        print(f"   Centroid merge threshold: {self.params.CENTROID_MERGE_THRESHOLD}px")
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # === BINARY DETECTION PROCESSING ===
        print("\n--- BINARY DETECTION PHASE ---")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Create mask with threshold (inverted for inpainting)
        dark_mask = (gray < self.params.BINARY_DARK_THRESHOLD).astype(np.uint8) * 255
        
        # Apply inpainting
        inpainted = cv2.inpaint(self.original_image, dark_mask, inpaintRadius=self.params.BINARY_INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)
        
        # Convert inpainted image to grayscale for binarization
        inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        
        # Apply binarization with threshold
        _, binary_plate = cv2.threshold(inpainted_gray, self.params.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Save binary image with timestamp
        binary_filename = f"binary_plate_{timestamp}.png"
        cv2.imwrite(binary_filename, binary_plate)
        print(f"💾 Binary image saved: {binary_filename}")
        
        # Show binary visualization if requested
        if self.show_visualizations:
            self.show_binary_visualization(binary_plate, inpainted)
        
        # Find centroids from binary detection
        binary_centroids = self.find_binary_centroids(
            binary_plate, 
            self.params.BINARY_MIN_DIAMETER, 
            self.params.BINARY_MAX_DIAMETER, 
            self.params.BINARY_CIRCULARITY_THRESHOLD
        )
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
        
        # Show color visualization if requested
        if self.show_visualizations and len(self.color_centroids) > 0:
            self.show_color_visualization(color_filtered_image if 'color_filtered_image' in locals() else None)
        
        print(f"Total centroids found: {len(self.binary_centroids)} binary + {len(self.color_centroids)} color")

    def show_binary_visualization(self, binary_plate, inpainted):
        """Show binary detection visualization"""
        print("📊 Showing binary detection visualization...")
        
        # Create overlay with detected centroids
        binary_display = cv2.cvtColor(binary_plate, cv2.COLOR_GRAY2BGR)
        for (cx, cy) in self.binary_centroids:
            cv2.circle(binary_display, (cx, cy), 8, (0, 0, 255), 2)  # Red circles
            cv2.circle(binary_display, (cx, cy), 3, (0, 0, 255), -1)  # Red center
        
        # Show side by side: inpainted original vs binary with detections
        inpainted_resized = cv2.resize(inpainted, (800, 600))
        binary_resized = cv2.resize(binary_display, (800, 600))
        combined = np.hstack([inpainted_resized, binary_resized])
        
        # Add titles
        cv2.putText(combined, f"Inpainted Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, f"Binary Detection ({len(self.binary_centroids)} found)", (820, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Binary Detection Results", combined)
        cv2.waitKey(3000)  # Show for 3 seconds
        cv2.destroyAllWindows()

    def show_color_visualization(self, color_filtered_image):
        """Show color detection visualization"""
        print("📊 Showing color detection visualization...")
        
        if color_filtered_image is not None:
            # Create overlay with detected centroids
            color_display = color_filtered_image.copy()
            for (cx, cy) in self.color_centroids:
                cv2.rectangle(color_display, (cx-6, cy-6), (cx+6, cy+6), (0, 255, 0), 2)  # Green squares
                cv2.rectangle(color_display, (cx-2, cy-2), (cx+2, cy+2), (0, 255, 0), -1)  # Green center
            
            # Show side by side: original vs color with detections
            original_resized = cv2.resize(self.original_image, (800, 600))
            color_resized = cv2.resize(color_display, (800, 600))
            combined = np.hstack([original_resized, color_resized])
            
            # Add titles
            cv2.putText(combined, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, f"Color Detection ({len(self.color_centroids)} found)", (820, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Color Detection Results", combined)
            cv2.waitKey(3000)  # Show for 3 seconds
            cv2.destroyAllWindows()

    def find_binary_centroids(self, image, min_diameter=None, max_diameter=None, circularity_threshold=None):
        """Find centroids from binary detection (inverted image)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Invert the image for binary detection
        gray = cv2.bitwise_not(gray)
        
        kernel = np.ones((3, 3), np.uint8)
        all_centroids = []

        for i in range(self.params.BINARY_EROSION_STAGES):  # 0 to N erosions
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
        merged_centroids = self.merge_close_centroids(all_centroids, threshold=self.params.CENTROID_MERGE_THRESHOLD)
        print(f"Binary detection - After merging: {len(merged_centroids)} centroids")

        return merged_centroids

    def create_image_with_organoids(self):
        """Create a copy of the original image with organoids plotted on it"""
        print("Creating image with plotted organoids...")
        
        # Create a copy of the original image
        self.image_with_organoids = self.original_image.copy()
        
        # Plot binary centroids (red circles)
        for (cx, cy) in self.binary_centroids:
            cv2.circle(self.image_with_organoids, (cx, cy), self.params.ORGANOID_MARKER_SIZE, (0, 0, 255), 2)  # Red
            cv2.circle(self.image_with_organoids, (cx, cy), 3, (0, 0, 255), -1)  # Red center
        
        # Plot color centroids (green squares)
        marker_size = self.params.ORGANOID_MARKER_SIZE
        for (cx, cy) in self.color_centroids:
            # Draw square by drawing rectangle
            cv2.rectangle(self.image_with_organoids, (cx-marker_size//2, cy-marker_size//2), 
                         (cx+marker_size//2, cy+marker_size//2), (0, 255, 0), 2)  # Green
            cv2.rectangle(self.image_with_organoids, (cx-2, cy-2), (cx+2, cy+2), (0, 255, 0), -1)  # Green center
        
        # Save the image with organoids
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_with_organoids_path = f"image_with_organoids_{timestamp}.png"
        cv2.imwrite(self.image_with_organoids_path, self.image_with_organoids)
        print(f"Saved image with organoids: {self.image_with_organoids_path}")

    def add_well_boundaries_to_image(self, image, well_crops):
        """Add well bounding boxes to an image with 10% expansion"""
        result_image = image.copy()
        
        for well in well_crops:
            x, y, w, h = well['bbox']
            grid_label = well['grid_label']
            
            # Expand bounding box by 10%
            expansion = self.params.WELL_BBOX_EXPANSION
            expand_w = int(w * expansion)
            expand_h = int(h * expansion)
            
            # Calculate expanded coordinates
            x1 = max(0, x - expand_w // 2)
            y1 = max(0, y - expand_h // 2)
            x2 = min(image.shape[1] - 1, x + w + expand_w // 2)
            y2 = min(image.shape[0] - 1, y + h + expand_h // 2)
            
            # Draw expanded bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Magenta boxes
            
            # Add well label
            label_x = x1 + 5
            label_y = y1 + 25
            cv2.putText(result_image, grid_label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_image, grid_label, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 1)
        
        return result_image
    
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
        
        # Create and save an image with well boundaries marked
        print("Creating image with well boundaries...")
        image_with_wells = self.add_well_boundaries_to_image(self.image_with_organoids, self.well_crops)
        wells_filename = f"image_with_wells_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(wells_filename, image_with_wells)
        print(f"💾 Image with well boundaries saved: {wells_filename}")
        
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
                    radius=self.params.WELL_BOUNDARY_RADIUS, 
                    tolerance=self.params.WELL_BOUNDARY_TOLERANCE, 
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
    
    def run_interactive_well_editing(self, output_dir):
        """Run interactive editing for each well with organoids"""
        print("\n--- INTERACTIVE WELL EDITING ---")
        
        # Process ALL wells, not just those with existing organoids
        wells_to_edit = self.well_crops.copy()
        
        print(f"Starting interactive editing for {len(wells_to_edit)} wells...")
        
        for i, well in enumerate(wells_to_edit):
            grid_label = well['grid_label']
            print(f"\n--- Editing Well {grid_label} ({i+1}/{len(wells_to_edit)}) ---")
            
            # Load well image
            well_image_path = Path(output_dir) / f"Color_well_{grid_label}.png"
            if not well_image_path.exists():
                print(f"  Well image not found: {well_image_path}")
                continue
            
            well_image = cv2.imread(str(well_image_path))
            if well_image is None:
                print(f"  Could not load well image: {well_image_path}")
                continue
            
            # Get well boundary info
            well_result = self.well_boundary_results.get(grid_label)
            if well_result and well_result['success']:
                well_centroid = (well_result['centroid_x'], well_result['centroid_y'])
                well_bbox = (well_result['bbox_x1'], well_result['bbox_y1'], 
                           well_result['bbox_x2'], well_result['bbox_y2'])
            else:
                # Use well image center as fallback
                h, w = well_image.shape[:2]
                well_centroid = (w//2, h//2)
                well_bbox = (0, 0, w, h)
            
            # Prepare organoid data for editor (convert to crop-relative coordinates)
            organoid_centroids = []
            detection_types = []
            
            for organoid in well['organoid_centroids']:
                rel_x, rel_y = organoid['relative_pos']
                # Convert relative position to absolute coordinates in the well crop
                # Since relative_pos is relative to the well bbox corner, we add them directly
                abs_x = rel_x
                abs_y = rel_y
                organoid_centroids.append((abs_x, abs_y))
                detection_types.append(organoid['detection_type'])
            
            # Run interactive editor
            editor = InteractiveWellEditor(
                well_image, well_centroid, well_bbox, 
                organoid_centroids, detection_types, grid_label
            )
            
            updated_centroids = editor.run_editor()
            
            # Check if operation was cancelled
            if updated_centroids is None:
                print("Analysis cancelled by user.")
                return False
            
            # Update well data with new coordinates
            well['organoid_centroids'] = []  # Clear existing data
            
            for centroid_data in updated_centroids:
                rel_x, rel_y = centroid_data['relative_pos']
                abs_x, abs_y = centroid_data['absolute_pos']
                detection_type = centroid_data['detection_type']
                distance = centroid_data['distance_from_center']
                
                well['organoid_centroids'].append({
                    'relative_pos': (rel_x, rel_y),
                    'absolute_pos': (abs_x, abs_y),
                    'detection_type': detection_type,
                    'distance_from_center': distance
                })
            
            print(f"  Updated well with {len(updated_centroids)} organoids")
        
        return True
    
    def save_results_csv(self):
        """Save comprehensive results to CSV with coordinates relative to well center"""
        print("\n--- SAVING RESULTS ---")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"well_organoid_analysis_{timestamp}.csv"
        
        headers = [
            'Well_ID', 'Grid_Label', 'Well_Detected', 'Well_Center_X', 'Well_Center_Y',
            'Organoid_Count', 'Organoid_Relative_X', 'Organoid_Relative_Y', 'Detection_Type', 'Distance_From_Center'
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
            
            # Organoid info - now relative to well center
            organoid_count = len(well['organoid_centroids'])
            
            if organoid_count > 0:
                # Create separate rows for each organoid
                for j, organoid in enumerate(well['organoid_centroids']):
                    rel_x, rel_y = organoid['relative_pos']
                    detection_type = organoid['detection_type']
                    distance = organoid.get('distance_from_center', 0)
                    
                    results.append([
                        i, grid_label, well_detected, well_center_x, well_center_y,
                        organoid_count, rel_x, rel_y, detection_type, distance
                    ])
            else:
                # Add row for wells with no organoids
                results.append([
                    i, grid_label, well_detected, well_center_x, well_center_y,
                    0, '', '', '', ''
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
        wells_with_data = [w for w in self.well_crops if len(w['organoid_centroids']) > 0 or 
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
        
        # Step 6: Interactive well editing
        if self.params.ENABLE_INTERACTIVE_FINAL_CHECK:
            editing_success = self.run_interactive_well_editing(output_dir)
            if not editing_success:
                print("Analysis cancelled by user.")
                return
        
        # Step 7: Save results (after interactive editing)
        csv_file = self.save_results_csv()
        
        # Step 8: Create final visualization (if requested)
        viz_file = None
        if self.params.SHOW_DISPLAY_GRAPHS:
            viz_file = self.create_final_visualization(output_dir)
        
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
        if viz_file:
            print(f"Visualization: {viz_file}")

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
            cv2.circle(self.display_img, center, radius, color, -1)
        
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

    def find_circular_contours_with_centroids(self, image, min_diameter=None, max_diameter=None, circularity_threshold=None):
        """Find circular contours and return centroids using parameter definitions"""
        # Use parameter defaults if not provided
        min_diameter = min_diameter or self.params.COLOR_MIN_DIAMETER
        max_diameter = max_diameter or self.params.COLOR_MAX_DIAMETER
        circularity_threshold = circularity_threshold or self.params.COLOR_CIRCULARITY_THRESHOLD
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        kernel = np.ones((3, 3), np.uint8)
        all_centroids = []

        for i in range(self.params.COLOR_EROSION_STAGES):  # 0 to N erosions
            eroded = gray.copy() if i == 0 else cv2.erode(gray.copy(), kernel, iterations=i)
            blur_size = self.params.COLOR_GAUSSIAN_BLUR_SIZE
            blurred = cv2.GaussianBlur(eroded, (blur_size, blur_size), 0)
            edged = cv2.Canny(blurred, self.params.COLOR_CANNY_LOW_THRESHOLD, self.params.COLOR_CANNY_HIGH_THRESHOLD)
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
        print(f"Color detection - Before merging: {len(all_centroids)} centroids")
        merged_centroids = self.merge_close_centroids(all_centroids, threshold=self.params.CENTROID_MERGE_THRESHOLD)
        print(f"Color detection - After merging: {len(merged_centroids)} centroids")

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
    import argparse
    
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Well Organoid Analyzer - Detect organoids in well plate images")
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='Show intermediate visualization windows during processing')
    parser.add_argument('--parameters', '-p', action='store_true',
                       help='Show parameter definitions and exit')
    
    args = parser.parse_args()
    
    # Show parameters if requested
    if args.parameters:
        print("=== WELL ORGANOID ANALYZER PARAMETERS ===")
        print()
        param_info = AnalysisParameters.get_parameter_info()
        
        # Group parameters by category
        categories = {
            'Binary Detection': [k for k in param_info.keys() if k.startswith('BINARY_')],
            'Color Detection': [k for k in param_info.keys() if k.startswith('COLOR_')],
            'Processing': [k for k in param_info.keys() if not k.startswith(('BINARY_', 'COLOR_'))]
        }
        
        for category, params in categories.items():
            print(f"=== {category.upper()} ===")
            for param in params:
                value = getattr(AnalysisParameters, param, 'N/A')
                description = param_info.get(param, 'No description available')
                print(f"  {param}: {value}")
                print(f"    → {description}")
                print()
        return
    
    # Run analyzer
    print("=== WELL ORGANOID ANALYZER ===")
    if args.visualize:
        print("🔍 Visualization mode enabled - intermediate results will be shown")
    else:
        print("⚡ Fast mode - no intermediate visualizations")
    
    analyzer = WellOrganoidAnalyzer(show_visualizations=args.visualize)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 