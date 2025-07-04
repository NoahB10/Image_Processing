import numpy as np
import matplotlib.pyplot as plt
import discorpy.losa.loadersaver as losa
import discorpy.prep.preprocessing as prep
import discorpy.proc.processing as proc
import discorpy.post.postprocessing as post
import discorpy.util.utility as util
import rawpy
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from matplotlib.path import Path

# Single Camera Fisheye Distortion Correction Script V4
# Processes DNG files from a single camera
# HAS POLYGON MASKING FUNCTION BUILT IN
# Uses PROPER FISHEYE DOT PATTERN analysis following Discorpy_Fisheye_Example.py
# SAVES BOTH RADIAL AND PERSPECTIVE COEFFICIENTS SEPARATELY (like V3)
# Based on: Discorpy_Fisheye_Example.py workflow for dot patterns

class SingleCameraFisheyeDistortionCorrectionV4:
    def __init__(self):
        # Cropping parameters (from GUI_IMX708_Dirsotion_Correction_v1.1.py)
        self.crop_params = {
            'cam0': {'width': 3950, 'start_x': 0, 'height': 3496}  # Single camera
        }
        
        # Processing parameters for PROPER FISHEYE DOT PATTERN analysis
        self.num_coef = 3  # Number of polynomial coefficients for fisheye
        self.sigma_normalization = 8  # FFT normalization (10 like the example)
        
        # Dot pattern parameters (following Discorpy_Fisheye_Example.py)
        self.dot_pattern_params = {
            'cam0': {
                'binarization_ratio': 0.8,  # Ratio for binarization
                'size_distance_ratio': 0.3,  # Ratio for size/distance calculation
                'slope_ratio': 0.3           # Ratio for slope calculation
            }
        }

        # Dot parameters (set to None to use calc_size_distance, or specify values)
        self.dot_parameters = {
            'cam0': {
                'dot_size': 30.0,  # Set to specific value or None to auto-calculate
                'dot_dist': None   # Set to specific value or None to auto-calculate
            }
        }
        
        # Polygon masking parameters (like V1)
        self.mask_params = {
            'cam0': {
                'polygon_verts': [
                    (167.7, 104.7),
                    (411.7, 54.2),
                    (630.6, 20.5),
                    (916.7, 12.1),
                    (1282.8, 12.1),
                    (1535.2, 33.2),
                    (1766.7, 66.8),
                    (2065.4, 125.7),
                    (2065.4, 2334.8),
                    (1425.8, 2452.6),
                    (1021.9, 2490.5),
                    (626.3, 2482.1),
                    (247.6, 2427.4),
                    (180.3, 2086.6),
                    (121.4, 1707.9),
                    (96.2, 1354.4),
                    (87.7, 1009.4),
                    (104.6, 693.8),
                    (138.2, 403.4),
                    (176.1, 108.9)
                    ]
            }
        }
        
        # Grouping parameters (following Discorpy_Fisheye_Example.py exactly)
        self.grouping_params = {
            'cam0': {
                'ratio': 0.4,               # Grouping tolerance ratio (from example)
                'num_dot_miss': 3,          # Number of missing dots allowed
                'accepted_ratio': 0.65,     # Acceptance ratio for grouping
                'order': 2,                 # Polynomial order for polyfit
                'residual_threshold': 20.0   # Residual threshold (from example)
            }
        }
        
        # Fisheye-specific parameters (following the example)
        self.fisheye_params = {
            'vanishing_point_iterations': 2,  # Iterations for center finding
            'enable_perspective_correction': True,  # Apply perspective correction
            'padding': 400  # Padding for unwarping
        }
        
        # Perspective correction parameters (like V3)
        self.perspective_params = {
            'equal_dist': True,     # Equal distance for perspective correction
            'scale': 'mean',        # Scale method ('mean' or other)
            'optimizing': False     # Optimizing parameter
        }
        
        # Toggle options
        self.apply_masking = 1  # Set to 1 to enable polygon masking
        self.interactive_polygon_drawing = 0  # Set to 1 to enable interactive polygon drawing
        self.test_images = 0  # Set to 1 to test images
        self.debug_plots = 1  # Set to 1 to enable debug plots
        self.save_intermediate = 1  # Set to 1 to save intermediate images
        self.save_perspective_coefficients = 1  # Set to 1 to save separate perspective coefficients
        
        # Results storage
        self.results = {
            'cam0': {'xcenter': None, 'ycenter': None, 'coeffs': None, 'pers_coef': None}
        }
        
        # Store processed data for comparison
        self.processed_data = {
            'cam0': {}
        }
    
    def load_dng_image(self, filepath, to_grayscale=True):
        """Load image file using rawpy for DNG files or losa.load_image for other formats"""
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # Try to load as DNG/RAW file first if it has a RAW extension
        if file_ext in ['.dng', '.cr2', '.nef', '.arw', '.orf', '.raf', '.rw2']:
            try:
                print(f"Loading DNG/RAW file: {os.path.basename(filepath)}")
                raw = rawpy.imread(filepath)
                rgb = raw.postprocess(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=False,
                    output_bps=16,
                    output_color=rawpy.ColorSpace.sRGB,
                )
                
                if to_grayscale:
                    # Convert to grayscale for distortion analysis
                    gray = np.mean(rgb, axis=2).astype(np.uint16)
                    print(f"   Converted to grayscale: {gray.shape}, range: [{gray.min()}, {gray.max()}]")
                    return gray, rgb
                else:
                    print(f"   Loaded RGB: {rgb.shape}, range: [{rgb.min()}, {rgb.max()}]")
                    return rgb, rgb
                    
            except Exception as e:
                print(f"[WARNING] Failed to load as RAW file {filepath}: {e}")
                print("[INFO] Attempting to load as regular image file...")
        
        # Fallback: load as regular image file (JPEG, PNG, TIFF, etc.) using losa
        try:
            print(f"Loading image file: {os.path.basename(filepath)}")
            # Load image using losa (can handle various formats)
            rgb = losa.load_image(filepath, average=False)
            
            # Handle different image formats and bit depths
            if rgb.ndim == 2:
                # Grayscale image
                if to_grayscale:
                    gray = rgb.astype(np.uint16) if rgb.dtype != np.uint16 else rgb
                    print(f"   Loaded grayscale: {gray.shape}, range: [{gray.min()}, {gray.max()}]")
                    return gray, np.stack([rgb, rgb, rgb], axis=2)  # Create RGB version for compatibility
                else:
                    rgb_version = np.stack([rgb, rgb, rgb], axis=2)
                    print(f"   Loaded grayscale as RGB: {rgb_version.shape}, range: [{rgb_version.min()}, {rgb_version.max()}]")
                    return rgb_version, rgb_version
            elif rgb.ndim == 3:
                # Color image
                # Ensure consistent data type
                if rgb.dtype == np.uint8:
                    # Convert 8-bit to 16-bit for consistency with DNG processing
                    rgb = (rgb.astype(np.uint16) * 256)
                elif rgb.dtype == np.float32 or rgb.dtype == np.float64:
                    # Normalize float to 16-bit
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 65535).astype(np.uint16)
                    else:
                        rgb = (rgb / rgb.max() * 65535).astype(np.uint16)
                
                if to_grayscale:
                    # Convert to grayscale for distortion analysis
                    gray = np.mean(rgb, axis=2).astype(np.uint16)
                    print(f"   Converted to grayscale: {gray.shape}, range: [{gray.min()}, {gray.max()}]")
                    return gray, rgb
                else:
                    print(f"   Loaded RGB: {rgb.shape}, range: [{rgb.min()}, {rgb.max()}]")
                    return rgb, rgb
            else:
                print(f"[ERROR] Unsupported image dimensions: {rgb.shape}")
                return None, None
                
        except Exception as e:
            print(f"[ERROR] Failed to load image file {filepath}: {e}")
            return None, None
    
    def crop_image(self, image, cam_name):
        """Crop image according to camera-specific parameters"""
        params = self.crop_params[cam_name]
        start_x = params['start_x']
        width = params['width']
        height = params['height']
        
        # Crop the image: [y_start:y_end, x_start:x_end]
        cropped = image[:height, start_x:start_x + width]
        print(f"   Cropped {cam_name}: {image.shape} -> {cropped.shape}")
        return cropped
    
   
    def create_polygon_mask(self, image, cam_name):
        """
        Create polygon mask for the image using V1-style polygon masking
        
        Parameters
        ----------
        image : array_like
            2D array of the image
        cam_name : str
            Camera name ('cam0' or 'cam1')
            
        Returns
        -------
        mask : array_like
            2D boolean mask array
        """
        height, width = image.shape
        polygon_verts = self.mask_params[cam_name]['polygon_verts']
        
        # Create mesh grid for coordinates
        xv, yv = np.meshgrid(np.arange(width), np.arange(height))
        coords = np.vstack((xv.flatten(), yv.flatten())).T
        
        # Create path from polygon vertices
        path = Path(polygon_verts)
        mask_flat = path.contains_points(coords)
        mask = mask_flat.reshape((height, width))
        
        print(f"      {cam_name}: Created polygon mask with {len(polygon_verts)} vertices")
        return mask

    def apply_polygon_mask_to_points(self, points, image, cam_name):
        """
        Apply polygon mask to filter points (V1 style)
        
        Parameters
        ----------
        points : array_like
            Array of (y, x) coordinates
        image : array_like
            2D array of the image
        cam_name : str
            Camera name ('cam0' or 'cam1')
            
        Returns
        -------
        filtered_points : array_like
            Filtered points within the polygon mask
        """
        mask = self.create_polygon_mask(image, cam_name)
        
        # Convert points to integers for indexing
        y_coords = np.clip(np.round(points[:, 0]).astype(int), 0, mask.shape[0] - 1)
        x_coords = np.clip(np.round(points[:, 1]).astype(int), 0, mask.shape[1] - 1)
        
        # Filter points using mask
        valid_indices = mask[y_coords, x_coords]
        filtered_points = points[valid_indices]
        
        print(f"      {cam_name}: Polygon mask filtered {len(points)} -> {len(filtered_points)} points")
        return filtered_points

    def save_mask_visualization(self, image, cam_name, output_dir):
        """
        Save mask visualization for debugging (V1 style)
        """
        if self.apply_masking:
            mask = self.create_polygon_mask(image, cam_name)
            
            # Create visualization
            masked_image = image.copy().astype(np.float32)
            masked_image[~mask] *= 0.3  # Darken areas outside mask
            
            # Save mask and visualization
            mask_path = f"{output_dir}/polygon_mask_{cam_name}.png"
            mask_vis_path = f"{output_dir}/polygon_mask_{cam_name}_visualization.png"
            
            losa.save_image(mask_path, (mask * 255).astype(np.uint8))
            losa.save_image(mask_vis_path, masked_image.astype(np.uint8))
            
            print(f"      {cam_name}: Saved mask visualization to {mask_vis_path}")
            return mask
        return None

    def draw_interactive_polygon(self, image, cam_name, title="Draw Polygon Mask"):
        """
        Interactive polygon drawing for mask creation (V1 style)
        
        Parameters
        ----------
        image : array_like
            2D array of the image to draw polygon on
        cam_name : str
            Camera name ('cam0' or 'cam1')
        title : str
            Title for the interactive window
            
        Returns
        -------
        polygon_verts : list
            List of (x, y) polygon vertices
        """
        print(f"   Drawing interactive polygon mask for {cam_name}...")
        
        class PolygonDrawer:
            def __init__(self, image, title):
                self.image = image
                self.polygon_points = []
                self.is_drawing = False
                self.fig, self.ax = plt.subplots(figsize=(12, 8))
                self.ax.imshow(image, cmap='gray')
                self.ax.set_title(f'{title}\nLeft click to add points, Right click to finish, "r" to reset')
                self.ax.axis('on')
                
                # Connect events
                self.fig.canvas.mpl_connect('button_press_event', self.on_click)
                self.fig.canvas.mpl_connect('key_press_event', self.on_key)
                
                self.line, = self.ax.plot([], [], 'r-o', linewidth=2, markersize=6)
                self.polygon_complete = False
                
            def on_click(self, event):
                if event.inaxes != self.ax:
                    return
                    
                if event.button == 1:  # Left click - add point
                    self.polygon_points.append((event.xdata, event.ydata))
                    self.update_plot()
                elif event.button == 3:  # Right click - finish polygon
                    if len(self.polygon_points) >= 3:
                        self.finish_polygon()
                    
            def on_key(self, event):
                if event.key == 'r':  # Reset
                    self.polygon_points = []
                    self.line.set_data([], [])
                    self.ax.set_title(f'{self.ax.get_title().split("\\n")[0]}\\nLeft click to add points, Right click to finish, "r" to reset')
                    self.fig.canvas.draw()
                    
            def update_plot(self):
                if self.polygon_points:
                    x_coords = [p[0] for p in self.polygon_points]
                    y_coords = [p[1] for p in self.polygon_points]
                    self.line.set_data(x_coords, y_coords)
                    self.fig.canvas.draw()
                    
            def finish_polygon(self):
                if len(self.polygon_points) >= 3:
                    # Close the polygon by connecting to first point
                    x_coords = [p[0] for p in self.polygon_points] + [self.polygon_points[0][0]]
                    y_coords = [p[1] for p in self.polygon_points] + [self.polygon_points[0][1]]
                    self.line.set_data(x_coords, y_coords)

                    self.ax.set_title(f'{self.ax.get_title().split("\\n")[0]}\nPolygon created with {len(self.polygon_points)} vertices! Close window to continue.')
                    self.fig.canvas.draw()
                    self.polygon_complete = True

                    # Print in desired format
                    print("'polygon_verts': [")
                    for x, y in self.polygon_points:
                        print(f"    ({x:.1f}, {y:.1f}),")
                    print("]")

                    
        # Create the polygon drawer
        drawer = PolygonDrawer(image, title)
        plt.show()
        
        if drawer.polygon_complete and len(drawer.polygon_points) >= 3:
            print(f"   {cam_name}: Created polygon with {len(drawer.polygon_points)} vertices")
            return drawer.polygon_points
        else:
            print(f"   {cam_name}: No polygon created, using default")
            return self.mask_params[cam_name]['polygon_verts']

    def save_jpeg_from_array(self, image_array, output_path, quality=95):
        """Save numpy array as JPEG"""
        try:
            # Normalize to 8-bit if needed
            if image_array.dtype == np.uint16:
                image_8bit = (image_array / 256).astype(np.uint8)
            elif image_array.dtype == np.float32 or image_array.dtype == np.float64:
                if image_array.max() <= 1.0:
                    image_8bit = (image_array * 255).astype(np.uint8)
                else:
                    image_8bit = np.clip(image_array / image_array.max() * 255, 0, 255).astype(np.uint8)
            else:
                image_8bit = np.clip(image_array, 0, 255).astype(np.uint8)
            
            losa.save_image(output_path, image_8bit)
            print(f"   Saved JPEG: {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save JPEG {output_path}: {e}")
            return False

    def process_fisheye_dot_pattern(self, image, cam_name, output_dir):
        """Process fisheye dot pattern following Discorpy_Fisheye_Example.py exactly"""
        print(f"      Processing {cam_name} using proper fisheye dot pattern workflow...")
        
        height, width = image.shape
        dot_params = self.dot_pattern_params[cam_name]
        mask_params = self.mask_params[cam_name]
        group_params = self.grouping_params[cam_name]
        config = self.dot_parameters[cam_name]
        
        # Step 1: Background normalization (following the example)
        print(f"      Step 1: FFT normalization for {cam_name}...")
        img_norm = prep.normalization_fft(image, sigma=self.sigma_normalization)
        
        if self.save_intermediate:
            losa.save_image(f"{output_dir}/01_normalized.jpg", img_norm)
        
        # Step 2: Binarization (following the example)
        print(f"      Step 2: Binarization for {cam_name}...")
        mat1 = prep.binarization(img_norm, ratio=dot_params['binarization_ratio'])
        
        if self.save_intermediate:
            losa.save_image(f"{output_dir}/02_binarized.jpg", mat1)
            
        # Save mask visualization if masking is enabled
        if self.apply_masking and self.save_intermediate:
            self.save_mask_visualization(image, cam_name, output_dir)
        
        # Step 3: Calculate dot size and distance (following the example)
        
        # Try to get configured values first
        dot_size = config.get('dot_size')
        dot_dist = config.get('dot_dist')
        
        # If either is None, calculate them
        if dot_size is None or dot_dist is None:
            try:
                print(f"      Step 3: Calculate dot parameters for {cam_name}...")
                calc_dot_size, calc_dot_dist = prep.calc_size_distance(mat1, ratio=dot_params['size_distance_ratio'])
                if dot_size is None:
                    dot_size = calc_dot_size
                if dot_dist is None:
                    dot_dist = calc_dot_dist
                print(f"   {cam_name}: Calculated dot_size={calc_dot_size:.1f}, dot_dist={calc_dot_dist:.1f}")
            except Exception as e:
                print(f"   {cam_name}: Failed to calculate dot parameters, using defaults: {e}")
                if dot_size is None:
                    dot_size = 70  # Default from demo_05.py
                if dot_dist is None:
                    dot_dist = 162  # Default from demo_05.py
 

        # Step 4: Calculate slopes (following the example)
        print(f"      Step 4: Calculate slopes for {cam_name}...")
        slope_hor = prep.calc_hor_slope(mat1, ratio=dot_params['slope_ratio'])
        slope_ver = prep.calc_ver_slope(mat1, ratio=dot_params['slope_ratio'])
        dist_hor = dist_ver = dot_dist
        
        print(f"      {cam_name}: slope_hor={slope_hor:.4f}, slope_ver={slope_ver:.4f}")
        print(f"      {cam_name}: dist_hor={dist_hor:.1f}, dist_ver={dist_ver:.1f}")
        
        # Step 5: Extract ALL reference points (modified to get all dot pixels)
        print(f"      Step 5: Extract ALL dot points for {cam_name}...")
        list_points = prep.get_points_dot_pattern(mat1, binarize=False)  # Get ALL dot pixels, not just centroids
        list_points_hor_lines = list_points
        list_points_ver_lines = np.copy(list_points)
        
        print(f"      {cam_name}: Found {len(list_points)} dot points")
        
        # Step 6: Apply polygon masking (V1 style)
        if self.apply_masking:
            print(f"      Step 6: Apply polygon masking for {cam_name} (V1 style)...")
            
            # Option to draw new polygon interactively
            if self.interactive_polygon_drawing:
                new_polygon = self.draw_interactive_polygon(image, cam_name, f"Draw Polygon Mask for {cam_name}")
                # Update the polygon vertices for this camera
                self.mask_params[cam_name]['polygon_verts'] = new_polygon
            
            # Apply polygon mask to filter points
            list_points_filtered = self.apply_polygon_mask_to_points(list_points, image, cam_name)
            list_points_hor_lines = list_points_filtered
            list_points_ver_lines = np.copy(list_points_filtered)
            
            print(f"      {cam_name}: After polygon masking - {len(list_points_filtered)} points remain")
        else:
            list_points_hor_lines = list_points
            list_points_ver_lines = np.copy(list_points)
            print(f"      {cam_name}: No masking applied - using all {len(list_points)} points")
        
        # Step 7: Group points into lines (following the example with polyfit)
        print(f"      Step 7: Group points into lines for {cam_name}...")
        
        list_hor_lines = prep.group_dots_hor_lines_based_polyfit(
            list_points_hor_lines, slope_hor, dist_hor,
            ratio=group_params['ratio'],
            num_dot_miss=group_params['num_dot_miss'],
            accepted_ratio=group_params['accepted_ratio'],
            order=group_params['order'])
        
        list_ver_lines = prep.group_dots_ver_lines_based_polyfit(
            list_points_ver_lines, slope_ver, dist_ver,
            ratio=group_params['ratio'],
            num_dot_miss=group_params['num_dot_miss'],
            accepted_ratio=group_params['accepted_ratio'],
            order=group_params['order'])
        
        print(f"      {cam_name}: Grouped into {len(list_hor_lines)} hor lines, {len(list_ver_lines)} ver lines")
        
        # Step 8: Remove residual dots (following the example)
        print(f"      Step 8: Remove residual dots for {cam_name}...")
        
        list_hor_lines = prep.remove_residual_dots_hor(list_hor_lines, slope_hor, 
                                                       group_params['residual_threshold'])
        list_ver_lines = prep.remove_residual_dots_ver(list_ver_lines, slope_ver, 
                                                       group_params['residual_threshold'])
        
        print(f"      {cam_name}: After residual removal - {len(list_hor_lines)} hor lines, {len(list_ver_lines)} ver lines")
        
        # Debug visualization
        if self.debug_plots:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Normalized image with points
            ax1.imshow(img_norm, cmap='gray', origin='lower')
            if self.apply_masking:
                ax1.plot(list_points_hor_lines[:, 1], list_points_hor_lines[:, 0], ".", color="red", markersize=1)
                ax1.plot(list_points_ver_lines[:, 1], list_points_ver_lines[:, 0], ".", color="blue", markersize=1)
            else:
                ax1.plot(list_points[:, 1], list_points[:, 0], ".", color="green", markersize=1)
            ax1.set_title(f"{cam_name} - Detected Points")
            ax1.axis('off')
            
            # Binarized image
            ax2.imshow(mat1, cmap='gray', origin='lower')
            ax2.set_title(f"{cam_name} - Binarized")
            ax2.axis('off')
            
            # Horizontal lines
            ax3.imshow(img_norm, cmap='gray', origin='lower')
            for line in list_hor_lines:
                ax3.plot(line[:, 1], line[:, 0], "-o", color="red", markersize=2, linewidth=1)
            ax3.set_title(f"{cam_name} - Horizontal Lines ({len(list_hor_lines)})")
            ax3.axis('off')
            
            # Vertical lines
            ax4.imshow(img_norm, cmap='gray', origin='lower')
            for line in list_ver_lines:
                ax4.plot(line[:, 1], line[:, 0], "-o", color="blue", markersize=2, linewidth=1)
            ax4.set_title(f"{cam_name} - Vertical Lines ({len(list_ver_lines)})")
            ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/debug_{cam_name}_fisheye_processing.png", dpi=150, bbox_inches='tight')
            plt.show()
        
        # Save intermediate line plots
        if self.save_intermediate:
            losa.save_plot_image(f"{output_dir}/03_horizontal_lines.png", list_hor_lines, height, width)
            losa.save_plot_image(f"{output_dir}/03_vertical_lines.png", list_ver_lines, height, width)
        
        # Step 9: Find center of distortion using vanishing points (following the example)
        print(f"      Step 9: Find center of distortion for {cam_name}...")
        try:
            xcenter, ycenter = proc.find_center_based_vanishing_points_iteration(
                list_hor_lines, list_ver_lines, 
                iteration=self.fisheye_params['vanishing_point_iterations'])
            print(f"      {cam_name}: Center of distortion: X={xcenter:.4f}, Y={ycenter:.4f}")
        except Exception as e:
            print(f"      Warning: Vanishing point method failed for {cam_name}, using coarse method: {e}")
            xcenter, ycenter = proc.find_cod_coarse(list_hor_lines, list_ver_lines)
            print(f"      {cam_name}: Center of distortion (coarse): X={xcenter:.4f}, Y={ycenter:.4f}")
        
        # Step 10: Correct perspective distortion (following the example)
        print(f"      Step 10: Correct perspective effect for {cam_name}...")
        if self.fisheye_params['enable_perspective_correction']:
            try:
                corr_hor_lines, corr_ver_lines = proc.correct_perspective_effect(
                    list_hor_lines, list_ver_lines, xcenter, ycenter)
                print(f"      {cam_name}: Perspective correction applied")
               
            except Exception as e:
                print(f"      Warning: Perspective correction failed for {cam_name}, using original lines: {e}")
                corr_hor_lines, corr_ver_lines = list_hor_lines, list_ver_lines
        else:
            corr_hor_lines, corr_ver_lines = list_hor_lines, list_ver_lines
            print(f"      {cam_name}: Perspective correction skipped")
        
           # Close the figure to free memory


        # Step 11: Calculate polynomial coefficients (following the example)
        print(f"      Step 11: Calculate radial distortion coefficients for {cam_name}...")
        try:
            list_bfact = proc.calc_coef_backward(corr_hor_lines, corr_ver_lines, 
                                               xcenter, ycenter, self.num_coef)
            print(f"      {cam_name}: Polynomial coefficients: {list_bfact}")
        except Exception as e:
            print(f"      Error: Failed to calculate coefficients for {cam_name}: {e}")
            list_bfact = [1.0] + [0.0] * (self.num_coef - 1)
            print(f"      {cam_name}: Using default coefficients: {list_bfact}")
        
        # Save coefficients
        losa.save_metadata_txt(f"{output_dir}/coefficients_radial_distortion.txt", xcenter, ycenter, list_bfact)
        losa.save_metadata_json(f"{output_dir}/distortion_parameters.json", xcenter, ycenter, list_bfact)
        
        # Calculate unwarped lines (needed for both intermediate saving and perspective coefficients)
        print(f"      Step 12: Calculate unwarped lines for {cam_name}...")
        try:
            # Test correction on lines
            list_uhor_lines = post.unwarp_line_backward(corr_hor_lines, xcenter, ycenter, list_bfact)
            list_uver_lines = post.unwarp_line_backward(corr_ver_lines, xcenter, ycenter, list_bfact)
            print(f"      {cam_name}: Unwarped lines calculated successfully")
        except Exception as e:
            print(f"      Warning: Could not calculate unwarped lines for {cam_name}: {e}")
            list_uhor_lines = corr_hor_lines
            list_uver_lines = corr_ver_lines
        
        # Save intermediate results if enabled
        if self.save_intermediate:
            print(f"      Step 13: Save correction results for {cam_name}...")
            
            try:
                # Save corrected line plots
                losa.save_plot_image(f"{output_dir}/04_unwarpped_horizontal_lines.png", list_uhor_lines, height, width)
                losa.save_plot_image(f"{output_dir}/04_unwarpped_vertical_lines.png", list_uver_lines, height, width)
                
                # Apply correction to the image (following the example)
                img_corr = util.unwarp_color_image_backward(image, xcenter, ycenter, list_bfact, 
                                                          pad=self.fisheye_params['padding'])
                losa.save_image(f"{output_dir}/05_corrected_image.jpg", img_corr)
                
                print(f"      {cam_name}: Correction results saved")
                
            except Exception as e:
                print(f"      Warning: Could not complete correction analysis for {cam_name}: {e}")
                img_corr = None
        else:
            img_corr = None
        
        # Calculate and save separate perspective coefficients (like V3)
        pers_coef = None
        if self.save_perspective_coefficients:
            print(f"      Step 14: Calculate separate perspective coefficients for {cam_name}...")
            try:
                # Generate source and target points for perspective correction
                source_points, target_points = proc.generate_source_target_perspective_points(
                    list_uhor_lines, list_uver_lines, 
                    equal_dist=self.perspective_params['equal_dist'],
                    scale=self.perspective_params['scale'],
                    optimizing=self.perspective_params['optimizing'])
                
                # Calculate perspective coefficients
                pers_coef = proc.calc_perspective_coefficients(source_points, target_points, mapping="backward")
                
                # Save perspective coefficients
                np.savetxt(f"{output_dir}/perspective_coefficients.txt", pers_coef.reshape(-1, 1))
                
                # Apply perspective correction to the radially corrected image (if available)
                if self.save_intermediate and img_corr is not None:
                    try:
                        image_pers_corr = post.correct_perspective_image(img_corr, pers_coef)
                        losa.save_image(f"{output_dir}/06_corrected_image_radial_perspective.jpg", image_pers_corr)
                        
                        # Calculate and save perspective difference
                        pers_diff = image_pers_corr.astype(np.float32) - img_corr.astype(np.float32)
                        pers_diff = np.clip(pers_diff + 128, 0, 255).astype(np.uint8)
                        losa.save_image(f"{output_dir}/06_difference_perspective.jpg", pers_diff)
                        
                        print(f"      {cam_name}: Perspective correction applied and saved")
                    except Exception as e:
                        print(f"      Warning: Could not save perspective corrected image for {cam_name}: {e}")
                
                print(f"      {cam_name}: Perspective coefficients calculated and saved")
                print(f"      {cam_name}: Perspective coefficients: {pers_coef}")
                
            except Exception as e:
                print(f"      Warning: Perspective coefficient calculation failed for {cam_name}: {e}")
                pers_coef = None
        else:
            print(f"      {cam_name}: Perspective coefficient calculation disabled")
        
        # Store perspective coefficients in results
        self.results[cam_name]['pers_coef'] = pers_coef
        
        return xcenter, ycenter, list_bfact

    def process_single_camera(self, image, output_dir):
        """Process single camera using proper fisheye dot pattern workflow"""
        print(f"\n=== Processing Single Camera (Proper Fisheye Dot Pattern Workflow) ===")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save input image
        self.save_jpeg_from_array(image, f"{output_dir}/00_input_cropped.jpg")
        
        # Process camera using the proper fisheye dot pattern workflow
        print(f"\n   Processing camera (cam0)...")
        
        xcenter, ycenter, coeffs = self.process_fisheye_dot_pattern(image, 'cam0', output_dir)
        
        results = {
            'cam0': {
                'xcenter': float(xcenter),
                'ycenter': float(ycenter),
                'coeffs': [float(c) for c in coeffs],
                'pers_coef': [float(c) for c in self.results['cam0']['pers_coef']] if self.results['cam0']['pers_coef'] is not None else None,
                'crop_params': self.crop_params['cam0']
            }
        }
        
        self.results['cam0']['xcenter'] = xcenter
        self.results['cam0']['ycenter'] = ycenter
        self.results['cam0']['coeffs'] = coeffs
        
        return results

    def process_single_camera_file(self, dng_path, output_base):
        """Process single camera DNG file using proper fisheye dot pattern analysis"""
        print("=== Single Camera Fisheye Distortion Correction V4 (Proper Dot Pattern Workflow) ===")
        print(f"DNG: {os.path.basename(dng_path)}")
        print(f"Output base: {output_base}")
        print("Following Discorpy_Fisheye_Example.py workflow exactly")
        
        # Create main output directory
        os.makedirs(output_base, exist_ok=True)
        
        # Load and process camera (cam0)
        print("\n--- Loading Camera (cam0) ---")
        gray, rgb = self.load_dng_image(dng_path, to_grayscale=True)
        if gray is None:
            raise ValueError("Failed to load DNG file")
        
        # Crop image
        cropped = self.crop_image(gray, 'cam0')
        
        # Save cropped RGB version as JPEG
        rgb_cropped = self.crop_image(rgb, 'cam0')
        self.save_jpeg_from_array(rgb_cropped, f"{output_base}/cam0_cropped.jpg")
        
        # Process camera using proper fisheye workflow
        try:
            results = self.process_single_camera(cropped, output_base)
            
            # Save results
            with open(f"{output_base}/distortion_coefficients.json", 'w') as f:
                json.dump(results, f, indent=4)
            
            # Save summary
            with open(f"{output_base}/summary.txt", 'w') as f:
                f.write("=== Single Camera Fisheye Distortion Correction Results (Proper Dot Pattern Workflow) ===\n\n")
                f.write(f"Camera (cam0):\n")
                f.write(f"  Center: ({results['cam0']['xcenter']:.4f}, {results['cam0']['ycenter']:.4f})\n")
                f.write(f"  Radial Coefficients: {results['cam0']['coeffs']}\n")
                if results['cam0']['pers_coef'] is not None:
                    f.write(f"  Perspective Coefficients: {results['cam0']['pers_coef']}\n")
                else:
                    f.write(f"  Perspective Coefficients: Not calculated\n")
                f.write(f"  Crop: {self.crop_params['cam0']}\n\n")
                
                f.write(f"Processing parameters:\n")
                f.write(f"  Number of coefficients: {self.num_coef}\n")
                f.write(f"  Analysis method: Proper fisheye dot pattern calibration\n")
                f.write(f"  Based exactly on Discorpy_Fisheye_Example.py\n")
                f.write(f"  FFT normalization sigma: {self.sigma_normalization}\n")
                f.write(f"  Vanishing point iterations: {self.fisheye_params['vanishing_point_iterations']}\n")
                f.write(f"  Perspective correction: {self.fisheye_params['enable_perspective_correction']}\n")
                f.write(f"  Separate perspective coefficients: {self.save_perspective_coefficients}\n")
                f.write(f"  Padding for unwarp: {self.fisheye_params['padding']}\n")
                
                # Perspective correction parameters
                f.write(f"\nPerspective coefficient calculation settings:\n")
                f.write(f"  Equal distance: {self.perspective_params['equal_dist']}\n")
                f.write(f"  Scale method: {self.perspective_params['scale']}\n")
                f.write(f"  Optimizing: {self.perspective_params['optimizing']}\n")
                
                # Write parameters for camera
                dot_config = self.dot_pattern_params['cam0']
                group_config = self.grouping_params['cam0']
                mask_config = self.mask_params['cam0']
                
                f.write(f"\nCamera (cam0) parameters:\n")
                f.write(f"  Binarization ratio: {dot_config['binarization_ratio']}\n")
                f.write(f"  Size/distance ratio: {dot_config['size_distance_ratio']}\n")
                f.write(f"  Slope ratio: {dot_config['slope_ratio']}\n")
                f.write(f"  Grouping ratio: {group_config['ratio']}\n")
                f.write(f"  Num dot miss: {group_config['num_dot_miss']}\n")
                f.write(f"  Accepted ratio: {group_config['accepted_ratio']}\n")
                f.write(f"  Polynomial order: {group_config['order']}\n")
                f.write(f"  Residual threshold: {group_config['residual_threshold']}\n")
                f.write(f"  Polygon vertices: {len(mask_config['polygon_verts'])} points\n")
                f.write(f"  Masking type: V1-style polygon masking\n")
            
            print("\n=== PROCESSING COMPLETE ===")
            print(f"Camera (cam0):  Center: ({results['cam0']['xcenter']:.4f}, {results['cam0']['ycenter']:.4f})")
            print(f"                Radial Coeffs: {results['cam0']['coeffs']}")
            if results['cam0']['pers_coef'] is not None:
                print(f"                Perspective: Available")
            else:
                print(f"                Perspective: Failed/Skipped")
            print(f"Results saved to: {output_base}")
            print(f"\nFinal Parameters Used:")
            print(f"  Analysis method: Proper fisheye dot pattern calibration")
            print(f"  Based exactly on: Discorpy_Fisheye_Example.py")
            print(f"  Number of coefficients: {self.num_coef}")
            print(f"  FFT normalization sigma: {self.sigma_normalization}")
            print(f"  Parabola masking: {'Enabled' if self.apply_masking else 'Disabled'}")
            print(f"  Perspective correction: {self.fisheye_params['enable_perspective_correction']}")
            print(f"  Separate perspective coefficients: {'Enabled' if self.save_perspective_coefficients else 'Disabled'}")
            print(f"  Vanishing point iterations: {self.fisheye_params['vanishing_point_iterations']}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False



    def test_correction_on_images(self, test_image, output_base):
        """Test the correction on additional images using proper fisheye correction"""
        if not self.test_images:
            return
            
        print("\n=== Testing correction on additional images ===")
        
        if not test_image or not os.path.exists(test_image):
            print(f"Skipping test - file not found: {test_image}")
            return
                
        print(f"Testing camera...")
        
        # Load test image
        if test_image.lower().endswith('.dng'):
            test_gray, test_rgb = self.load_dng_image(test_image, to_grayscale=False)
        else:
            test_rgb = losa.load_image(test_image, average=False)
        
        if test_rgb is None:
            print(f"Failed to load test image")
            return
        
        # Crop test image
        test_cropped = self.crop_image(test_rgb, 'cam0')
        
        # Apply correction using stored results
        xcenter = self.results['cam0']['xcenter']
        ycenter = self.results['cam0']['ycenter'] 
        coeffs = self.results['cam0']['coeffs']
        
        if xcenter is None:
            print(f"No correction parameters available")
            return
        
        # Apply fisheye correction (following the example)
        test_corrected = util.unwarp_color_image_backward(
            test_cropped, xcenter, ycenter, coeffs, 
            pad=self.fisheye_params['padding'])
        
        # Save results
        test_output_dir = f"{output_base}_test"
        self.save_jpeg_from_array(test_cropped, f"{test_output_dir}/test_image_original.jpg")
        self.save_jpeg_from_array(test_corrected, f"{test_output_dir}/test_image_corrected.jpg")
        
        print(f"   Test correction saved")

# Main execution function
def main():
    # Create a simple tkinter root window for file dialogs
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    print("=== Single Camera Fisheye Distortion Correction V4 (Proper Dot Pattern Workflow) ===")
    print("Following Discorpy_Fisheye_Example.py exactly for proper fisheye dot calibration")
    print("Please select the calibration file...")
    
    # Select camera DNG file
    dng_path = filedialog.askopenfilename(
        title="Select Camera DNG Calibration File",
        filetypes=[
            ("DNG files", "*.dng"),
            ("All files", "*.*")
        ],
        initialdir=r"C:\Users\NoahB\Documents\HebrewU Bioengineering\Equipment\Camera"
    )
    
    if not dng_path:
        print("No DNG file selected. Exiting...")
        return
    
    # Select output directory
    output_base = filedialog.askdirectory(
        title="Select Output Directory for Results",
        initialdir=os.path.dirname(dng_path)  # Start in same directory as input file
    )
    
    if not output_base:
        # If no directory selected, use the same directory as the input file
        output_base = os.path.join(os.path.dirname(dng_path), "Single_Fisheye_V4_Results")
        print(f"No output directory selected. Using: {output_base}")
    
    # Display selected files
    print(f"\nSelected files:")
    print(f"DNG:  {os.path.basename(dng_path)}")
    print(f"Output:    {output_base}")
    
    # Ask user for processing options
    response = messagebox.askyesno(
        "Processing Options", 
        "Enable debug plots and detailed visualization?\n\n"
        "YES = Show debug plots and save intermediate images\n"
        "NO = Quick processing with minimal output"
    )
    
    debug_enabled = response
    
    # Ask about masking
    masking_response = messagebox.askyesnocancel(
        "Polygon Masking (V1 Style)",
        "Enable polygon masking to limit detection area?\n\n"
        "YES = Apply polygon masking (recommended)\n"
        "NO = Use all detected points\n"
        "CANCEL = Skip masking entirely"
    )
    
    # Ask about perspective coefficients
    perspective_response = messagebox.askyesno(
        "Perspective Coefficients",
        "Calculate and save separate perspective coefficients?\n\n"
        "YES = Calculate both radial AND perspective coefficients (like V3)\n"
        "NO = Only save combined radial coefficients (like original fisheye example)\n\n"
        "Perspective coefficients allow separate correction of perspective distortion\n"
        "when calibration target isn't perfectly parallel to camera."
    )
    
    # Configure masking
    if masking_response is None:  # Cancel - disable masking
        masking_enabled = False
        interactive_polygon = False
    elif masking_response:  # Yes - enable masking
        masking_enabled = True
        # Ask about interactive polygon drawing
        interactive_response = messagebox.askyesno(
            "Interactive Polygon Drawing",
            "Do you want to draw new polygon masks interactively?\n\n"
            "YES = Draw new polygons for each camera\n"
            "NO = Use default polygon from V1"
        )
        interactive_polygon = interactive_response
    else:  # No - disable masking
        masking_enabled = False
        interactive_polygon = False
    
    # Create processor
    processor = SingleCameraFisheyeDistortionCorrectionV4()
    
    # Configure processing options based on user choice
    processor.debug_plots = 1 if debug_enabled else 0
    processor.save_intermediate = 1 if debug_enabled else 0
    processor.apply_masking = 1 if masking_enabled else 0
    processor.interactive_polygon_drawing = 1 if interactive_polygon else 0
    processor.save_perspective_coefficients = 1 if perspective_response else 0
    processor.test_images = 0  # Can be enabled separately if needed
    
    print(f"\nProcessing configuration:")
    print(f"  Analysis type: Proper fisheye dot pattern (following Discorpy_Fisheye_Example.py)")
    print(f"  Debug plots: {'Enabled' if processor.debug_plots else 'Disabled'}")
    print(f"  Save intermediate: {'Enabled' if processor.save_intermediate else 'Disabled'}")
    print(f"  Polygon masking: {'Enabled' if processor.apply_masking else 'Disabled'}")
    if processor.apply_masking:
        print(f"  Interactive polygon drawing: {'Enabled' if processor.interactive_polygon_drawing else 'Disabled (using V1 defaults)'}")
    print(f"  Number of coefficients: {processor.num_coef}")
    print(f"  FFT normalization sigma: {processor.sigma_normalization}")
    print(f"  Vanishing point iterations: {processor.fisheye_params['vanishing_point_iterations']}")
    print(f"  Perspective correction: {processor.fisheye_params['enable_perspective_correction']}")
    print(f"  Separate perspective coefficients: {'Enabled' if processor.save_perspective_coefficients else 'Disabled'}")
    
    # Verify file exists
    if not os.path.exists(dng_path):
        messagebox.showerror("Error", f"DNG file not found:\n{dng_path}")
        return
    
    try:
        # Process camera
        print(f"\nStarting processing...")
        success = processor.process_single_camera_file(dng_path, output_base)
        
        if success:
            # Ask if user wants to test on additional images
            if messagebox.askyesno("Testing", "Processing complete!\n\nDo you want to test the correction on additional images?"):
                # Select test image
                test_image = filedialog.askopenfilename(
                    title="Select test image (optional - can cancel)",
                    filetypes=[
                        ("DNG files", "*.dng"),
                        ("Image files", "*.jpg;*.jpeg;*.png;*.tiff"),
                        ("All files", "*.*")
                    ],
                    initialdir=os.path.dirname(dng_path)
                )
                
                if test_image:
                    processor.test_images = 1  # Enable testing
                    processor.test_correction_on_images(test_image, output_base)
            
            print("\n=== All processing complete! ===")
            perspective_info = ""
            if processor.save_perspective_coefficients:
                perspective_info = "\n• perspective_coefficients.txt"
            
            messagebox.showinfo("Success", 
                f"Processing completed successfully!\n\n"
                f"Results saved to:\n{output_base}\n\n"
                f"Key files:\n"
                f"• distortion_coefficients.json (radial + perspective)\n"
                f"• summary.txt{perspective_info}\n"
                f"• Individual processing results in subdirectories\n\n"
                f"Analysis type: Proper fisheye dot pattern (Discorpy_Fisheye_Example.py)\n"
                f"Coefficients: {'Radial + Perspective' if processor.save_perspective_coefficients else 'Radial only'}")
            
        else:
            print("\n=== Processing failed! ===")
            messagebox.showerror("Error", "Processing failed! Check console for details.")
            
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"Unexpected error occurred:\n\n{str(e)}\n\nCheck console for details.")
    
    finally:
        root.destroy()  # Clean up the tkinter root window

if __name__ == "__main__":
    main() 