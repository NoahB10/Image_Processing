#!/usr/bin/env python3
"""
Image Post-Processing Script for IMX708 Dual Camera System (v1.2)

This script processes image files captured by the dual camera system:
- Loads DNG files from both cameras (left and right) with fallback to generic formats
- Supports JPEG, PNG, TIFF, BMP, and other common image formats
- Applies camera-specific cropping (optional)
- Applies camera-specific distortion correction (optional)
- Applies perspective correction (optional)
- Saves as JPEG, TIFF, or PNG with quality settings

Version 1.2 Improvements:
- Added fallback support for generic image formats (JPEG, PNG, TIFF, BMP, etc..)
- Enhanced error handling with graceful degradation from DNG to generic formats
- Updated GUI to support broader file type selection
- Fixed TIFF saving issues by using imageio for reliable TIFF output
- Improved DNG loading with better color space handling
- Enhanced distortion correction with better error handling
- Added support for batch processing of image pairs
- Improved GUI with better parameter display
- Added support for combined side-by-side image output
- Better handling of image data types and ranges
- More robust error handling and logging
- Improved file naming with processing status indicators
- Added perspective correction support

Usage:
    python image_post_processing.py [left_image] [right_image] [options]
    
Or run interactively to select files via GUI.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import json
import argparse
from PIL import Image
import rawpy
import discorpy.post.postprocessing as post
from datetime import datetime
import imageio
import discorpy.util.utility as util
import re

class DualImagePostProcessor:
    def __init__(self):
        # Default cropping parameters (same as main GUI)
        self.crop_params = {
            'cam0': {'width': 2070, 'start_x': 1260, 'height': 2592},  # Left camera
            'cam1': {'width': 2050, 'start_x': 1400, 'height': 2592}   # Right camera
        }

        self.current_filename = 'cam'
        # Default distortion correction parameters (including perspective coefficients)
        self.distortion_params = {}

        # Processing options
        self.apply_cropping = False
        self.enable_distortion_correction = True
        self.enable_perspective_correction = True  # New perspective correction flag
        self.apply_rotation = True  # New flag for left image rotation
        self.jpeg_quality = 95
        self.output_format = 'JPEG'  # 'JPEG', 'TIFF', 'PNG'
        self.save_combined = True  # Save side-by-side combined image
        self.save_individual = True  # Save individual processed images

        # Load saved coefficients if available
        self.load_distortion_coefficients()

    def load_distortion_coefficients(self):
        """Load distortion correction coefficients from saved file"""
        coeff_file = 'distortion_coefficients.json'
        if os.path.exists(coeff_file):
            try:
                try: 
                    with open(coeff_file, 'r') as f:
                        saved_params = json.load(f)
                except json.JSONDecodeError as e:
                    print("JSON decode error:", e)
                except FileNotFoundError:
                    print("File not found:", coeff_file)
                    return
                    
                
                # Update distortion parameters with loaded data
                keys = ['xcenter', 'ycenter', 'coeffs', 'pers_coef', 'rotation_angle']
                for cam in ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7']:
                    if cam in saved_params:
                        if cam not in self.distortion_params:
                            self.distortion_params[cam] = {key: None for key in keys}
                    for key in keys:
                        if key in saved_params[cam]:    
                            self.distortion_params[cam][key] = saved_params[cam][key]
                        else:
                            self.distortion_params[cam][key] = None

                
                print("[INFO] Distortion parameters now in use for processing:")
                print(json.dumps(self.distortion_params, indent=2))
            except Exception as e:
                print(f"[WARNING] Failed to load saved coefficients: {e}")
                print("[INFO] Using default distortion coefficients")

    def load_image(self, filepath):
        """Load image file as generic image format using OpenCV """
        try:
            # OpenCV loads images in BGR format
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                print(f"[ERROR] Failed to load image with OpenCV: {os.path.basename(filepath)}")
                return None
            
            # Convert BGR to RGB if it's a color image
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # Handle RGBA images
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            
            print(f"[SUCCESS] Loaded generic image: {os.path.basename(filepath)}")
            print(f"   Shape: {image.shape}, dtype: {image.dtype}")
            print(f"   Range: [{image.min()}, {image.max()}]")
            return image
            
        except Exception as fallback_error:
            print(f"[ERROR] Failed to load image with fallback method: {os.path.basename(filepath)} - {fallback_error}")
            return None

    def crop_image(self, image, cam_name):
        """Crop image according to camera-specific parameters"""
        if not self.apply_cropping or cam_name not in self.crop_params:
            return image
            
        params = self.crop_params[cam_name]
        start_x = params['start_x']
        width = params['width']
        height = params['height']
        
        # Ensure we don't exceed image boundaries
        img_height, img_width = image.shape[:2]
        end_x = min(start_x + width, img_width)
        end_y = min(height, img_height)
        
        # Crop the image: [y_start:y_end, x_start:x_end]
        cropped = image[:end_y, start_x:end_x]
        print(f"[INFO] Cropped {cam_name}: {image.shape} -> {cropped.shape}")
        return cropped

    def apply_distortion_correction(self, image, cam_name):
        """Apply distortion correction to the image"""
        if not self.enable_distortion_correction or cam_name not in self.distortion_params:
            return image
            
        params = self.distortion_params[cam_name]
        xcenter = params['xcenter']
        ycenter = params['ycenter']
        coeffs = params['coeffs']
        
        try:
            # Store original data type and range
            original_dtype = image.dtype
            original_min = image.min()
            original_max = image.max()
            
            # Convert to float for processing
            if image.dtype != np.float64:
                image_float = image.astype(np.float64)
            else:
                image_float = image.copy()
            
            if image_float.ndim == 2:
                # Grayscale image
                corrected = post.unwarp_image_backward(image_float, xcenter, ycenter, coeffs, mode='constant')
            else:
                # Multi-channel image
                # corrected = np.zeros_like(image_float)
                # for c in range(image_float.shape[2]):
                #     corrected[:, :, c] = post.unwarp_image_backward(image_float[:, :, c], xcenter, ycenter, coeffs, mode='constant')
                corrected = util.unwarp_color_image_backward(image,xcenter,ycenter,coeffs,pad=True) #autocalc padding cuts off some ofthe edges of the image, increase if want full image
                print(f"image unwarping applied. Variables used: xcenter: {xcenter}, ycenter: {ycenter}, list_bfact: {coeffs}")
            
            # Handle potential NaN or infinite values
            corrected = np.nan_to_num(corrected, nan=0.0, posinf=original_max, neginf=0.0)
            
            # Clip to reasonable range
            corrected = np.clip(corrected, 0, original_max)
            
            # Convert back to original data type
            if original_dtype == np.uint8:
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            elif original_dtype == np.uint16:
                corrected = np.clip(corrected, 0, 65535).astype(np.uint16)
            else:
                corrected = corrected.astype(original_dtype)
            
            print(f"[INFO] Applied distortion correction to {cam_name}")
            print(f"   Center: ({xcenter:.1f}, {ycenter:.1f})")
            print(f"   Input range: [{original_min}, {original_max}], Output range: [{corrected.min()}, {corrected.max()}]")
            
            return corrected
            
        except Exception as e:
            print(f"[ERROR] Distortion correction failed for {cam_name}: {e}")
            return image

    def apply_perspective_correction(self, image, cam_name):
        """Apply perspective correction if coefficients are available"""
        if not self.enable_perspective_correction or cam_name not in self.distortion_params:
            return image
            
        params = self.distortion_params[cam_name]
        pers_coef = params.get('pers_coef')
        
        if pers_coef is None:
            print(f"[INFO] No perspective coefficients available for {cam_name}, skipping")
            return image
        
        try:
            # Store original data type and range
            original_dtype = image.dtype
            original_min = image.min()
            original_max = image.max()
            
            # Convert to float for processing
            if image.dtype != np.float64:
                image_float = image.astype(np.float64)
            else:
                image_float = image.copy()
            
            if image_float.ndim == 2:
                # Grayscale image
                corrected = post.correct_perspective_image(image_float, pers_coef, mode='constant')
            else:
                # Multi-channel image
                corrected = np.zeros_like(image_float)
                for c in range(image_float.shape[2]):
                    corrected[:, :, c] = post.correct_perspective_image(image_float[:, :, c], pers_coef, mode='constant')
            
            # Handle potential NaN or infinite values
            corrected = np.nan_to_num(corrected, nan=0.0, posinf=original_max, neginf=0.0)
            
            # Clip to reasonable range
            corrected = np.clip(corrected, 0, original_max)
            
            # Convert back to original data type
            if original_dtype == np.uint8:
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            elif original_dtype == np.uint16:
                corrected = np.clip(corrected, 0, 65535).astype(np.uint16)
            else:
                corrected = corrected.astype(original_dtype)
            
            print(f"[INFO] Applied perspective correction to {cam_name}")
            print(f"   Input range: [{original_min}, {original_max}], Output range: [{corrected.min()}, {corrected.max()}]")
            
            return corrected
            
        except Exception as e:
            print(f"[ERROR] Perspective correction failed for {cam_name}: {e}")
            return image

    def rotate_image(self, image, cam_name):
        """Rotate the image by the specified angle"""
        if not self.apply_rotation or self.distortion_params[cam_name]['rotation_angle'] is None:
            print(f"[INFO] No rotation angle available for {cam_name}, skipping")
            return image
            
        rotation_angle = self.distortion_params[cam_name]['rotation_angle']

        try:
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Calculate rotation matrix
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_LINEAR, 
                                   borderMode=cv2.BORDER_REFLECT_101)
            
            print(f"[INFO] Applied {rotation_angle}° rotation to image")
            print(f"   Rotation center: ({center[0]}, {center[1]})")
            print(f"   Image dimensions: {width}x{height}")
            return rotated
            
        except Exception as e:
            print(f"[ERROR] image rotation failed: {e}")
            return image

    def save_processed_image(self, image, output_path, cam_name, format_type=None):
        """Save processed image in specified format with improved TIFF handling"""
        if format_type is None:
            format_type = self.output_format

        # Check if output_path is a directory (no file extension) or if the filename doesn't contain processing keywords
        output_basename = os.path.basename(output_path)
        has_extension = '.' in output_basename
        has_processing_keywords = any(keyword in output_basename for keyword in ["cropped", "corrected", "perspective", "rotated", "processed"])
        
        if not has_extension or not has_processing_keywords:
            base_name = cam_name
            suffixes = []
            if self.apply_cropping:
                suffixes.append("cropped")
            if self.enable_distortion_correction:
                suffixes.append("corrected")
            if self.enable_perspective_correction:
                suffixes.append("perspective")
            if self.apply_rotation:
                suffixes.append("rotated")
            
            suffix_str = "_" + "_".join(suffixes) if suffixes else "_processed"
            ext_map = {'JPEG': '.jpg', 'TIFF': '.tiff', 'PNG': '.png'}
            ext = ext_map.get(self.output_format.upper(), '.jpg')
            
            filename= f"{base_name}{suffix_str}{ext}"
            output_path = os.path.join(output_path, filename)

        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"[INFO] Created output directory: {output_dir}")
            
            # Ensure we have a valid image
            if image is None or image.size == 0:
                print(f"[ERROR] Invalid image data for {output_path}")
                return False
           
        
            

            print(f"[DEBUG] Saving image: {output_path}")
            print(f"[DEBUG] Original image shape: {image.shape}, dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
            
            # For TIFF, use imageio directly - simple and clean
            if format_type.upper() == 'TIFF':
                # Use imageio for TIFF - handles everything automatically
                imageio.imsave(output_path, image)
                print(f"[SUCCESS] Saved TIFF using imageio: {output_path}")
                return True
                    
            else:
                # For JPEG/PNG, convert to 8-bit and use PIL
                if image.dtype == np.uint16:
                    # Convert 16-bit to 8-bit
                    image_save = (image / 256).astype(np.uint8)
                elif image.dtype == np.float32 or image.dtype == np.float64:
                    # Normalize float to 8-bit
                    if image.max() <= 1.0:
                        image_save = (image * 255).astype(np.uint8)
                    else:
                        image_save = np.clip(image / image.max() * 255, 0, 255).astype(np.uint8)
                else:
                    image_save = np.clip(image, 0, 255).astype(np.uint8)
                
                # Handle BGR to RGB conversion for PIL if needed
                if len(image_save.shape) == 3 and image_save.shape[2] == 3:
                    # Convert BGR to RGB for PIL
                    image_rgb = image_save[:,:,[2,1,0]]
                else:
                    # Grayscale or already in correct format
                    image_rgb = image_save
                
                # Create PIL Image
                if len(image_rgb.shape) == 2:
                    # Grayscale
                    pil_image = Image.fromarray(image_rgb, mode='L')
                elif image_rgb.shape[2] == 3:
                    # RGB
                    pil_image = Image.fromarray(image_rgb, mode='RGB')
                else:
                    print(f"[ERROR] Unsupported image format: {image_rgb.shape}")
                    return False
                
                # Save with format-specific options
                if format_type.upper() == 'JPEG':
                    pil_image.save(output_path, 'JPEG', quality=self.jpeg_quality, optimize=True)
                elif format_type.upper() == 'PNG':
                    pil_image.save(output_path, 'PNG', optimize=True)
                else:
                    pil_image.save(output_path)
                
                print(f"[SUCCESS] Saved processed image: {output_path}")
                return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save image {output_path}: {e}")
            import traceback
            traceback.print_exc()
            return False


    def create_gui(self):
        """Create a GUI for dual image processing"""
        root = tk.Tk()
        root.title("Dual Camera DNG Post-Processor")
        root.geometry("700x700")
        
        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Processing options
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", padding="10")
        options_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Cropping checkbox
        self.crop_var = tk.BooleanVar(value=self.apply_cropping)
        ttk.Checkbutton(options_frame, text="Apply Cropping", 
                       variable=self.crop_var).grid(row=0, column=0, sticky=tk.W)
        
        # Distortion correction checkbox
        self.distortion_var = tk.BooleanVar(value=self.enable_distortion_correction)
        ttk.Checkbutton(options_frame, text="Apply Distortion Correction", 
                       variable=self.distortion_var).grid(row=1, column=0, sticky=tk.W)
        
        # Perspective correction checkbox
        self.perspective_var = tk.BooleanVar(value=self.enable_perspective_correction)
        ttk.Checkbutton(options_frame, text="Apply Perspective Correction", 
                       variable=self.perspective_var).grid(row=2, column=0, sticky=tk.W)
        
        # rotation checkbox
        self.rotation_var = tk.BooleanVar(value=self.apply_rotation)
        ttk.Checkbutton(options_frame, text="Apply Image Rotation", 
                       variable=self.rotation_var).grid(row=3, column=0, sticky=tk.W)
        
       
        # Output format
        ttk.Label(options_frame, text="Output Format:").grid(row=6, column=0, sticky=tk.W, pady=(10, 0))
        self.format_var = tk.StringVar(value=self.output_format)
        format_combo = ttk.Combobox(options_frame, textvariable=self.format_var, 
                                   values=['JPEG', 'TIFF', 'PNG'], state='readonly', width=10)
        format_combo.grid(row=6, column=1, sticky=tk.W, padx=(10, 0), pady=(10, 0))
        
        # JPEG quality
        ttk.Label(options_frame, text="JPEG Quality:").grid(row=7, column=0, sticky=tk.W, pady=(5, 0))
        self.quality_var = tk.IntVar(value=self.jpeg_quality)
        quality_scale = ttk.Scale(options_frame, from_=50, to=100, variable=self.quality_var, 
                                 orient=tk.HORIZONTAL, length=200)
        quality_scale.grid(row=7, column=1, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # Quality label
        self.quality_label = ttk.Label(options_frame, text=f"{self.jpeg_quality}%")
        self.quality_label.grid(row=7, column=2, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        
        def update_quality_label(*args):
            self.quality_label.config(text=f"{self.quality_var.get()}%")
        quality_scale.config(command=update_quality_label)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        

        # Batch processing
        ttk.Button(file_frame, text="Process Multiple Image Files", 
                  command=self.gui_process_multiple).grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)

        # Single image processing (legacy)
        ttk.Button(file_frame, text="Process Single Image File", 
                  command=self.gui_process_single).grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Current parameters display
        params_frame = ttk.LabelFrame(main_frame, text="Current Parameters", padding="10")
        params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Create text widget for parameters
        self.params_text = tk.Text(params_frame, height=10, width=70)
        self.params_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for text widget
        scrollbar = ttk.Scrollbar(params_frame, orient=tk.VERTICAL, command=self.params_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.params_text.config(yscrollcommand=scrollbar.set)
        
        # Update parameters display
        self.update_params_display()
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
    
        
        ttk.Button(buttons_frame, text="Load Parameters from File", 
                  command=self.update_gui_from_params).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(buttons_frame, text="Exit", 
                  command=root.quit).pack(side=tk.RIGHT)
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        file_frame.columnconfigure(0, weight=1)
        params_frame.columnconfigure(0, weight=1)
        
        return root
    
    def update_params_display(self):
        """Update the parameters display in the GUI"""
        if hasattr(self, 'params_text'):
            self.params_text.delete(1.0, tk.END)
            
            # Crop parameters
            self.params_text.insert(tk.END, "Crop Parameters:\n")
            for cam, params in self.crop_params.items():
                self.params_text.insert(tk.END, f"  {cam}: {params['width']}x{params['height']} @ ({params['start_x']},0)\n")
            
            self.params_text.insert(tk.END, "\nDistortion Parameters:\n")
            for cam, params in self.distortion_params.items():
                self.params_text.insert(tk.END, f"  {cam}:\n")
                self.params_text.insert(tk.END, f"    Center: ({params['xcenter']:.1f}, {params['ycenter']:.1f})\n")
                self.params_text.insert(tk.END, f"    Coefficients: {params['coeffs']}\n")
            
            self.params_text.insert(tk.END, "\nPerspective Correction:\n")
            for cam, params in self.distortion_params.items():
                pers_coef = params.get('pers_coef')
                if pers_coef is not None:
                    # Format perspective coefficients to 4 decimal points
                    formatted_coeffs = [f"{coeff:.4f}" for coeff in pers_coef]
                    self.params_text.insert(tk.END, f"  {cam}: {formatted_coeffs}\n")
                else:
                    self.params_text.insert(tk.END, f"  {cam}: Not available\n")
            
            self.params_text.insert(tk.END, "\nRotation Parameters:\n")
            for cam, params in self.distortion_params.items():
                rotation_angle = params.get('rotation_angle')
                rotation_angle_str = f"{rotation_angle:.1f}" if rotation_angle is not None else "N/A"
                self.params_text.insert(tk.END, f"  {cam}: {rotation_angle_str}°\n")

    
   
    
    def gui_process_single(self):
        """GUI handler for single file processing (legacy support)"""
        # Update settings from GUI
        self.apply_cropping = self.crop_var.get()
        self.enable_distortion_correction = self.distortion_var.get()
        self.enable_perspective_correction = self.perspective_var.get()
        self.apply_rotation = self.rotation_var.get()
        self.output_format = self.format_var.get()
        self.jpeg_quality = self.quality_var.get()
        
        # Select input file
        input_file = filedialog.askopenfilename(
            title="Select image file to process",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp"),("DNG files", "*.dng"), ("All files", "*.*")]
        )
        
        if not input_file:
            return
        
        # Detect camera type
        cam_name = 'cam0'  # Default
        basename = os.path.basename(input_file).lower()
        
        # Select output file
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        suffixes = []
        if self.apply_cropping:
            suffixes.append("cropped")
        if self.enable_distortion_correction:
            suffixes.append("corrected")
        if self.enable_perspective_correction:
            suffixes.append("perspective")
        if self.apply_rotation:
            suffixes.append("rotated")
        
        suffix_str = "_" + "_".join(suffixes) if suffixes else "_processed"
        ext_map = {'JPEG': '.jpg', 'TIFF': '.tiff', 'PNG': '.png'}
        ext = ext_map.get(self.output_format.upper(), '.jpg')
        
        default_output = f"{base_name}{suffix_str}{ext}"
        
        output_file = filedialog.asksaveasfilename(
            title="Save processed image as",
            defaultextension=ext,
            initialfile=default_output,
            filetypes=[(f"{self.output_format} files", f"*{ext}"), ("All files", "*.*")]
        )
        
        if not output_file:
            return
        
        # Process the file
        success = self.process_single_image(input_file, output_file, cam_name)
        
        if success:
            messagebox.showinfo("Success", f"Image processed successfully!\nSaved as: {os.path.basename(output_file)}")
        else:
            messagebox.showerror("Error", "Failed to process image. Check console for details.")
    
    def gui_process_multiple(self):
        """GUI handler for multiple file processing (legacy support)"""
        # Update settings from GUI
        self.apply_cropping = self.crop_var.get()
        self.enable_distortion_correction = self.distortion_var.get()
        self.enable_perspective_correction = self.perspective_var.get()
        self.apply_rotation = self.rotation_var.get()
        self.output_format = self.format_var.get()
        self.jpeg_quality = self.quality_var.get()
        
        # Select input folder, read camera names from folder
        input_folder = filedialog.askdirectory(
            title="Select folder containing PNG images to process",
        )
        png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
        cam_name_to_path = {}
        for png_file in png_files:
            png_path = os.path.join(input_folder, png_file)
            basename = os.path.basename(png_file).lower()
            if 'cam' in basename:
                cam_name = re.search(r'cam(\d+)', basename).group()
                cam_name_to_path[cam_name] = png_path
        if not input_folder:
            return
        
        # Select output folder
        output_folder = filedialog.askdirectory(
            title="Select output folder",
        )
        
        if not output_folder:
            return
        
        # Select output file
        
        # base_name = cam_name
        # suffixes = []
        # if self.apply_cropping:
        #     suffixes.append("cropped")
        # if self.enable_distortion_correction:
        #     suffixes.append("corrected")
        # if self.enable_perspective_correction:
        #     suffixes.append("perspective")
        # if self.apply_rotation:
        #     suffixes.append("rotated")
        
        # suffix_str = "_" + "_".join(suffixes) if suffixes else "_processed"
        # ext_map = {'JPEG': '.jpg', 'TIFF': '.tiff', 'PNG': '.png'}
        # ext = ext_map.get(self.output_format.upper(), '.jpg')
        
        # self.current_filename = f"{base_name}{suffix_str}{ext}"
        
        # Process the folder
        success = self.process_multiple_images(input_folder, output_folder)
        
        if success:
            messagebox.showinfo("Success", f"Image processed successfully!\nSaved to: {os.path.basename(output_folder)}")
        else:
            messagebox.showerror("Error", "Failed to process image. Check console for details.")

    def process_single_image(self, input_path, output_path, cam_name):
        """Process a single PNG image (legacy method)"""
        print(f"\n[INFO] Processing {cam_name} image: {os.path.basename(input_path)}")
        
        # Load the image
        image = self.load_image(input_path)
        if image is None:
            return False
        
        # Apply processing steps
        processed_image = image
        
        # Apply cropping
        if self.apply_cropping:
            processed_image = self.crop_image(processed_image, cam_name)
        
        # Apply distortion correction
        if self.enable_distortion_correction:
            processed_image = self.apply_distortion_correction(processed_image, cam_name)
        
        # Apply perspective correction
        if self.enable_perspective_correction:
            processed_image = self.apply_perspective_correction(processed_image, cam_name)
        
        # image rotation
        if self.apply_rotation:
            processed_image = self.rotate_image(processed_image, cam_name)
        

        # Save the processed image
        success = self.save_processed_image(processed_image, output_path, cam_name)
        
        if success:
            print(f"[SUCCESS] Completed processing: {cam_name} is saved to the folder {os.path.basename(output_path)} ")
            print(f"   Original: {image.shape} -> Processed: {processed_image.shape}")
        
        return success
    
    def process_multiple_images(self, folder_path, output_path):
        """Process multiple images in a folder"""
        
        #get all png files in folder
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        print(f"Found {len(png_files)} PNG files in folder. Processing...")

        cam_name_to_path = {}
        for png_file in png_files:
            print(f"Processing {png_file}...")
            png_path = os.path.join(folder_path, png_file)
            basename = os.path.basename(png_file).lower()
            if 'cam' in basename:
                cam_name = re.search(r'cam(\d+)', basename).group()
                cam_name_to_path[cam_name] = png_path
        print(f"Found {len(cam_name_to_path)} images in folder")
        print('starting processing of images...')  

        for cam_name, png_path in cam_name_to_path.items():
            print(f"Processing {cam_name} image: {os.path.basename(png_path)}")
            self.process_single_image(png_path, output_path, cam_name)
        
        print(f'ALL {len(cam_name_to_path)} IMAGES PROCESSED')
        return True
        


    
    def reload_coefficients(self):
        """Reload distortion coefficients and update display"""
        self.load_distortion_coefficients()
        self.update_params_display()
        self.update_gui_from_params()
        messagebox.showinfo("Coefficients Reloaded", "Distortion coefficients have been reloaded from file and GUI updated.")
    
    def update_gui_from_params(self):
        """Update GUI fields with current parameter values AND USES THOSE PARAMETERS"""
        # Prompt user to select distortion coefficients file
        coeff_file = filedialog.askopenfilename(
            title="Select distortion coefficients file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="distortion_coefficients.json"
        )
        
        if not coeff_file:
            messagebox.showinfo("Cancelled", "No file selected. Parameters not updated.")
            return False  # Indicate failure
        
        try:
            # Load the selected coefficients file
            with open(coeff_file, 'r') as f:
                loaded_params = json.load(f)
            
            # Update distortion parameters with loaded data
            keys = ['xcenter', 'ycenter', 'coeffs', 'pers_coef', 'rotation_angle']
            for cam in ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6', 'cam7']:
                for cam in loaded_params:
                    if cam not in self.distortion_params:
                        self.distortion_params[cam] = {key: None for key in keys}
                    for key in keys:
                        self.distortion_params[cam][key] = loaded_params[cam].get(key, None)
            
                # Update cropping parameters if available in the file
                if 'crop_params' in loaded_params:
                    self.crop_params.update(loaded_params['crop_params'])
                
                
            
            # Update the GUI display
            self.update_params_display()
            
            # if hasattr(self, 'crop_left_width_var'):
            #     self.crop_left_width_var.set(self.crop_params['cam0']['width'])
            #     self.crop_left_height_var.set(self.crop_params['cam0']['height'])
            #     self.crop_left_startx_var.set(self.crop_params['cam0']['start_x'])
            #     self.crop_right_width_var.set(self.crop_params['cam1']['width'])
            #     self.crop_right_height_var.set(self.crop_params['cam1']['height'])
            #     self.crop_right_startx_var.set(self.crop_params['cam1']['start_x'])
            
            # if hasattr(self, 'dist_left_centerx_var'):
            #     self.dist_left_centerx_var.set(self.distortion_params['cam0']['xcenter'])
            #     self.dist_left_centery_var.set(self.distortion_params['cam0']['ycenter'])
            #     self.dist_left_coeffs_var.set(str(self.distortion_params['cam0']['coeffs']))
            #     self.dist_right_centerx_var.set(self.distortion_params['cam1']['xcenter'])
            #     self.dist_right_centery_var.set(self.distortion_params['cam1']['ycenter'])
            #     self.dist_right_coeffs_var.set(str(self.distortion_params['cam1']['coeffs']))
            
            # if hasattr(self, 'rotation_var'):
            #     self.rotation_var.set(self.rotation_angle)
            
            # Check if perspective coefficients were loaded
            pers_coef_loaded = False
            for cam in loaded_params:
                if cam in self.distortion_params and 'pers_coef' in self.distortion_params[cam]:
                    if self.distortion_params[cam]['pers_coef'] is not None:
                        pers_coef_loaded = True
                        break
            
            success_msg = f"Parameters updated from: {os.path.basename(coeff_file)}"
            if pers_coef_loaded:
                success_msg += "\n\nPerspective correction coefficients loaded and available for use."
            
            messagebox.showinfo("Success", success_msg)
            self.sync_gui_with_params()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load parameters from file:\n{str(e)}")
            print(f"[ERROR] Failed to load parameters from {coeff_file}: {e}")
            return False  # Indicate failure

    def sync_gui_with_params(self):
        # Update all GUI widgets to reflect the current processor state
        self.crop_var.set(self.apply_cropping)
        self.distortion_var.set(self.enable_distortion_correction)
        self.perspective_var.set(self.enable_perspective_correction)
        self.rotation_var.set(self.apply_rotation)
        self.format_var.set(self.output_format)
        self.quality_var.set(self.jpeg_quality)
        self.update_params_display()

def main():
    parser = argparse.ArgumentParser(description="Post-process DNG image pairs from IMX708 dual camera system")
    parser.add_argument("left", nargs="?", help="Left camera DNG file")
    parser.add_argument("right", nargs="?", help="Right camera DNG file")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--no-crop", action="store_true", help="Disable cropping")
    parser.add_argument("--no-distortion", action="store_true", help="Disable distortion correction")
    parser.add_argument("--no-perspective", action="store_true", help="Disable perspective correction")
    parser.add_argument("--no-rotation", action="store_true", help="Disable left image rotation")
    parser.add_argument("--format", choices=['JPEG', 'TIFF', 'PNG'], default='JPEG', help="Output format")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (50-100)")
    parser.add_argument("--no-individual", action="store_true", help="Don't save individual images")
    parser.add_argument("--no-combined", action="store_true", help="Don't save combined image")
    parser.add_argument("--batch", action="store_true", help="Process all DNG file pairs in directory")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    
    args = parser.parse_args()
    
    # Create processor
    processor = DualImagePostProcessor()
    
    # Update settings from command line
    if args.no_crop:
        processor.apply_cropping = False
    if args.no_distortion:
        processor.enable_distortion_correction = False
    if args.no_perspective:
        processor.enable_perspective_correction = False
    if args.no_rotation:
        processor.apply_rotation = False
    if args.no_individual:
        processor.save_individual = False
    if args.no_combined:
        processor.save_combined = False
    processor.output_format = args.format
    processor.jpeg_quality = args.quality
    
    # Launch GUI if requested or no input provided
    if args.gui or (args.left is None and args.right is None):
        root = processor.create_gui()
        root.mainloop()
        return
    
    # Command line processing
    if args.batch:
        # Batch processing - use left argument as directory
        input_dir = args.left if args.left else "."
        processor.process_batch_pairs(input_dir, args.output)
    elif args.left and args.right:
        # Dual image processing
        processor.process_dual_images(args.left, args.right, args.output)
    else:
        print("Error: Please provide both left and right image files, or use --gui or --batch")
        parser.print_help()

if __name__ == "__main__":
    main() 