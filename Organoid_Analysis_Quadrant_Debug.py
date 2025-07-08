#!/usr/bin/env python3
"""
Organoid Analysis with Debug GUI Components
This version includes the step-by-step debugging interface for parameter tuning
"""

# Import the base analyzer from the simple version
from Organoid_Analysis_Quadrant import WellOrganoidAnalyzer, AnalysisParameters
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

# PyQt6 imports with availability checking
PYQT_AVAILABLE = True
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                                 QWidget, QPushButton, QLabel, QSlider, QGroupBox,
                                 QScrollArea, QTextEdit, QGridLayout, QSpinBox, 
                                 QDoubleSpinBox, QFileDialog)
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QPixmap, QImage
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt6 not available. Install with: pip install PyQt6")

class ParameterDebugGUI(QMainWindow):
    """Step-by-step debugging GUI for organoid detection"""
    
    def __init__(self, analyzer):
        super().__init__()
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt6 is required for the debug GUI")
            
        self.analyzer = analyzer
        self.current_step = 0
        self.processing_images = {}
        
        # Step tracking
        self.binary_complete = False
        self.color_complete = False
        
        # Alpha transparency handling
        self.alpha_mask = None
        self.crop_info = None
        
        # Morphological operation parameters
        self.kernel_size = 3
        self.dilation_iterations = 0
        
        # Batch processing variables
        self.input_folder = None
        self.batch_images = []
        
        self.init_ui()
        self.load_images()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Organoid Analysis - Debug Interface")
        self.setGeometry(100, 100, 1400, 900)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # Step buttons
        step_group = QGroupBox("Binary Detection Steps")
        step_layout = QVBoxLayout(step_group)
        
        self.step1_btn = QPushButton("Step 1: Convert to Grayscale")
        self.step2_btn = QPushButton("Step 2: Create Dark Mask")
        self.step3_btn = QPushButton("Step 3: Inpaint Dark Areas")
        self.step4_btn = QPushButton("Step 4: Convert Inpainted to Gray")
        self.step5_btn = QPushButton("Step 5: Apply Binary Threshold")
        self.step6_btn = QPushButton("Step 6: Detect Binary Centroids")
        
        self.step1_btn.clicked.connect(self.step1_grayscale)
        self.step2_btn.clicked.connect(self.step2_dark_mask)
        self.step3_btn.clicked.connect(self.step3_inpainting)
        self.step4_btn.clicked.connect(self.step4_inpaint_gray)
        self.step5_btn.clicked.connect(self.step5_binary)
        self.step6_btn.clicked.connect(self.step6_detect_binary)
        
        step_layout.addWidget(self.step1_btn)
        step_layout.addWidget(self.step2_btn)
        step_layout.addWidget(self.step3_btn)
        step_layout.addWidget(self.step4_btn)
        step_layout.addWidget(self.step5_btn)
        step_layout.addWidget(self.step6_btn)
        
        left_layout.addWidget(step_group)
        
        # Color detection buttons
        color_group = QGroupBox("Color Detection")
        color_layout = QVBoxLayout(color_group)
        
        self.color_sample_btn = QPushButton("Sample Colors")
        self.color_detect_btn = QPushButton("Detect Color Centroids")
        self.skip_binary_btn = QPushButton("Skip Binary Detection")
        self.skip_color_btn = QPushButton("Skip Color Detection")
        
        self.color_sample_btn.clicked.connect(self.color_step_sample)
        self.color_detect_btn.clicked.connect(self.color_step_detect)
        self.skip_binary_btn.clicked.connect(self.skip_binary)
        self.skip_color_btn.clicked.connect(self.skip_color)
        
        color_layout.addWidget(self.color_sample_btn)
        color_layout.addWidget(self.color_detect_btn)
        color_layout.addWidget(self.skip_binary_btn)
        color_layout.addWidget(self.skip_color_btn)
        
        left_layout.addWidget(color_group)
        
        # Parameter controls
        self.create_parameters_panel(left_layout)
        
        # Erosion stage visualization
        erosion_group = QGroupBox("Erosion Stage Visualization")
        erosion_layout = QVBoxLayout(erosion_group)
        
        self.erosion_btn = QPushButton("Show Erosion Stages")
        self.erosion_btn.clicked.connect(self.show_erosion_stages)
        self.erosion_btn.setEnabled(False)
        
        erosion_layout.addWidget(self.erosion_btn)
        left_layout.addWidget(erosion_group)
        
        # File loading section - add before the continue button
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        self.load_image_btn = QPushButton("Select Main Image")
        self.load_mask_btn = QPushButton("Select Mask (Optional)")
        self.start_analysis_btn = QPushButton("Start Analysis")
        
        self.load_image_btn.clicked.connect(self.load_main_image)
        self.load_mask_btn.clicked.connect(self.load_mask_image)
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        
        # Initialize button states
        self.start_analysis_btn.setEnabled(False)
        
        file_layout.addWidget(self.load_image_btn)
        file_layout.addWidget(self.load_mask_btn)
        file_layout.addWidget(self.start_analysis_btn)
        
        left_layout.addWidget(file_group)
        
        # Batch processing section
        batch_group = QGroupBox("Batch Processing")
        batch_layout = QVBoxLayout(batch_group)
        
        self.select_folder_btn = QPushButton("Select Input Folder")
        self.batch_detect_btn = QPushButton("Batch Detect")
        self.batch_progress_label = QLabel("No folder selected")
        
        # Parameter save/load buttons
        params_layout = QHBoxLayout()
        self.save_params_btn = QPushButton("Save Parameters")
        self.load_params_btn = QPushButton("Load Parameters")
        self.save_params_btn.clicked.connect(self.save_parameters)
        self.load_params_btn.clicked.connect(self.load_parameters)
        params_layout.addWidget(self.save_params_btn)
        params_layout.addWidget(self.load_params_btn)
        
        self.select_folder_btn.clicked.connect(self.select_input_folder)
        self.batch_detect_btn.clicked.connect(self.run_batch_detection)
        
        # Initialize batch button states
        self.batch_detect_btn.setEnabled(False)
        
        batch_layout.addWidget(self.select_folder_btn)
        batch_layout.addWidget(self.batch_progress_label)
        batch_layout.addLayout(params_layout)
        batch_layout.addWidget(self.batch_detect_btn)
        
        left_layout.addWidget(batch_group)
        
        # Continue button
        self.continue_btn = QPushButton("Continue Analysis")
        self.continue_btn.clicked.connect(self.continue_analysis)
        self.continue_btn.setEnabled(False)
        left_layout.addWidget(self.continue_btn)
        
        # Status display
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        left_layout.addWidget(QLabel("Status:"))
        left_layout.addWidget(self.status_text)
        
        main_layout.addWidget(left_panel)
        
        # Right panel for image display
        self.image_label = QLabel()
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_label)
        
    def create_parameters_panel(self, layout):
        """Create parameter adjustment controls"""
        params_group = QGroupBox("Parameters")
        params_main_layout = QVBoxLayout(params_group)
        
        # Create scroll area for parameters
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        params_layout = QVBoxLayout(scroll_widget)
        
        # Dark threshold
        dark_group = QGroupBox("Dark Threshold")
        dark_layout = QGridLayout(dark_group)
        
        self.dark_slider = QSlider(Qt.Orientation.Horizontal)
        self.dark_slider.setRange(0, 255)
        self.dark_slider.setValue(self.analyzer.params.BINARY_DARK_THRESHOLD)
        self.dark_slider.valueChanged.connect(self.update_dark_threshold)
        
        self.dark_label = QLabel(str(self.analyzer.params.BINARY_DARK_THRESHOLD))
        
        dark_layout.addWidget(QLabel("Value:"), 0, 0)
        dark_layout.addWidget(self.dark_slider, 0, 1)
        dark_layout.addWidget(self.dark_label, 0, 2)
        
        params_layout.addWidget(dark_group)
        
        # Inpaint radius
        inpaint_group = QGroupBox("Inpaint Radius")
        inpaint_layout = QGridLayout(inpaint_group)
        
        self.inpaint_slider = QSlider(Qt.Orientation.Horizontal)
        self.inpaint_slider.setRange(1, 50)
        self.inpaint_slider.setValue(self.analyzer.params.BINARY_INPAINT_RADIUS)
        self.inpaint_slider.valueChanged.connect(self.update_inpaint_radius)
        
        self.inpaint_label = QLabel(str(self.analyzer.params.BINARY_INPAINT_RADIUS))
        
        inpaint_layout.addWidget(QLabel("Value:"), 0, 0)
        inpaint_layout.addWidget(self.inpaint_slider, 0, 1)
        inpaint_layout.addWidget(self.inpaint_label, 0, 2)
        
        params_layout.addWidget(inpaint_group)
        
        # Binary threshold
        binary_group = QGroupBox("Binary Threshold")
        binary_layout = QGridLayout(binary_group)
        
        self.binary_slider = QSlider(Qt.Orientation.Horizontal)
        self.binary_slider.setRange(0, 255)
        self.binary_slider.setValue(self.analyzer.params.BINARY_THRESHOLD)
        self.binary_slider.valueChanged.connect(self.update_binary_threshold)
        
        self.binary_label = QLabel(str(self.analyzer.params.BINARY_THRESHOLD))
        
        binary_layout.addWidget(QLabel("Value:"), 0, 0)
        binary_layout.addWidget(self.binary_slider, 0, 1)
        binary_layout.addWidget(self.binary_label, 0, 2)
        
        params_layout.addWidget(binary_group)
        
        # Erosion stages
        erosion_group = QGroupBox("Erosion Stages")
        erosion_layout = QGridLayout(erosion_group)
        
        self.erosion_slider = QSlider(Qt.Orientation.Horizontal)
        self.erosion_slider.setRange(0, 10)
        self.erosion_slider.setValue(self.analyzer.params.BINARY_EROSION_STAGES)
        self.erosion_slider.valueChanged.connect(self.update_erosion_stages)
        
        self.erosion_label = QLabel(str(self.analyzer.params.BINARY_EROSION_STAGES))
        
        erosion_layout.addWidget(QLabel("Stages:"), 0, 0)
        erosion_layout.addWidget(self.erosion_slider, 0, 1)
        erosion_layout.addWidget(self.erosion_label, 0, 2)
        
        params_layout.addWidget(erosion_group)
        
        # Dilation parameters
        dilation_group = QGroupBox("Dilation Parameters")
        dilation_layout = QGridLayout(dilation_group)
        
        # Kernel size
        self.kernel_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_size_slider.setRange(1, 15)
        self.kernel_size_slider.setValue(3)  # Default 3x3 kernel
        self.kernel_size_slider.valueChanged.connect(self.update_kernel_size)
        
        self.kernel_size_label = QLabel("3")
        
        dilation_layout.addWidget(QLabel("Kernel Size:"), 0, 0)
        dilation_layout.addWidget(self.kernel_size_slider, 0, 1)
        dilation_layout.addWidget(self.kernel_size_label, 0, 2)
        
        # Dilation iterations
        self.dilation_slider = QSlider(Qt.Orientation.Horizontal)
        self.dilation_slider.setRange(0, 10)
        self.dilation_slider.setValue(0)  # Default no dilation
        self.dilation_slider.valueChanged.connect(self.update_dilation_iterations)
        
        self.dilation_label = QLabel("0")
        
        dilation_layout.addWidget(QLabel("Dilation:"), 1, 0)
        dilation_layout.addWidget(self.dilation_slider, 1, 1)
        dilation_layout.addWidget(self.dilation_label, 1, 2)
        
        params_layout.addWidget(dilation_group)
        
        # Min/Max Diameter parameters
        size_group = QGroupBox("Size Parameters")
        size_layout = QGridLayout(size_group)
        
        # Min Diameter
        self.min_diameter_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_diameter_slider.setRange(1, 100)
        self.min_diameter_slider.setValue(self.analyzer.params.BINARY_MIN_DIAMETER)
        self.min_diameter_slider.valueChanged.connect(self.update_min_diameter)
        
        self.min_diameter_label = QLabel(str(self.analyzer.params.BINARY_MIN_DIAMETER))
        
        size_layout.addWidget(QLabel("Min Diameter:"), 0, 0)
        size_layout.addWidget(self.min_diameter_slider, 0, 1)
        size_layout.addWidget(self.min_diameter_label, 0, 2)
        
        # Max Diameter
        self.max_diameter_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_diameter_slider.setRange(10, 200)
        self.max_diameter_slider.setValue(self.analyzer.params.BINARY_MAX_DIAMETER)
        self.max_diameter_slider.valueChanged.connect(self.update_max_diameter)
        
        self.max_diameter_label = QLabel(str(self.analyzer.params.BINARY_MAX_DIAMETER))
        
        size_layout.addWidget(QLabel("Max Diameter:"), 1, 0)
        size_layout.addWidget(self.max_diameter_slider, 1, 1)
        size_layout.addWidget(self.max_diameter_label, 1, 2)
        
        params_layout.addWidget(size_group)
        
        # Shape parameters
        shape_group = QGroupBox("Shape Parameters")
        shape_layout = QGridLayout(shape_group)
        
        # Circularity
        self.circularity_slider = QSlider(Qt.Orientation.Horizontal)
        self.circularity_slider.setRange(0, 100)
        self.circularity_slider.setValue(int(self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD * 100))
        self.circularity_slider.valueChanged.connect(self.update_circularity)
        
        self.circularity_label = QLabel(f"{self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD:.2f}")
        
        shape_layout.addWidget(QLabel("Circularity:"), 0, 0)
        shape_layout.addWidget(self.circularity_slider, 0, 1)
        shape_layout.addWidget(self.circularity_label, 0, 2)
        
        params_layout.addWidget(shape_group)
        
        # Color detection parameters
        color_params_group = QGroupBox("Color Detection Parameters")
        color_params_layout = QGridLayout(color_params_group)
        
        # Color Min Diameter
        self.color_min_diameter_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_min_diameter_slider.setRange(1, 150)
        self.color_min_diameter_slider.setValue(self.analyzer.params.COLOR_MIN_DIAMETER)
        self.color_min_diameter_slider.valueChanged.connect(self.update_color_min_diameter)
        
        self.color_min_diameter_label = QLabel(str(self.analyzer.params.COLOR_MIN_DIAMETER))
        
        color_params_layout.addWidget(QLabel("Color Min Diameter:"), 0, 0)
        color_params_layout.addWidget(self.color_min_diameter_slider, 0, 1)
        color_params_layout.addWidget(self.color_min_diameter_label, 0, 2)
        
        # Color Max Diameter
        self.color_max_diameter_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_max_diameter_slider.setRange(20, 300)
        self.color_max_diameter_slider.setValue(self.analyzer.params.COLOR_MAX_DIAMETER)
        self.color_max_diameter_slider.valueChanged.connect(self.update_color_max_diameter)
        
        self.color_max_diameter_label = QLabel(str(self.analyzer.params.COLOR_MAX_DIAMETER))
        
        color_params_layout.addWidget(QLabel("Color Max Diameter:"), 1, 0)
        color_params_layout.addWidget(self.color_max_diameter_slider, 1, 1)
        color_params_layout.addWidget(self.color_max_diameter_label, 1, 2)
        
        # Color Circularity
        self.color_circularity_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_circularity_slider.setRange(0, 100)
        self.color_circularity_slider.setValue(int(self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD * 100))
        self.color_circularity_slider.valueChanged.connect(self.update_color_circularity)
        
        self.color_circularity_label = QLabel(f"{self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD:.2f}")
        
        color_params_layout.addWidget(QLabel("Color Circularity:"), 2, 0)
        color_params_layout.addWidget(self.color_circularity_slider, 2, 1)
        color_params_layout.addWidget(self.color_circularity_label, 2, 2)
        
        # Color Erosion Stages
        self.color_erosion_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_erosion_slider.setRange(0, 15)
        self.color_erosion_slider.setValue(self.analyzer.params.COLOR_EROSION_STAGES)
        self.color_erosion_slider.valueChanged.connect(self.update_color_erosion)
        
        self.color_erosion_label = QLabel(str(self.analyzer.params.COLOR_EROSION_STAGES))
        
        color_params_layout.addWidget(QLabel("Color Erosion:"), 3, 0)
        color_params_layout.addWidget(self.color_erosion_slider, 3, 1)
        color_params_layout.addWidget(self.color_erosion_label, 3, 2)
        
        # Gaussian Blur
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(1, 15)
        self.blur_slider.setValue(self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE)
        self.blur_slider.valueChanged.connect(self.update_blur_size)
        
        self.blur_label = QLabel(str(self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE))
        
        color_params_layout.addWidget(QLabel("Gaussian Blur:"), 4, 0)
        color_params_layout.addWidget(self.blur_slider, 4, 1)
        color_params_layout.addWidget(self.blur_label, 4, 2)
        
        # Canny Low Threshold
        self.canny_low_slider = QSlider(Qt.Orientation.Horizontal)
        self.canny_low_slider.setRange(1, 200)
        self.canny_low_slider.setValue(self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD)
        self.canny_low_slider.valueChanged.connect(self.update_canny_low)
        
        self.canny_low_label = QLabel(str(self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD))
        
        color_params_layout.addWidget(QLabel("Canny Low:"), 5, 0)
        color_params_layout.addWidget(self.canny_low_slider, 5, 1)
        color_params_layout.addWidget(self.canny_low_label, 5, 2)
        
        # Canny High Threshold
        self.canny_high_slider = QSlider(Qt.Orientation.Horizontal)
        self.canny_high_slider.setRange(10, 300)
        self.canny_high_slider.setValue(self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD)
        self.canny_high_slider.valueChanged.connect(self.update_canny_high)
        
        self.canny_high_label = QLabel(str(self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD))
        
        color_params_layout.addWidget(QLabel("Canny High:"), 6, 0)
        color_params_layout.addWidget(self.canny_high_slider, 6, 1)
        color_params_layout.addWidget(self.canny_high_label, 6, 2)
        
        params_layout.addWidget(color_params_group)
        
        # Merge threshold
        merge_group = QGroupBox("Merge Parameters")
        merge_layout = QGridLayout(merge_group)
        
        self.merge_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.merge_threshold_slider.setRange(1, 100)
        self.merge_threshold_slider.setValue(self.analyzer.params.CENTROID_MERGE_THRESHOLD)
        self.merge_threshold_slider.valueChanged.connect(self.update_merge_threshold)
        
        self.merge_threshold_label = QLabel(str(self.analyzer.params.CENTROID_MERGE_THRESHOLD))
        
        merge_layout.addWidget(QLabel("Merge Threshold:"), 0, 0)
        merge_layout.addWidget(self.merge_threshold_slider, 0, 1)
        merge_layout.addWidget(self.merge_threshold_label, 0, 2)
        
        params_layout.addWidget(merge_group)
        
        scroll_widget.setLayout(params_layout)
        scroll_area.setWidget(scroll_widget)
        params_main_layout.addWidget(scroll_area)
        
        layout.addWidget(params_group)
        
    def load_images(self):
        """Load and display the original image"""
        if self.analyzer.original_image is not None:
            self.display_image(self.analyzer.original_image)
            self.log_status("✅ Images loaded successfully")
            self.log_status(f"Image size: {self.analyzer.width} x {self.analyzer.height}")
        else:
            self.log_status("❌ No image loaded")
    
    def load_main_image(self):
        """Load main image via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Main Image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All Files (*)"
        )
        
        if file_path:
            # Clear any existing processing images to avoid confusion
            self.processing_images.clear()
            self.current_step = 0
            self.binary_complete = False
            self.color_complete = False
            
            # Log the exact file being loaded
            self.log_status(f"🔍 Loading file: {file_path}")
            
            # Load the image directly from the selected file (preserving transparency)
            loaded_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if loaded_image is not None:
                # Store the loaded image
                self.analyzer.original_image = loaded_image.copy()  # Make a copy to ensure no reference issues
                self.analyzer.height, self.analyzer.width = self.analyzer.original_image.shape[:2]
                self.analyzer.image_path = file_path
                self.analyzer.zoom_center_x = self.analyzer.width // 2
                self.analyzer.zoom_center_y = self.analyzer.height // 2
                
                # Check and log transparency status
                has_alpha = len(self.analyzer.original_image.shape) == 3 and self.analyzer.original_image.shape[2] == 4
                if has_alpha:
                    alpha_channel = self.analyzer.original_image[:, :, 3]
                    transparent_pixels = np.sum(alpha_channel < 255)
                    total_pixels = self.analyzer.height * self.analyzer.width
                    transparency_percentage = 100 * transparent_pixels / total_pixels
                    self.log_status(f"🎨 Image has transparency: {transparent_pixels}/{total_pixels} pixels ({transparency_percentage:.1f}%)")
                else:
                    self.log_status(f"ℹ️  Image is regular RGB (no transparency)")
                
                # Clear any zoom/pan state
                self.analyzer.zoom_factor = self.analyzer.params.DISPLAY_ZOOM_FACTOR
                self.analyzer.display_img = None
                
                # Verify the image content by checking a few pixels
                test_pixel = self.analyzer.original_image[0, 0]  # Top-left pixel
                center_pixel = self.analyzer.original_image[self.analyzer.height//2, self.analyzer.width//2]  # Center pixel
                self.log_status(f"   🔍 Image verification - Top-left pixel: {test_pixel}, Center pixel: {center_pixel}")
                
                # Display the loaded image immediately
                self.display_image(self.analyzer.original_image)
                
                # Log success with detailed info
                self.log_status(f"✅ Loaded main image: {os.path.basename(file_path)}")
                self.log_status(f"   Full path: {file_path}")
                self.log_status(f"   Size: {self.analyzer.width} x {self.analyzer.height}")
                self.log_status(f"   File size: {os.path.getsize(file_path)} bytes")
                
                # Determine if this looks like a well crop
                is_likely_crop = (
                    self.analyzer.width < 500 or self.analyzer.height < 500 or
                    "well" in os.path.basename(file_path).lower() or
                    "crop" in os.path.basename(file_path).lower()
                )
                
                if is_likely_crop:
                    self.log_status(f"   🔬 Detected as well crop (small size or filename)")
                    self.log_status(f"   💡 Tip: Use simplified parameters for well crops")
                else:
                    self.log_status(f"   🧬 Detected as full plate image")
                
                # Enable start analysis button
                self.start_analysis_btn.setEnabled(True)
                self.start_analysis_btn.setText("Start Analysis")
                
                # Reset continue button
                self.continue_btn.setEnabled(False)
                
            else:
                self.log_status(f"❌ Could not load image: {file_path}")
                self.log_status(f"   File exists: {os.path.exists(file_path)}")
                if os.path.exists(file_path):
                    self.log_status(f"   File size: {os.path.getsize(file_path)} bytes")
        else:
            self.log_status("❌ No image selected")
    
    def load_mask_image(self):
        """Load mask image via file dialog"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mask Image (Optional)",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All Files (*)"
        )
        
        if file_path:
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                self.analyzer.mask_path = file_path
                self.log_status(f"✅ Loaded mask: {file_path}")
                self.log_status(f"   Size: {mask.shape[1]} x {mask.shape[0]}")
            else:
                self.log_status(f"❌ Could not load mask: {file_path}")
        else:
            self.log_status("❌ No mask selected (optional)")
    
    def start_analysis(self):
        """Start the analysis with loaded images"""
        if self.analyzer.original_image is None:
            self.log_status("❌ Please load a main image first")
            return
        
        # Set a dummy mask path if none provided
        if not hasattr(self.analyzer, 'mask_path') or not self.analyzer.mask_path:
            self.analyzer.mask_path = "dummy_mask.png"
        
        self.log_status("🚀 Starting analysis...")
        self.log_status("📋 Use the step buttons to go through binary detection")
        self.log_status("🎨 Use 'Sample Colors' for color detection when ready")
        
        # ALPHA CROPPING APPROACH: Keep alpha intact but crop out as much transparent area as possible
        if len(self.analyzer.original_image.shape) == 3 and self.analyzer.original_image.shape[2] == 4:
            self.log_status("🎨 Alpha channel detected - cropping transparent regions...")
            
            # Find bounding box of non-transparent pixels
            alpha_channel = self.analyzer.original_image[:, :, 3]
            non_transparent_mask = alpha_channel > 0
            
            if np.any(non_transparent_mask):
                # Find the bounding box
                rows = np.any(non_transparent_mask, axis=1)
                cols = np.any(non_transparent_mask, axis=0)
                
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                
                # Add small padding to avoid edge artifacts
                padding = 5
                y_min = max(0, y_min - padding)
                y_max = min(self.analyzer.original_image.shape[0] - 1, y_max + padding)
                x_min = max(0, x_min - padding)
                x_max = min(self.analyzer.original_image.shape[1] - 1, x_max + padding)
                
                # Store crop info for later use
                self.crop_info = {
                    'x_min': x_min, 'x_max': x_max,
                    'y_min': y_min, 'y_max': y_max,
                    'original_shape': self.analyzer.original_image.shape
                }
                
                # Crop the image to the bounding box
                cropped_image = self.analyzer.original_image[y_min:y_max+1, x_min:x_max+1]
                self.analyzer.original_image = cropped_image
                
                original_pixels = self.crop_info['original_shape'][0] * self.crop_info['original_shape'][1]
                cropped_pixels = cropped_image.shape[0] * cropped_image.shape[1]
                reduction_percent = (1 - cropped_pixels / original_pixels) * 100
                
                self.log_status(f"   ✂️ Cropped from {self.crop_info['original_shape'][:2]} to {cropped_image.shape[:2]}")
                self.log_status(f"   📊 Reduced processing area by {reduction_percent:.1f}% ({original_pixels:,} → {cropped_pixels:,} pixels)")
                self.log_status(f"   🎯 Alpha channel preserved in cropped region")
            else:
                self.log_status("   ⚠️ Image is completely transparent - using full image")
                self.crop_info = None
            
            self.alpha_mask = None  # Not using simple masking approach
        else:
            self.crop_info = None
            self.alpha_mask = None
            self.log_status("📷 Regular RGB image - full image analysis")
        
        # Display the loaded image
        self.display_image(self.analyzer.original_image)
        
        # Initialize file loading state
        self.start_analysis_btn.setEnabled(False)
        self.start_analysis_btn.setText("Analysis Started")
    
    def step1_grayscale(self):
        """Step 1: Convert to grayscale"""
        if self.analyzer.original_image is None:
            self.log_status("❌ No image loaded")
            return
        
        # Get RGB version for processing (removes alpha if present)
        rgb_image = self.get_rgb_for_processing(self.analyzer.original_image)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Store the grayscale image (cropped if alpha was present)
        self.processing_images['gray'] = gray
        
        # Display with transparency preserved if original had alpha
        if len(self.analyzer.original_image.shape) == 3 and self.analyzer.original_image.shape[2] == 4:
            # Show grayscale with alpha channel for display
            gray_display = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
            gray_display[:, :, 0] = gray
            gray_display[:, :, 1] = gray  
            gray_display[:, :, 2] = gray
            gray_display[:, :, 3] = self.analyzer.original_image[:, :, 3]
            self.display_image(gray_display)
        else:
            # Regular grayscale to BGR
            gray_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            self.display_image(gray_display)
        
        self.current_step = 1
        self.log_status("✅ Step 1: Converted to grayscale")
        if hasattr(self, 'crop_info') and self.crop_info is not None:
            self.log_status("   ✂️ Working on cropped alpha image")
        
    def step2_dark_mask(self):
        """Step 2: Create dark mask"""
        if 'gray' not in self.processing_images:
            self.log_status("❌ Run Step 1 first")
            return
            
        gray = self.processing_images['gray']
        _, dark_mask = cv2.threshold(gray, self.analyzer.params.BINARY_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        self.processing_images['dark_mask'] = dark_mask
        
        # Display with transparency preserved if original had alpha
        if len(self.analyzer.original_image.shape) == 3 and self.analyzer.original_image.shape[2] == 4:
            # Show mask with alpha channel for display
            mask_display = np.zeros((dark_mask.shape[0], dark_mask.shape[1], 4), dtype=np.uint8)
            mask_display[:, :, 0] = dark_mask
            mask_display[:, :, 1] = dark_mask
            mask_display[:, :, 2] = dark_mask
            mask_display[:, :, 3] = self.analyzer.original_image[:, :, 3]
            self.display_image(mask_display)
        else:
            # Regular mask to BGR
            mask_display = cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR)
            self.display_image(mask_display)
        
        self.current_step = 2
        self.log_status(f"✅ Step 2: Created dark mask (threshold: {self.analyzer.params.BINARY_DARK_THRESHOLD})")
        
    def step3_inpainting(self):
        """Step 3: Inpaint dark areas"""
        if 'dark_mask' not in self.processing_images:
            self.log_status("❌ Run Step 2 first")
            return
            
        dark_mask = self.processing_images['dark_mask']
        # Get RGB version for inpainting (inpaint doesn't work with alpha channels)
        rgb_image = self.get_rgb_for_processing(self.analyzer.original_image)
        inpainted = cv2.inpaint(rgb_image, dark_mask, self.analyzer.params.BINARY_INPAINT_RADIUS, cv2.INPAINT_TELEA)
        
        self.processing_images['inpainted'] = inpainted
        
        # Display with transparency preserved if original had alpha
        if len(self.analyzer.original_image.shape) == 3 and self.analyzer.original_image.shape[2] == 4:
            # Create RGBA version for display
            inpainted_display = np.zeros((inpainted.shape[0], inpainted.shape[1], 4), dtype=np.uint8)
            inpainted_display[:, :, :3] = inpainted
            inpainted_display[:, :, 3] = self.analyzer.original_image[:, :, 3]
            self.display_image(inpainted_display)
        else:
            self.display_image(inpainted)
        
        self.current_step = 3
        self.log_status(f"✅ Step 3: Inpainted dark areas (radius: {self.analyzer.params.BINARY_INPAINT_RADIUS})")
        
    def step4_inpaint_gray(self):
        """Step 4: Convert inpainted to grayscale"""
        if 'inpainted' not in self.processing_images:
            self.log_status("❌ Run Step 3 first")
            return
            
        inpainted = self.processing_images['inpainted']
        inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
        
        self.processing_images['inpainted_gray'] = inpainted_gray
        
        # Display with transparency preserved if original had alpha
        if len(self.analyzer.original_image.shape) == 3 and self.analyzer.original_image.shape[2] == 4:
            # Show grayscale with alpha channel for display
            gray_display = np.zeros((inpainted_gray.shape[0], inpainted_gray.shape[1], 4), dtype=np.uint8)
            gray_display[:, :, 0] = inpainted_gray
            gray_display[:, :, 1] = inpainted_gray
            gray_display[:, :, 2] = inpainted_gray
            gray_display[:, :, 3] = self.analyzer.original_image[:, :, 3]
            self.display_image(gray_display)
        else:
            # Regular grayscale to BGR
            gray_display = cv2.cvtColor(inpainted_gray, cv2.COLOR_GRAY2BGR)
            self.display_image(gray_display)
        
        self.current_step = 4
        self.log_status("✅ Step 4: Converted inpainted to grayscale")
        
    def step5_binary(self):
        """Step 5: Apply binary threshold"""
        if 'inpainted_gray' not in self.processing_images:
            self.log_status("❌ Run Step 4 first")
            return
            
        inpainted_gray = self.processing_images['inpainted_gray']
        _, binary_plate = cv2.threshold(inpainted_gray, self.analyzer.params.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        self.processing_images['binary_plate'] = binary_plate
        
        # Display with transparency preserved if original had alpha
        if len(self.analyzer.original_image.shape) == 3 and self.analyzer.original_image.shape[2] == 4:
            # Show binary with alpha channel for display
            binary_display = np.zeros((binary_plate.shape[0], binary_plate.shape[1], 4), dtype=np.uint8)
            binary_display[:, :, 0] = binary_plate
            binary_display[:, :, 1] = binary_plate
            binary_display[:, :, 2] = binary_plate
            binary_display[:, :, 3] = self.analyzer.original_image[:, :, 3]
            self.display_image(binary_display)
        else:
            # Regular binary to BGR
            binary_display = cv2.cvtColor(binary_plate, cv2.COLOR_GRAY2BGR)
            self.display_image(binary_display)
        
        self.current_step = 5
        self.log_status(f"✅ Step 5: Applied binary threshold ({self.analyzer.params.BINARY_THRESHOLD})")
        if hasattr(self, 'crop_info') and self.crop_info is not None:
            self.log_status("   ✂️ Binary ready for detection on cropped alpha image")
        
    def step6_detect_binary(self):
        """Step 6: Detect binary centroids with morphological operations"""
        if 'binary_plate' not in self.processing_images:
            self.log_status("❌ Run Step 5 first")
            return
            
        binary_plate = self.processing_images['binary_plate']
        
        # Find centroids directly from binary plate (like the working version)
        # The find_binary_centroids method will handle the inversion and erosion internally
        centroids = self.analyzer.find_binary_centroids(binary_plate)
        self.analyzer.binary_centroids = centroids
        self.processing_images['binary_centroids'] = centroids
        
        # Display original image with centroids (preserve transparency)
        result_img = self.analyzer.original_image.copy()
        
        # Draw centroids on RGB channels only (skip alpha if present)
        rgb_result = self.get_rgb_for_processing(result_img)
        for cx, cy in centroids:
            cv2.circle(rgb_result, (cx, cy), 8, (255, 0, 0), 2)  # Blue circles
            cv2.circle(rgb_result, (cx, cy), 2, (255, 0, 0), -1)
        
        # Preserve alpha channel if original had it
        result_img = self.preserve_alpha_in_result(rgb_result, self.analyzer.original_image)
        
        self.display_image(result_img)
        
        self.current_step = 6
        self.binary_complete = True
        self.erosion_btn.setEnabled(True)  # Enable erosion visualization
        self.log_status(f"✅ Step 6: Detected {len(centroids)} binary centroids")
        self.log_status(f"   Erosion stages: {self.analyzer.params.BINARY_EROSION_STAGES}")
        self.log_status(f"   Min diameter: {self.analyzer.params.BINARY_MIN_DIAMETER}")
        self.log_status(f"   Max diameter: {self.analyzer.params.BINARY_MAX_DIAMETER}")
        self.log_status(f"   Circularity threshold: {self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD}")
        self.check_continue_ready()
        
    def apply_morphological_operations(self, binary_image):
        """Apply erosion and dilation operations to binary image for visualization only"""
        # This method is now used only for erosion stage visualization
        # The actual detection uses the method from the simple analyzer
        
        # Invert binary image (organoids should be white)
        inverted_binary = cv2.bitwise_not(binary_image)
        
        # Create kernel
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        # Apply erosion stages
        if self.analyzer.params.BINARY_EROSION_STAGES > 0:
            inverted_binary = cv2.erode(inverted_binary, kernel, iterations=self.analyzer.params.BINARY_EROSION_STAGES)
        
        # Apply dilation if specified
        if self.dilation_iterations > 0:
            inverted_binary = cv2.dilate(inverted_binary, kernel, iterations=self.dilation_iterations)
        
        return cv2.bitwise_not(inverted_binary)  # Convert back to original polarity
        
    def show_erosion_stages(self):
        """Show visualization of erosion stages"""
        if 'binary_plate' not in self.processing_images:
            self.log_status("❌ No binary image available for erosion visualization")
            return
            
        import matplotlib.pyplot as plt
        
        binary_plate = self.processing_images['binary_plate']
        inverted_binary = cv2.bitwise_not(binary_plate)
        
        # Create kernel
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        # Calculate number of stages to show
        max_stages = min(self.analyzer.params.BINARY_EROSION_STAGES + 2, 8)
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Erosion Stages Visualization (Kernel: {self.kernel_size}x{self.kernel_size})', fontsize=14)
        
        axes = axes.flatten()
        
        # Show original
        axes[0].imshow(inverted_binary, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original Binary')
        axes[0].axis('off')
        axes[0].set_aspect('equal')
        
        # Show erosion stages
        current_image = inverted_binary.copy()
        for i in range(1, max_stages):
            if i <= self.analyzer.params.BINARY_EROSION_STAGES:
                current_image = cv2.erode(current_image, kernel, iterations=1)
                title = f'Erosion Stage {i}'
            else:
                title = f'Future Stage {i}'
                current_image = cv2.erode(current_image, kernel, iterations=1)
            
            axes[i].imshow(current_image, cmap='gray', vmin=0, vmax=255)
            axes[i].set_title(title)
            axes[i].axis('off')
            axes[i].set_aspect('equal')
            
            # Count objects at this stage
            contours, _ = cv2.findContours(current_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
            axes[i].text(0.02, 0.98, f'Objects: {len(valid_contours)}', 
                        transform=axes[i].transAxes, fontsize=10, 
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Apply dilation visualization if enabled
        if self.dilation_iterations > 0:
            # Start from the erosion result
            eroded_image = inverted_binary.copy()
            if self.analyzer.params.BINARY_EROSION_STAGES > 0:
                eroded_image = cv2.erode(eroded_image, kernel, iterations=self.analyzer.params.BINARY_EROSION_STAGES)
            
            # Show dilation stages
            current_dilated = eroded_image.copy()
            for j in range(min(self.dilation_iterations, max_stages - self.analyzer.params.BINARY_EROSION_STAGES - 1)):
                stage_idx = self.analyzer.params.BINARY_EROSION_STAGES + 1 + j
                if stage_idx < max_stages:
                    current_dilated = cv2.dilate(current_dilated, kernel, iterations=1)
                    axes[stage_idx].imshow(current_dilated, cmap='gray', vmin=0, vmax=255)
                    axes[stage_idx].set_title(f'Dilation {j+1}')
                    axes[stage_idx].axis('off')
                    axes[stage_idx].set_aspect('equal')
                    
                    # Count objects
                    contours, _ = cv2.findContours(current_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
                    axes[stage_idx].text(0.02, 0.98, f'Objects: {len(valid_contours)}', 
                                        transform=axes[stage_idx].transAxes, fontsize=10, 
                                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        self.log_status("📊 Erosion stages visualization displayed")
        
    def color_step_sample(self):
        """Sample colors for color detection"""
        self.log_status("🎨 Starting color sampling...")
        self.log_status("Use the color sampling interface to collect samples")
        
        # Run color sampling
        self.analyzer.run_color_sampling()
        
        if self.analyzer.sample_masks:
            self.log_status(f"✅ Collected {len(self.analyzer.sample_masks)} color samples")
        else:
            self.log_status("❌ No color samples collected")
        
    def color_step_detect(self):
        """Detect color centroids"""
        if not self.analyzer.sample_masks:
            self.log_status("❌ No color samples available. Run 'Sample Colors' first.")
            return
            
        centroids = self.analyzer.process_color_samples_for_centroids()
        self.analyzer.color_centroids = centroids
        
        # Display original image with all centroids (preserve transparency)
        result_img = self.analyzer.original_image.copy()
        
        # Draw centroids on RGB channels only (skip alpha if present)
        rgb_result = self.get_rgb_for_processing(result_img)
        
        # Draw binary centroids in blue
        for cx, cy in self.analyzer.binary_centroids:
            cv2.circle(rgb_result, (cx, cy), 8, (255, 0, 0), 2)
            cv2.circle(rgb_result, (cx, cy), 2, (255, 0, 0), -1)
        
        # Draw color centroids in red
        for cx, cy in centroids:
            cv2.circle(rgb_result, (cx, cy), 8, (0, 0, 255), 2)  # Red circles
            cv2.circle(rgb_result, (cx, cy), 2, (0, 0, 255), -1)
        
        # Preserve alpha channel if original had it
        result_img = self.preserve_alpha_in_result(rgb_result, self.analyzer.original_image)
        
        self.display_image(result_img)
        
        self.color_complete = True
        self.log_status(f"✅ Color detection: Found {len(centroids)} centroids")
        self.check_continue_ready()
        
    def skip_binary(self):
        """Skip binary detection"""
        self.analyzer.binary_centroids = []
        self.binary_complete = True
        self.log_status("⏭️ Skipped binary detection")
        self.check_continue_ready()
        
    def skip_color(self):
        """Skip color detection"""
        self.analyzer.color_centroids = []
        self.color_complete = True
        self.log_status("⏭️ Skipped color detection")
        self.check_continue_ready()
        
    def check_continue_ready(self):
        """Check if ready to continue analysis"""
        if self.binary_complete and self.color_complete:
            self.continue_btn.setEnabled(True)
            total_centroids = len(self.analyzer.binary_centroids) + len(self.analyzer.color_centroids)
            self.log_status(f"🎯 Ready to continue! Total centroids: {total_centroids}")
        
    def update_dark_threshold(self, value):
        """Update dark threshold parameter"""
        self.analyzer.params.BINARY_DARK_THRESHOLD = value
        self.dark_label.setText(str(value))
        
        # Re-run step 2 if we're past it
        if self.current_step >= 2:
            self.step2_dark_mask()
            
    def update_inpaint_radius(self, value):
        """Update inpaint radius parameter"""
        self.analyzer.params.BINARY_INPAINT_RADIUS = value
        self.inpaint_label.setText(str(value))
        
        # Re-run step 3 if we're past it
        if self.current_step >= 3:
            self.step3_inpainting()
            
    def update_binary_threshold(self, value):
        """Update binary threshold parameter"""
        self.analyzer.params.BINARY_THRESHOLD = value
        self.binary_label.setText(str(value))
        
        # Re-run step 5 if we're past it
        if self.current_step >= 5:
            self.step5_binary()
            
    def update_erosion_stages(self, value):
        """Update erosion stages parameter"""
        self.analyzer.params.BINARY_EROSION_STAGES = value
        self.erosion_label.setText(str(value))
        
        # Re-run step 6 if we're past it
        if self.current_step >= 6:
            self.step6_detect_binary()
            
        # Enable erosion visualization if we have a binary image
        if 'binary_plate' in self.processing_images:
            self.erosion_btn.setEnabled(True)
            
    def update_kernel_size(self, value):
        """Update morphological kernel size"""
        # Ensure odd number for kernel size
        if value % 2 == 0:
            value += 1
        self.kernel_size_slider.setValue(value)
        self.kernel_size_label.setText(str(value))
        
        # Store kernel size for operations
        self.kernel_size = value
        
        # Re-run operations if needed
        if self.current_step >= 6:
            self.step6_detect_binary()
            
    def update_dilation_iterations(self, value):
        """Update dilation iterations"""
        self.dilation_label.setText(str(value))
        self.dilation_iterations = value
        
        # Re-run operations if needed
        if self.current_step >= 6:
            self.step6_detect_binary()
    
    def update_min_diameter(self, value):
        """Update minimum diameter parameter"""
        self.analyzer.params.BINARY_MIN_DIAMETER = value
        self.min_diameter_label.setText(str(value))
        
        # Re-run step 6 if we're past it
        if self.current_step >= 6:
            self.step6_detect_binary()
            
    def update_max_diameter(self, value):
        """Update maximum diameter parameter"""
        self.analyzer.params.BINARY_MAX_DIAMETER = value
        self.max_diameter_label.setText(str(value))
        
        # Re-run step 6 if we're past it
        if self.current_step >= 6:
            self.step6_detect_binary()
            
    def update_circularity(self, value):
        """Update circularity threshold parameter"""
        threshold = value / 100.0
        self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD = threshold
        self.circularity_label.setText(f"{threshold:.2f}")
        
        # Re-run step 6 if we're past it
        if self.current_step >= 6:
            self.step6_detect_binary()
            
    def update_color_min_diameter(self, value):
        """Update color detection minimum diameter"""
        self.analyzer.params.COLOR_MIN_DIAMETER = value
        self.color_min_diameter_label.setText(str(value))
        
    def update_color_max_diameter(self, value):
        """Update color detection maximum diameter"""
        self.analyzer.params.COLOR_MAX_DIAMETER = value
        self.color_max_diameter_label.setText(str(value))
        
    def update_color_circularity(self, value):
        """Update color detection circularity threshold"""
        threshold = value / 100.0
        self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD = threshold
        self.color_circularity_label.setText(f"{threshold:.2f}")
        
    def update_color_erosion(self, value):
        """Update color detection erosion stages"""
        self.analyzer.params.COLOR_EROSION_STAGES = value
        self.color_erosion_label.setText(str(value))
        
    def update_blur_size(self, value):
        """Update Gaussian blur size"""
        # Ensure odd number for blur size
        if value % 2 == 0:
            value += 1
        self.blur_slider.setValue(value)
        self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE = value
        self.blur_label.setText(str(value))
        
    def update_canny_low(self, value):
        """Update Canny low threshold"""
        self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD = value
        self.canny_low_label.setText(str(value))
        
        # Ensure low threshold is less than high threshold
        if value >= self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD:
            new_high = value + 10
            self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD = new_high
            self.canny_high_slider.setValue(new_high)
            self.canny_high_label.setText(str(new_high))
        
    def update_canny_high(self, value):
        """Update Canny high threshold"""
        self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD = value
        self.canny_high_label.setText(str(value))
        
        # Ensure high threshold is greater than low threshold
        if value <= self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD:
            new_low = max(1, value - 10)
            self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD = new_low
            self.canny_low_slider.setValue(new_low)
            self.canny_low_label.setText(str(new_low))
            
    def update_merge_threshold(self, value):
        """Update centroid merge threshold"""
        self.analyzer.params.CENTROID_MERGE_THRESHOLD = value
        self.merge_threshold_label.setText(str(value))
        
    def display_image(self, cv_image):
        """Display OpenCV image in the QLabel with transparency support"""
        if cv_image is None:
            self.log_status("⚠️ display_image called with None image")
            return
        
        # Log image info for debugging
        has_alpha = len(cv_image.shape) == 3 and cv_image.shape[2] == 4
        channels_info = f"RGBA" if has_alpha else f"RGB/BGR" if len(cv_image.shape) == 3 else "Grayscale"
        self.log_status(f"🖼️ Displaying image: {cv_image.shape[1]}x{cv_image.shape[0]} ({channels_info})")
        
        # Handle different image formats
        if len(cv_image.shape) == 2:
            # Grayscale image
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
            qt_format = QImage.Format.Format_RGB888
            
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            # Regular BGR/RGB image
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            qt_format = QImage.Format.Format_RGB888
            
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
            # RGBA image with transparency
            # Convert BGRA to RGBA
            rgba_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA)
            rgb_image = rgba_image  # Keep as RGBA for transparency support
            qt_format = QImage.Format.Format_RGBA8888
            self.log_status("🎨 Image has transparency - preserving alpha channel in display")
            
        else:
            self.log_status(f"⚠️ Unsupported image format: {cv_image.shape}")
            return
        
        h, w = rgb_image.shape[:2]
        ch = rgb_image.shape[2] if len(rgb_image.shape) == 3 else 1
        bytes_per_line = ch * w
        
        # Create QImage with proper format
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, qt_format)
        
        # Scale to fit label while maintaining aspect ratio
        label_size = self.image_label.size()
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        self.image_label.setPixmap(scaled_pixmap)
        
        # Log successful display with transparency info
        transparency_info = " (with transparency)" if has_alpha else ""
        self.log_status(f"✅ Image displayed: {scaled_pixmap.width()}x{scaled_pixmap.height()} (scaled){transparency_info}")
        
    def log_status(self, message):
        """Add message to status log"""
        self.status_text.append(message)
        self.status_text.ensureCursorVisible()
        
    def continue_analysis(self):
        """Continue with the rest of the analysis"""
        self.log_status("🚀 Continuing with complete analysis...")
        
        # Create visualization
        self.analyzer.create_image_with_organoids()
        
        # Create well crops and correlate if enabled
        if self.analyzer.params.ENABLE_WELL_ANALYSIS:
            self.analyzer.create_well_crops()
            self.analyzer.correlate_centroids_to_wells()
        
        # Save results
        self.analyzer.save_results_csv()
        
        # Show final results
        total_detections = len(self.analyzer.binary_centroids) + len(self.analyzer.color_centroids)
        self.log_status(f"✅ Analysis complete!")
        self.log_status(f"   Binary: {len(self.analyzer.binary_centroids)}")
        self.log_status(f"   Color: {len(self.analyzer.color_centroids)}")
        self.log_status(f"   Total: {total_detections}")
        
        # Close the GUI
        self.close()
        
    def closeEvent(self, event):
        """Handle window close event"""
        cv2.destroyAllWindows()
        event.accept()

    def select_input_folder(self):
        """Select input folder for batch processing"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Input Folder",
            ""
        )
        
        if folder_path:
            self.input_folder = folder_path
            
            # Find all image files in the folder (avoid duplicates)
            self.batch_images = []
            
            # Use os.listdir to get all files, then filter by extension
            try:
                all_files = os.listdir(folder_path)
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                
                for file in all_files:
                    file_lower = file.lower()
                    if any(file_lower.endswith(ext) for ext in image_extensions):
                        full_path = os.path.join(folder_path, file)
                        if os.path.isfile(full_path):  # Make sure it's actually a file
                            self.batch_images.append(full_path)
                
                self.batch_images.sort()  # Sort for consistent processing order
                
                if self.batch_images:
                    self.batch_progress_label.setText(f"Found {len(self.batch_images)} images")
                    self.batch_detect_btn.setEnabled(True)
                    self.log_status(f"✅ Selected folder: {folder_path}")
                    self.log_status(f"   Found {len(self.batch_images)} images to process")
                    
                    # Log first few files for verification
                    self.log_status("   Sample files:")
                    for i, img_path in enumerate(self.batch_images[:5]):
                        self.log_status(f"     {i+1}. {os.path.basename(img_path)}")
                    if len(self.batch_images) > 5:
                        self.log_status(f"     ... and {len(self.batch_images) - 5} more")
                        
                else:
                    self.batch_progress_label.setText("No images found in folder")
                    self.batch_detect_btn.setEnabled(False)
                    self.log_status(f"❌ No images found in folder: {folder_path}")
                    
            except Exception as e:
                self.batch_progress_label.setText("Error reading folder")
                self.batch_detect_btn.setEnabled(False)
                self.log_status(f"❌ Error reading folder: {str(e)}")
        else:
            self.log_status("❌ No folder selected")
    
    def run_batch_detection(self):
        """Run batch detection on all images in the selected folder"""
        if not self.input_folder or not self.batch_images:
            self.log_status("❌ No folder or images selected for batch processing")
            return
        
        import os
        from datetime import datetime
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(self.input_folder, f"batch_results_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        self.log_status(f"🚀 Starting batch detection on {len(self.batch_images)} images...")
        self.log_status(f"📁 Output directory: {output_dir}")
        
        # Log current parameters being used
        self.log_status("🔧 Current Parameters:")
        self.log_status(f"   Binary Dark Threshold: {self.analyzer.params.BINARY_DARK_THRESHOLD}")
        self.log_status(f"   Binary Threshold: {self.analyzer.params.BINARY_THRESHOLD}")
        self.log_status(f"   Inpaint Radius: {self.analyzer.params.BINARY_INPAINT_RADIUS}")
        self.log_status(f"   Min/Max Diameter: {self.analyzer.params.BINARY_MIN_DIAMETER}-{self.analyzer.params.BINARY_MAX_DIAMETER}")
        self.log_status(f"   Erosion Stages: {self.analyzer.params.BINARY_EROSION_STAGES}")
        self.log_status(f"   Circularity Threshold: {self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD:.2f}")
        
        # Check per-well filtering status
        per_well_enabled = getattr(self.analyzer.params, 'ENABLE_PER_WELL_FILTERING', False)
        if per_well_enabled:
            well_size = getattr(self.analyzer.params, 'WELL_SIZE_ESTIMATE', 120)
            self.log_status(f"🎯 Per-Well Filtering: ENABLED (max 1 per {well_size}px well)")
            self.log_status(f"   → Picks best detection by circularity, then diameter")
        else:
            self.log_status("🎯 Per-Well Filtering: DISABLED (multiple detections per well allowed)")
        
        # Check if color detection is available
        has_color_samples = hasattr(self.analyzer, 'sample_masks') and self.analyzer.sample_masks
        if has_color_samples:
            self.log_status(f"🎨 Color Detection Enabled: {len(self.analyzer.sample_masks)} color samples")
        else:
            self.log_status("🎨 Color Detection: Using saved palette or disabled")
        
        # Prepare CSV file for batch results
        csv_path = os.path.join(output_dir, f"batch_organoid_results_{timestamp}.csv")
        
        import csv
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image_Name', 'Image_Path', 'Binary_Count', 'Color_Count', 'Total_Count', 'Processing_Time_Seconds'])
            
            # Process each image
            for i, image_path in enumerate(self.batch_images):
                start_time = datetime.now()
                
                # Update progress
                image_filename = os.path.basename(image_path)
                progress_text = f"Processing {i+1}/{len(self.batch_images)}: {image_filename}"
                self.batch_progress_label.setText(progress_text)
                self.log_status(progress_text)
                
                # Process single image
                binary_count, color_count = self.process_single_image_batch(image_path, output_dir)
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Write to CSV
                total_count = binary_count + color_count
                writer.writerow([
                    image_filename,
                    image_path,
                    binary_count,
                    color_count,
                    total_count,
                    f"{processing_time:.2f}"
                ])
                
                self.log_status(f"   ✅ {image_filename}: {binary_count} binary, {color_count} color ({total_count} total)")
        
        # Save parameters used for this batch
        params_path = os.path.join(output_dir, f"batch_parameters_{timestamp}.json")
        import json
        params_dict = {
            'BATCH_TIMESTAMP': timestamp,
            'TOTAL_IMAGES_PROCESSED': len(self.batch_images),
            'BINARY_DARK_THRESHOLD': self.analyzer.params.BINARY_DARK_THRESHOLD,
            'BINARY_INPAINT_RADIUS': self.analyzer.params.BINARY_INPAINT_RADIUS,
            'BINARY_THRESHOLD': self.analyzer.params.BINARY_THRESHOLD,
            'BINARY_MIN_DIAMETER': self.analyzer.params.BINARY_MIN_DIAMETER,
            'BINARY_MAX_DIAMETER': self.analyzer.params.BINARY_MAX_DIAMETER,
            'BINARY_CIRCULARITY_THRESHOLD': self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD,
            'BINARY_EROSION_STAGES': self.analyzer.params.BINARY_EROSION_STAGES,
            'COLOR_MIN_DIAMETER': self.analyzer.params.COLOR_MIN_DIAMETER,
            'COLOR_MAX_DIAMETER': self.analyzer.params.COLOR_MAX_DIAMETER,
            'COLOR_CIRCULARITY_THRESHOLD': self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD,
            'COLOR_EROSION_STAGES': self.analyzer.params.COLOR_EROSION_STAGES,
            'COLOR_GAUSSIAN_BLUR_SIZE': self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE,
            'COLOR_CANNY_LOW_THRESHOLD': self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD,
            'COLOR_CANNY_HIGH_THRESHOLD': self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD,
            'CENTROID_MERGE_THRESHOLD': self.analyzer.params.CENTROID_MERGE_THRESHOLD,
            'ENABLE_PER_WELL_FILTERING': getattr(self.analyzer.params, 'ENABLE_PER_WELL_FILTERING', False),
            'WELL_SIZE_ESTIMATE': getattr(self.analyzer.params, 'WELL_SIZE_ESTIMATE', 120),
            'WELL_OVERLAP_TOLERANCE': getattr(self.analyzer.params, 'WELL_OVERLAP_TOLERANCE', 0.3),
            'HAS_COLOR_SAMPLES': hasattr(self.analyzer, 'sample_masks') and bool(self.analyzer.sample_masks),
            'COLOR_SAMPLES_COUNT': len(self.analyzer.sample_masks) if hasattr(self.analyzer, 'sample_masks') and self.analyzer.sample_masks else 0
        }
        
        try:
            with open(params_path, 'w') as f:
                json.dump(params_dict, f, indent=2)
        except Exception as e:
            self.log_status(f"⚠️ Could not save parameters: {str(e)}")
        
        self.batch_progress_label.setText(f"✅ Completed! Processed {len(self.batch_images)} images")
        self.log_status(f"🎉 Batch processing completed!")
        self.log_status(f"📊 Results saved to: {csv_path}")
        self.log_status(f"📁 Output images saved to: {output_dir}")
        self.log_status(f"🔧 Parameters saved to: {params_path}")
    
    def process_single_image_batch(self, image_path, output_dir):
        """Process a single image for batch detection using exact GUI pipeline"""
        import os
        
        # Log the image being processed
        image_filename = os.path.basename(image_path)
        self.log_status(f"     📸 Loading: {image_filename}")
        
        # Load image (preserving transparency)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            self.log_status(f"❌ Could not load: {image_filename}")
            return 0, 0
        
        # Log image properties
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        channels = image.shape[2] if len(image.shape) == 3 else 1
        self.log_status(f"     📏 Image size: {image.shape[1]}x{image.shape[0]}, Channels: {channels}")
        if has_alpha:
            self.log_status(f"     🎨 Image has transparency (RGBA) - will be handled properly")
        else:
            self.log_status(f"     🖼️ Standard image (RGB/Grayscale)")
        
        # Determine processing type based on size and filename
        is_well_crop = (
            image.shape[0] < 500 or image.shape[1] < 500 or  # Small image size
            "well" in image_filename.lower() or  # Contains "well" in filename
            "crop" in image_filename.lower()     # Contains "crop" in filename
        )
        
        if is_well_crop:
            self.log_status(f"     🔬 Processing as well crop (exact GUI pipeline)")
            return self.process_well_crop_batch(image, image_path, output_dir)
        else:
            self.log_status(f"     🧬 Processing as full plate (exact GUI pipeline)")
            return self.process_full_plate_batch(image, image_path, output_dir)
    
    def process_well_crop_batch(self, image, image_path, output_dir):
        """Process a single well crop using the EXACT same pipeline as GUI demo"""
        try:
            # Create temporary analyzer that mimics the GUI state
            temp_analyzer = WellOrganoidAnalyzer(debug_mode=False)
            temp_analyzer.original_image = image
            temp_analyzer.height, temp_analyzer.width = image.shape[:2]
            temp_analyzer.image_path = image_path
            
            # Copy ALL parameters exactly from current analyzer
            for attr in dir(self.analyzer.params):
                if not attr.startswith('_'):
                    setattr(temp_analyzer.params, attr, getattr(self.analyzer.params, attr))
            
            # For well crops, adjust some parameters to be more sensitive
            temp_analyzer.params.BINARY_MIN_DIAMETER = max(3, self.analyzer.params.BINARY_MIN_DIAMETER // 2)
            temp_analyzer.params.BINARY_MAX_DIAMETER = min(100, self.analyzer.params.BINARY_MAX_DIAMETER)
            temp_analyzer.params.BINARY_EROSION_STAGES = max(1, self.analyzer.params.BINARY_EROSION_STAGES // 2)
            temp_analyzer.params.CENTROID_MERGE_THRESHOLD = max(5, self.analyzer.params.CENTROID_MERGE_THRESHOLD // 2)
            
            # REPLICATE EXACT GUI PIPELINE - Step by step
            processing_images = {}
            
            # Step 1: Convert to grayscale (exact same as GUI)
            rgb_image = self.get_rgb_for_processing(image)
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            processing_images['gray'] = gray
            
            # Step 2: Create dark mask (exact same as GUI)
            _, dark_mask = cv2.threshold(gray, temp_analyzer.params.BINARY_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            processing_images['dark_mask'] = dark_mask
            
            # Step 3: Inpaint dark areas (exact same as GUI)
            inpainted = cv2.inpaint(rgb_image, dark_mask, temp_analyzer.params.BINARY_INPAINT_RADIUS, cv2.INPAINT_TELEA)
            processing_images['inpainted'] = inpainted
            
            # Step 4: Convert inpainted to grayscale (exact same as GUI)
            inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
            processing_images['inpainted_gray'] = inpainted_gray
            
            # Step 5: Apply binary threshold (exact same as GUI)
            _, binary_plate = cv2.threshold(inpainted_gray, temp_analyzer.params.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
            processing_images['binary_plate'] = binary_plate
            
            # Step 6: Detect binary centroids with enriched data (circularity + diameter)
            if hasattr(temp_analyzer, 'find_binary_centroids_enriched'):
                # Use enriched detection with per-well filtering
                enriched_binary = temp_analyzer.find_binary_centroids_enriched(binary_plate)
                binary_centroids = temp_analyzer.filter_centroids_per_well(enriched_binary)
            else:
                # Fallback to regular detection
                binary_centroids = temp_analyzer.find_binary_centroids(binary_plate)
            processing_images['binary_centroids'] = binary_centroids
            
            self.log_status(f"     🔍 Pipeline: Gray→Dark({temp_analyzer.params.BINARY_DARK_THRESHOLD})→Inpaint({temp_analyzer.params.BINARY_INPAINT_RADIUS})→Binary({temp_analyzer.params.BINARY_THRESHOLD})→Found({len(binary_centroids)})")
            
            # Color detection using same approach as GUI if available
            color_centroids = []
            if hasattr(self.analyzer, 'sample_masks') and self.analyzer.sample_masks:
                temp_analyzer.sample_masks = self.analyzer.sample_masks
                if hasattr(temp_analyzer, 'process_color_samples_for_centroids_enriched'):
                    # Use enriched color detection if available
                    enriched_color = temp_analyzer.process_color_samples_for_centroids_enriched()
                    color_centroids = temp_analyzer.filter_centroids_per_well(enriched_color)
                else:
                    color_centroids = temp_analyzer.process_color_samples_for_centroids()
                
            # Merge centroids using same method as GUI (after per-well filtering)
            if binary_centroids:
                binary_centroids = temp_analyzer.merge_close_centroids(binary_centroids, temp_analyzer.params.CENTROID_MERGE_THRESHOLD)
            if color_centroids:
                color_centroids = temp_analyzer.merge_close_centroids(color_centroids, temp_analyzer.params.CENTROID_MERGE_THRESHOLD)
            
            # Create result image EXACTLY like GUI demo
            result_rgb = self.get_rgb_for_processing(image).copy()
            
            # Draw centroids exactly like GUI
            for cx, cy in binary_centroids:
                cv2.circle(result_rgb, (cx, cy), 6, (255, 0, 0), 2)  # Blue circles
                cv2.circle(result_rgb, (cx, cy), 1, (255, 0, 0), -1)
            
            for cx, cy in color_centroids:
                cv2.circle(result_rgb, (cx, cy), 6, (0, 0, 255), 2)  # Red circles
                cv2.circle(result_rgb, (cx, cy), 1, (0, 0, 255), -1)
            
            # Preserve alpha channel exactly like GUI
            result_image = self.preserve_alpha_in_result(result_rgb, image)
            
            # Save result image
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            result_path = os.path.join(output_dir, f"{image_name}_detected.png")
            
            counter = 1
            while os.path.exists(result_path):
                result_path = os.path.join(output_dir, f"{image_name}_detected_{counter}.png")
                counter += 1
            
            # Save with transparency support
            if len(result_image.shape) == 3 and result_image.shape[2] == 4:
                if not result_path.lower().endswith('.png'):
                    result_path = os.path.splitext(result_path)[0] + '.png'
            cv2.imwrite(result_path, result_image)
            
            self.log_status(f"     ✅ Well crop: {len(binary_centroids)} binary, {len(color_centroids)} color → {os.path.basename(result_path)}")
            return len(binary_centroids), len(color_centroids)
            
        except Exception as e:
            import traceback
            self.log_status(f"❌ Error processing well crop {os.path.basename(image_path)}: {str(e)}")
            self.log_status(f"     Stack trace: {traceback.format_exc()}")
            return 0, 0
    
    def process_full_plate_batch(self, image, image_path, output_dir):
        """Process a full plate image using the EXACT same pipeline as GUI demo"""
        try:
            # Create temporary analyzer that mimics the GUI state
            temp_analyzer = WellOrganoidAnalyzer(debug_mode=False)
            temp_analyzer.original_image = image
            temp_analyzer.height, temp_analyzer.width = image.shape[:2]
            temp_analyzer.image_path = image_path
            
            # Copy ALL parameters exactly from current analyzer
            for attr in dir(self.analyzer.params):
                if not attr.startswith('_'):
                    setattr(temp_analyzer.params, attr, getattr(self.analyzer.params, attr))
            
            # REPLICATE EXACT GUI PIPELINE - Step by step (no shortcuts!)
            processing_images = {}
            
            # Step 1: Convert to grayscale (exact same as GUI step1_grayscale)
            rgb_image = self.get_rgb_for_processing(image)
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            processing_images['gray'] = gray
            
            # Step 2: Create dark mask (exact same as GUI step2_dark_mask)
            _, dark_mask = cv2.threshold(gray, temp_analyzer.params.BINARY_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            processing_images['dark_mask'] = dark_mask
            
            # Step 3: Inpaint dark areas (exact same as GUI step3_inpainting)
            inpainted = cv2.inpaint(rgb_image, dark_mask, temp_analyzer.params.BINARY_INPAINT_RADIUS, cv2.INPAINT_TELEA)
            processing_images['inpainted'] = inpainted
            
            # Step 4: Convert inpainted to grayscale (exact same as GUI step4_inpaint_gray)
            inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
            processing_images['inpainted_gray'] = inpainted_gray
            
            # Step 5: Apply binary threshold (exact same as GUI step5_binary)
            _, binary_plate = cv2.threshold(inpainted_gray, temp_analyzer.params.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
            processing_images['binary_plate'] = binary_plate
            
            # Step 6: Detect binary centroids (exact same as GUI step6_detect_binary)
            # Step 6: Detect binary centroids with enriched data (circularity + diameter)
            if hasattr(temp_analyzer, 'find_binary_centroids_enriched'):
                # Use enriched detection with per-well filtering
                enriched_binary = temp_analyzer.find_binary_centroids_enriched(binary_plate)
                binary_centroids = temp_analyzer.filter_centroids_per_well(enriched_binary)
            else:
                # Fallback to regular detection
                binary_centroids = temp_analyzer.find_binary_centroids(binary_plate)
            processing_images['binary_centroids'] = binary_centroids
            
            self.log_status(f"     🔍 Full Pipeline: Gray→Dark({temp_analyzer.params.BINARY_DARK_THRESHOLD})→Inpaint({temp_analyzer.params.BINARY_INPAINT_RADIUS})→Binary({temp_analyzer.params.BINARY_THRESHOLD})→Found({len(binary_centroids)})")
            
            # Color detection using same approach as GUI if available
            color_centroids = []
            if hasattr(self.analyzer, 'sample_masks') and self.analyzer.sample_masks:
                # Use the current analyzer's sample masks for color detection (same as GUI)
                temp_analyzer.sample_masks = self.analyzer.sample_masks
                if hasattr(temp_analyzer, 'process_color_samples_for_centroids_enriched'):
                    # Use enriched color detection if available
                    enriched_color = temp_analyzer.process_color_samples_for_centroids_enriched()
                    color_centroids = temp_analyzer.filter_centroids_per_well(enriched_color)
                else:
                    color_centroids = temp_analyzer.process_color_samples_for_centroids()
                self.log_status(f"     🎨 Color detection: Using {len(self.analyzer.sample_masks)} GUI samples → {len(color_centroids)} centroids")
            elif os.path.exists(temp_analyzer.params.COLOR_PALETTE_FILENAME):
                # Try to load saved color palette as fallback
                sample_pixels = temp_analyzer.load_color_palette()
                if sample_pixels:
                    color_centroids = temp_analyzer.process_saved_palette_for_centroids(sample_pixels)
                    self.log_status(f"     🎨 Color detection: Using saved palette → {len(color_centroids)} centroids")
            
            # Merge close centroids using same method as GUI (after per-well filtering)
            if binary_centroids:
                binary_centroids = temp_analyzer.merge_close_centroids(binary_centroids, temp_analyzer.params.CENTROID_MERGE_THRESHOLD)
            if color_centroids:
                color_centroids = temp_analyzer.merge_close_centroids(color_centroids, temp_analyzer.params.CENTROID_MERGE_THRESHOLD)
            
            # Create result image EXACTLY like GUI demo
            result_rgb = self.get_rgb_for_processing(image).copy()
            
            # Draw centroids exactly like GUI (same circle sizes and colors)
            for cx, cy in binary_centroids:
                cv2.circle(result_rgb, (cx, cy), 8, (255, 0, 0), 2)  # Blue circles
                cv2.circle(result_rgb, (cx, cy), 2, (255, 0, 0), -1)
            
            for cx, cy in color_centroids:
                cv2.circle(result_rgb, (cx, cy), 8, (0, 0, 255), 2)  # Red circles
                cv2.circle(result_rgb, (cx, cy), 2, (0, 0, 255), -1)
            
            # Preserve alpha channel exactly like GUI
            result_image = self.preserve_alpha_in_result(result_rgb, image)
            
            # Save result image
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            result_path = os.path.join(output_dir, f"{image_name}_detected.png")
            
            counter = 1
            while os.path.exists(result_path):
                result_path = os.path.join(output_dir, f"{image_name}_detected_{counter}.png")
                counter += 1
            
            # Save with transparency support
            if len(result_image.shape) == 3 and result_image.shape[2] == 4:
                if not result_path.lower().endswith('.png'):
                    result_path = os.path.splitext(result_path)[0] + '.png'
            cv2.imwrite(result_path, result_image)
            
            self.log_status(f"     ✅ Full plate: {len(binary_centroids)} binary, {len(color_centroids)} color → {os.path.basename(result_path)}")
            return len(binary_centroids), len(color_centroids)
            
        except Exception as e:
            import traceback
            self.log_status(f"❌ Error processing full plate {os.path.basename(image_path)}: {str(e)}")
            self.log_status(f"     Stack trace: {traceback.format_exc()}")
            return 0, 0

    def save_parameters(self):
        """Save current parameters to JSON file"""
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Parameters",
            "organoid_detection_parameters.json",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            params_dict = {
                'BINARY_DARK_THRESHOLD': self.analyzer.params.BINARY_DARK_THRESHOLD,
                'BINARY_INPAINT_RADIUS': self.analyzer.params.BINARY_INPAINT_RADIUS,
                'BINARY_THRESHOLD': self.analyzer.params.BINARY_THRESHOLD,
                'BINARY_MIN_DIAMETER': self.analyzer.params.BINARY_MIN_DIAMETER,
                'BINARY_MAX_DIAMETER': self.analyzer.params.BINARY_MAX_DIAMETER,
                'BINARY_CIRCULARITY_THRESHOLD': self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD,
                'BINARY_EROSION_STAGES': self.analyzer.params.BINARY_EROSION_STAGES,
                'COLOR_MIN_DIAMETER': self.analyzer.params.COLOR_MIN_DIAMETER,
                'COLOR_MAX_DIAMETER': self.analyzer.params.COLOR_MAX_DIAMETER,
                'COLOR_CIRCULARITY_THRESHOLD': self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD,
                'COLOR_EROSION_STAGES': self.analyzer.params.COLOR_EROSION_STAGES,
                'COLOR_GAUSSIAN_BLUR_SIZE': self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE,
                'COLOR_CANNY_LOW_THRESHOLD': self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD,
                'COLOR_CANNY_HIGH_THRESHOLD': self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD,
                'CENTROID_MERGE_THRESHOLD': self.analyzer.params.CENTROID_MERGE_THRESHOLD,
                'KERNEL_SIZE': self.kernel_size,
                'DILATION_ITERATIONS': self.dilation_iterations
            }
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(params_dict, f, indent=2)
                self.log_status(f"✅ Parameters saved to: {file_path}")
            except Exception as e:
                self.log_status(f"❌ Error saving parameters: {str(e)}")
    
    def load_parameters(self):
        """Load parameters from JSON file"""
        from PyQt6.QtWidgets import QFileDialog
        import json
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Parameters",
            "",
            "JSON files (*.json);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    params_dict = json.load(f)
                
                # Update parameters
                if 'BINARY_DARK_THRESHOLD' in params_dict:
                    self.analyzer.params.BINARY_DARK_THRESHOLD = params_dict['BINARY_DARK_THRESHOLD']
                    self.dark_slider.setValue(params_dict['BINARY_DARK_THRESHOLD'])
                
                if 'BINARY_INPAINT_RADIUS' in params_dict:
                    self.analyzer.params.BINARY_INPAINT_RADIUS = params_dict['BINARY_INPAINT_RADIUS']
                    self.inpaint_slider.setValue(params_dict['BINARY_INPAINT_RADIUS'])
                
                if 'BINARY_THRESHOLD' in params_dict:
                    self.analyzer.params.BINARY_THRESHOLD = params_dict['BINARY_THRESHOLD']
                    self.binary_slider.setValue(params_dict['BINARY_THRESHOLD'])
                
                if 'BINARY_MIN_DIAMETER' in params_dict:
                    self.analyzer.params.BINARY_MIN_DIAMETER = params_dict['BINARY_MIN_DIAMETER']
                    self.min_diameter_slider.setValue(params_dict['BINARY_MIN_DIAMETER'])
                
                if 'BINARY_MAX_DIAMETER' in params_dict:
                    self.analyzer.params.BINARY_MAX_DIAMETER = params_dict['BINARY_MAX_DIAMETER']
                    self.max_diameter_slider.setValue(params_dict['BINARY_MAX_DIAMETER'])
                
                if 'BINARY_CIRCULARITY_THRESHOLD' in params_dict:
                    self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD = params_dict['BINARY_CIRCULARITY_THRESHOLD']
                    self.circularity_slider.setValue(int(params_dict['BINARY_CIRCULARITY_THRESHOLD'] * 100))
                
                if 'BINARY_EROSION_STAGES' in params_dict:
                    self.analyzer.params.BINARY_EROSION_STAGES = params_dict['BINARY_EROSION_STAGES']
                    self.erosion_slider.setValue(params_dict['BINARY_EROSION_STAGES'])
                
                if 'COLOR_MIN_DIAMETER' in params_dict:
                    self.analyzer.params.COLOR_MIN_DIAMETER = params_dict['COLOR_MIN_DIAMETER']
                    self.color_min_diameter_slider.setValue(params_dict['COLOR_MIN_DIAMETER'])
                
                if 'COLOR_MAX_DIAMETER' in params_dict:
                    self.analyzer.params.COLOR_MAX_DIAMETER = params_dict['COLOR_MAX_DIAMETER']
                    self.color_max_diameter_slider.setValue(params_dict['COLOR_MAX_DIAMETER'])
                
                if 'COLOR_CIRCULARITY_THRESHOLD' in params_dict:
                    self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD = params_dict['COLOR_CIRCULARITY_THRESHOLD']
                    self.color_circularity_slider.setValue(int(params_dict['COLOR_CIRCULARITY_THRESHOLD'] * 100))
                
                if 'COLOR_EROSION_STAGES' in params_dict:
                    self.analyzer.params.COLOR_EROSION_STAGES = params_dict['COLOR_EROSION_STAGES']
                    self.color_erosion_slider.setValue(params_dict['COLOR_EROSION_STAGES'])
                
                if 'COLOR_GAUSSIAN_BLUR_SIZE' in params_dict:
                    self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE = params_dict['COLOR_GAUSSIAN_BLUR_SIZE']
                    self.blur_slider.setValue(params_dict['COLOR_GAUSSIAN_BLUR_SIZE'])
                
                if 'COLOR_CANNY_LOW_THRESHOLD' in params_dict:
                    self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD = params_dict['COLOR_CANNY_LOW_THRESHOLD']
                    self.canny_low_slider.setValue(params_dict['COLOR_CANNY_LOW_THRESHOLD'])
                
                if 'COLOR_CANNY_HIGH_THRESHOLD' in params_dict:
                    self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD = params_dict['COLOR_CANNY_HIGH_THRESHOLD']
                    self.canny_high_slider.setValue(params_dict['COLOR_CANNY_HIGH_THRESHOLD'])
                
                if 'CENTROID_MERGE_THRESHOLD' in params_dict:
                    self.analyzer.params.CENTROID_MERGE_THRESHOLD = params_dict['CENTROID_MERGE_THRESHOLD']
                    self.merge_threshold_slider.setValue(params_dict['CENTROID_MERGE_THRESHOLD'])
                
                if 'KERNEL_SIZE' in params_dict:
                    self.kernel_size = params_dict['KERNEL_SIZE']
                    self.kernel_size_slider.setValue(params_dict['KERNEL_SIZE'])
                
                if 'DILATION_ITERATIONS' in params_dict:
                    self.dilation_iterations = params_dict['DILATION_ITERATIONS']
                    self.dilation_slider.setValue(params_dict['DILATION_ITERATIONS'])
                
                self.log_status(f"✅ Parameters loaded from: {file_path}")
                
            except Exception as e:
                self.log_status(f"❌ Error loading parameters: {str(e)}")

    def handle_transparency(self, image):
        """Handle transparency in images - convert RGBA to RGB with white background if needed"""
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Image has alpha channel
            alpha = image[:, :, 3] / 255.0
            rgb_channels = image[:, :, :3]
            
            # Don't composite with white - preserve transparency by keeping the RGBA format
            # Only convert to RGB when specifically needed for processing that doesn't support alpha
            return image  # Return original RGBA image
        else:
            # Image is already RGB or grayscale
            return image
    
    def get_rgb_for_processing(self, image):
        """Get RGB version of image for processing that doesn't support alpha channels"""
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Image has alpha channel - extract RGB only and ensure contiguous array
            return np.ascontiguousarray(image[:, :, :3])
        else:
            # Image is already RGB or grayscale - ensure contiguous
            return np.ascontiguousarray(image)

    def apply_alpha_mask_simple(self, processed_image):
        """SIMPLE: Apply alpha mask to restrict processing to non-transparent areas only"""
        if self.alpha_mask is not None:
            # Set transparent pixels to 0 (or background value)
            if len(processed_image.shape) == 2:
                # Grayscale image
                processed_image[~self.alpha_mask] = 0
            elif len(processed_image.shape) == 3:
                # Color image
                processed_image[~self.alpha_mask] = 0
        return processed_image
    
    def preserve_alpha_in_result(self, result_rgb, original_rgba):
        """Preserve alpha channel in result image if original had transparency"""
        if len(original_rgba.shape) == 3 and original_rgba.shape[2] == 4:
            # Original has alpha, preserve it in result
            result_rgba = np.zeros((result_rgb.shape[0], result_rgb.shape[1], 4), dtype=np.uint8)
            result_rgba[:, :, :3] = result_rgb
            result_rgba[:, :, 3] = original_rgba[:, :, 3]  # Copy alpha channel
            return result_rgba
        else:
            # Original was RGB, return RGB result
            return result_rgb

    def get_transparency_mask(self, image):
        """Get mask of non-transparent pixels (where alpha > 0)"""
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Image has alpha channel - create mask where alpha > 0
            alpha_channel = image[:, :, 3]
            return alpha_channel > 0  # True for non-transparent pixels
        else:
            # No alpha channel - all pixels are "non-transparent"
            return np.ones((image.shape[0], image.shape[1]), dtype=bool)
    
    def apply_transparency_mask(self, processed_image, original_image, preserve_alpha=True):
        """Apply transparency mask to processed image - make transparent pixels transparent again"""
        if len(original_image.shape) == 3 and original_image.shape[2] == 4:
            # Original has alpha channel
            alpha_mask = self.get_transparency_mask(original_image)
            
            if len(processed_image.shape) == 2:
                # Grayscale processed image - set transparent pixels to 0
                processed_image[~alpha_mask] = 0
            elif len(processed_image.shape) == 3:
                # Color processed image - set transparent pixels to 0
                processed_image[~alpha_mask] = 0
            
            if preserve_alpha:
                # If we want to preserve alpha channel, add it back
                if len(processed_image.shape) == 2:
                    # Convert grayscale to RGBA
                    result = np.zeros((processed_image.shape[0], processed_image.shape[1], 4), dtype=np.uint8)
                    result[:, :, 0] = processed_image
                    result[:, :, 1] = processed_image
                    result[:, :, 2] = processed_image
                    result[:, :, 3] = original_image[:, :, 3]
                    return result
                elif len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                    # Convert RGB to RGBA
                    result = np.zeros((processed_image.shape[0], processed_image.shape[1], 4), dtype=np.uint8)
                    result[:, :, :3] = processed_image
                    result[:, :, 3] = original_image[:, :, 3]
                    return result
                else:
                    # Already RGBA
                    return processed_image
            else:
                return processed_image
        else:
            # No transparency to handle
            return processed_image

# Also add the dummy class for when PyQt6 is not available
if not PYQT_AVAILABLE:
    class ParameterDebugGUI:
        def __init__(self, analyzer):
            print("PyQt6 not available - debug GUI disabled")
            self.analyzer = analyzer
            
        def show(self):
            print("GUI not available - running automated analysis")
            self.analyzer.run_complete_analysis()
            
        def close(self):
            pass 