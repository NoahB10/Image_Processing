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
import json
import os

# PyQt6 imports with availability checking
PYQT_AVAILABLE = True
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                                 QWidget, QPushButton, QLabel, QSlider, QGroupBox,
                                 QScrollArea, QTextEdit, QGridLayout, QSpinBox, 
                                 QDoubleSpinBox, QTabWidget)
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
        
        # Morphological operation parameters
        self.kernel_size = 3
        self.dilation_iterations = 0
        
        # Color detection workflow
        self.color_current_step = 0
        self.color_processing_images = {}
        
        # Parameter file for saving/loading
        self.param_file = "debug_parameters.json"
        
        self.init_ui()
        self.load_parameters()
        
        # Don't load images automatically - wait for user selection
        # self.load_images()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Organoid Analysis - Debug Interface")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_panel.setFixedWidth(450)
        left_layout = QVBoxLayout(left_panel)
        
        # File upload section at the top
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # Image file selection
        image_file_layout = QHBoxLayout()
        self.image_file_btn = QPushButton("Select Image File")
        self.image_file_btn.clicked.connect(self.select_image_file)
        self.image_file_label = QLabel("No image selected")
        self.image_file_label.setWordWrap(True)
        
        image_file_layout.addWidget(self.image_file_btn)
        image_file_layout.addWidget(self.image_file_label, 1)
        file_layout.addLayout(image_file_layout)
        
        # Mask file selection
        mask_file_layout = QHBoxLayout()
        self.mask_file_btn = QPushButton("Select Mask File")
        self.mask_file_btn.clicked.connect(self.select_mask_file)
        self.mask_file_label = QLabel("No mask selected")
        self.mask_file_label.setWordWrap(True)
        
        mask_file_layout.addWidget(self.mask_file_btn)
        mask_file_layout.addWidget(self.mask_file_label, 1)
        file_layout.addLayout(mask_file_layout)
        
        # Start analysis button
        self.start_analysis_btn = QPushButton("Start Analysis")
        self.start_analysis_btn.clicked.connect(self.start_analysis)
        self.start_analysis_btn.setEnabled(False)
        file_layout.addWidget(self.start_analysis_btn)
        
        left_layout.addWidget(file_group)
        
        # Create tabbed interface
        self.tab_widget = QTabWidget()
        self.tab_widget.setEnabled(False)  # Disabled until analysis starts
        
        # Binary detection tab
        binary_tab = QWidget()
        binary_layout = QVBoxLayout(binary_tab)
        
        # Binary step buttons
        binary_step_group = QGroupBox("Binary Detection Steps")
        binary_step_layout = QVBoxLayout(binary_step_group)
        
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
        
        binary_step_layout.addWidget(self.step1_btn)
        binary_step_layout.addWidget(self.step2_btn)
        binary_step_layout.addWidget(self.step3_btn)
        binary_step_layout.addWidget(self.step4_btn)
        binary_step_layout.addWidget(self.step5_btn)
        binary_step_layout.addWidget(self.step6_btn)
        
        binary_layout.addWidget(binary_step_group)
        
        # Binary parameters
        self.create_binary_parameters_panel(binary_layout)
        
        # Binary erosion visualization
        binary_erosion_group = QGroupBox("Erosion Stage Visualization")
        binary_erosion_layout = QVBoxLayout(binary_erosion_group)
        
        self.erosion_btn = QPushButton("Show Erosion Stages")
        self.erosion_btn.clicked.connect(self.show_erosion_stages)
        self.erosion_btn.setEnabled(False)
        
        binary_erosion_layout.addWidget(self.erosion_btn)
        binary_layout.addWidget(binary_erosion_group)
        
        # Skip binary detection
        self.skip_binary_btn = QPushButton("Skip Binary Detection")
        self.skip_binary_btn.clicked.connect(self.skip_binary)
        binary_layout.addWidget(self.skip_binary_btn)
        
        self.tab_widget.addTab(binary_tab, "Binary Detection")
        
        # Color detection tab
        color_tab = QWidget()
        color_layout = QVBoxLayout(color_tab)
        
        # Color step buttons
        color_step_group = QGroupBox("Color Detection Steps")
        color_step_layout = QVBoxLayout(color_step_group)
        
        self.color_step1_btn = QPushButton("Step 1: Load Original Image")
        self.color_step2_btn = QPushButton("Step 2: Sample Colors")
        self.color_step3_btn = QPushButton("Step 3: Create Color Mask")
        self.color_step4_btn = QPushButton("Step 4: Apply Gaussian Blur")
        self.color_step5_btn = QPushButton("Step 5: Canny Edge Detection")
        self.color_step6_btn = QPushButton("Step 6: Apply Erosion")
        self.color_step7_btn = QPushButton("Step 7: Detect Color Centroids")
        
        self.color_step1_btn.clicked.connect(self.color_step1_load_image)
        self.color_step2_btn.clicked.connect(self.color_step2_sample_colors)
        self.color_step3_btn.clicked.connect(self.color_step3_create_mask)
        self.color_step4_btn.clicked.connect(self.color_step4_blur)
        self.color_step5_btn.clicked.connect(self.color_step5_canny)
        self.color_step6_btn.clicked.connect(self.color_step6_erosion)
        self.color_step7_btn.clicked.connect(self.color_step7_detect)
        
        color_step_layout.addWidget(self.color_step1_btn)
        color_step_layout.addWidget(self.color_step2_btn)
        color_step_layout.addWidget(self.color_step3_btn)
        color_step_layout.addWidget(self.color_step4_btn)
        color_step_layout.addWidget(self.color_step5_btn)
        color_step_layout.addWidget(self.color_step6_btn)
        color_step_layout.addWidget(self.color_step7_btn)
        
        color_layout.addWidget(color_step_group)
        
        # Color parameters
        self.create_color_parameters_panel(color_layout)
        
        # Color erosion visualization
        color_erosion_group = QGroupBox("Color Erosion Visualization")
        color_erosion_layout = QVBoxLayout(color_erosion_group)
        
        self.color_erosion_viz_btn = QPushButton("Show Color Erosion Stages")
        self.color_erosion_viz_btn.clicked.connect(self.show_color_erosion_stages)
        self.color_erosion_viz_btn.setEnabled(False)
        
        color_erosion_layout.addWidget(self.color_erosion_viz_btn)
        color_layout.addWidget(color_erosion_group)
        
        # Skip color detection
        self.skip_color_btn = QPushButton("Skip Color Detection")
        self.skip_color_btn.clicked.connect(self.skip_color)
        color_layout.addWidget(self.skip_color_btn)
        
        self.tab_widget.addTab(color_tab, "Color Detection")
        
        left_layout.addWidget(self.tab_widget)
        
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
        self.image_label.setMinimumSize(1000, 700)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_label)
        
    def create_binary_parameters_panel(self, layout):
        """Create binary detection parameter controls"""
        params_group = QGroupBox("Binary Parameters")
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
        
        # Size parameters
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
        
        # Circularity
        circ_group = QGroupBox("Circularity")
        circ_layout = QGridLayout(circ_group)
        
        self.circularity_slider = QSlider(Qt.Orientation.Horizontal)
        self.circularity_slider.setRange(0, 100)
        self.circularity_slider.setValue(int(self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD * 100))
        self.circularity_slider.valueChanged.connect(self.update_circularity)
        
        self.circularity_label = QLabel(f"{self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD:.2f}")
        
        circ_layout.addWidget(QLabel("Value:"), 0, 0)
        circ_layout.addWidget(self.circularity_slider, 0, 1)
        circ_layout.addWidget(self.circularity_label, 0, 2)
        
        params_layout.addWidget(circ_group)
        
        # Morphological operations
        morph_group = QGroupBox("Morphological Operations")
        morph_layout = QGridLayout(morph_group)
        
        # Erosion stages
        self.erosion_slider = QSlider(Qt.Orientation.Horizontal)
        self.erosion_slider.setRange(0, 10)
        self.erosion_slider.setValue(self.analyzer.params.BINARY_EROSION_STAGES)
        self.erosion_slider.valueChanged.connect(self.update_erosion_stages)
        
        self.erosion_label = QLabel(str(self.analyzer.params.BINARY_EROSION_STAGES))
        
        morph_layout.addWidget(QLabel("Erosion:"), 0, 0)
        morph_layout.addWidget(self.erosion_slider, 0, 1)
        morph_layout.addWidget(self.erosion_label, 0, 2)
        
        # Kernel size
        self.kernel_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.kernel_size_slider.setRange(1, 15)
        self.kernel_size_slider.setValue(self.kernel_size)
        self.kernel_size_slider.valueChanged.connect(self.update_kernel_size)
        
        self.kernel_size_label = QLabel(str(self.kernel_size))
        
        morph_layout.addWidget(QLabel("Kernel Size:"), 1, 0)
        morph_layout.addWidget(self.kernel_size_slider, 1, 1)
        morph_layout.addWidget(self.kernel_size_label, 1, 2)
        
        # Dilation iterations
        self.dilation_slider = QSlider(Qt.Orientation.Horizontal)
        self.dilation_slider.setRange(0, 10)
        self.dilation_slider.setValue(self.dilation_iterations)
        self.dilation_slider.valueChanged.connect(self.update_dilation_iterations)
        
        self.dilation_label = QLabel(str(self.dilation_iterations))
        
        morph_layout.addWidget(QLabel("Dilation:"), 2, 0)
        morph_layout.addWidget(self.dilation_slider, 2, 1)
        morph_layout.addWidget(self.dilation_label, 2, 2)
        
        params_layout.addWidget(morph_group)
        
        scroll_widget.setLayout(params_layout)
        scroll_area.setWidget(scroll_widget)
        params_main_layout.addWidget(scroll_area)
        
        layout.addWidget(params_group)

    def create_color_parameters_panel(self, layout):
        """Create color detection parameter controls"""
        params_group = QGroupBox("Color Parameters")
        params_main_layout = QVBoxLayout(params_group)
        
        # Create scroll area for parameters
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        params_layout = QVBoxLayout(scroll_widget)
        
        # Size parameters
        size_group = QGroupBox("Size Parameters")
        size_layout = QGridLayout(size_group)
        
        # Color Min Diameter
        self.color_min_diameter_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_min_diameter_slider.setRange(1, 150)
        self.color_min_diameter_slider.setValue(self.analyzer.params.COLOR_MIN_DIAMETER)
        self.color_min_diameter_slider.valueChanged.connect(self.update_color_min_diameter)
        
        self.color_min_diameter_label = QLabel(str(self.analyzer.params.COLOR_MIN_DIAMETER))
        
        size_layout.addWidget(QLabel("Min Diameter:"), 0, 0)
        size_layout.addWidget(self.color_min_diameter_slider, 0, 1)
        size_layout.addWidget(self.color_min_diameter_label, 0, 2)
        
        # Color Max Diameter
        self.color_max_diameter_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_max_diameter_slider.setRange(20, 300)
        self.color_max_diameter_slider.setValue(self.analyzer.params.COLOR_MAX_DIAMETER)
        self.color_max_diameter_slider.valueChanged.connect(self.update_color_max_diameter)
        
        self.color_max_diameter_label = QLabel(str(self.analyzer.params.COLOR_MAX_DIAMETER))
        
        size_layout.addWidget(QLabel("Max Diameter:"), 1, 0)
        size_layout.addWidget(self.color_max_diameter_slider, 1, 1)
        size_layout.addWidget(self.color_max_diameter_label, 1, 2)
        
        params_layout.addWidget(size_group)
        
        # Shape parameters
        shape_group = QGroupBox("Shape Parameters")
        shape_layout = QGridLayout(shape_group)
        
        # Color Circularity
        self.color_circularity_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_circularity_slider.setRange(0, 100)
        self.color_circularity_slider.setValue(int(self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD * 100))
        self.color_circularity_slider.valueChanged.connect(self.update_color_circularity)
        
        self.color_circularity_label = QLabel(f"{self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD:.2f}")
        
        shape_layout.addWidget(QLabel("Circularity:"), 0, 0)
        shape_layout.addWidget(self.color_circularity_slider, 0, 1)
        shape_layout.addWidget(self.color_circularity_label, 0, 2)
        
        params_layout.addWidget(shape_group)
        
        # Processing parameters
        proc_group = QGroupBox("Processing Parameters")
        proc_layout = QGridLayout(proc_group)
        
        # Gaussian Blur
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setRange(1, 15)
        self.blur_slider.setValue(self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE)
        self.blur_slider.valueChanged.connect(self.update_blur_size)
        
        self.blur_label = QLabel(str(self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE))
        
        proc_layout.addWidget(QLabel("Gaussian Blur:"), 0, 0)
        proc_layout.addWidget(self.blur_slider, 0, 1)
        proc_layout.addWidget(self.blur_label, 0, 2)
        
        # Canny Low Threshold
        self.canny_low_slider = QSlider(Qt.Orientation.Horizontal)
        self.canny_low_slider.setRange(1, 200)
        self.canny_low_slider.setValue(self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD)
        self.canny_low_slider.valueChanged.connect(self.update_canny_low)
        
        self.canny_low_label = QLabel(str(self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD))
        
        proc_layout.addWidget(QLabel("Canny Low:"), 1, 0)
        proc_layout.addWidget(self.canny_low_slider, 1, 1)
        proc_layout.addWidget(self.canny_low_label, 1, 2)
        
        # Canny High Threshold
        self.canny_high_slider = QSlider(Qt.Orientation.Horizontal)
        self.canny_high_slider.setRange(10, 300)
        self.canny_high_slider.setValue(self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD)
        self.canny_high_slider.valueChanged.connect(self.update_canny_high)
        
        self.canny_high_label = QLabel(str(self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD))
        
        proc_layout.addWidget(QLabel("Canny High:"), 2, 0)
        proc_layout.addWidget(self.canny_high_slider, 2, 1)
        proc_layout.addWidget(self.canny_high_label, 2, 2)
        
        # Color Erosion Stages
        self.color_erosion_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_erosion_slider.setRange(0, 15)
        self.color_erosion_slider.setValue(self.analyzer.params.COLOR_EROSION_STAGES)
        self.color_erosion_slider.valueChanged.connect(self.update_color_erosion)
        
        self.color_erosion_label = QLabel(str(self.analyzer.params.COLOR_EROSION_STAGES))
        
        proc_layout.addWidget(QLabel("Erosion:"), 3, 0)
        proc_layout.addWidget(self.color_erosion_slider, 3, 1)
        proc_layout.addWidget(self.color_erosion_label, 3, 2)
        
        params_layout.addWidget(proc_group)
        
        scroll_widget.setLayout(params_layout)
        scroll_area.setWidget(scroll_widget)
        params_main_layout.addWidget(scroll_area)
        
        layout.addWidget(params_group)
        
    def select_image_file(self):
        """Select image file for analysis"""
        try:
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()
            
            file_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
            )
            
            root.destroy()
            
            if file_path:
                # Load and validate the image
                test_image = cv2.imread(file_path)
                if test_image is not None:
                    self.analyzer.image_path = file_path
                    self.analyzer.original_image = test_image
                    self.analyzer.height, self.analyzer.width = test_image.shape[:2]
                    
                    # Update UI
                    filename = file_path.split('/')[-1]  # Get just the filename
                    self.image_file_label.setText(f"✅ {filename}")
                    
                    # Display preview image
                    self.display_image(test_image)
                    
                    # Check if we can enable start button
                    self.check_start_ready()
                    
                    self.log_status(f"✅ Image loaded: {self.analyzer.width}x{self.analyzer.height}")
                else:
                    self.log_status("❌ Could not load selected image file")
            else:
                self.log_status("No image file selected")
                
        except Exception as e:
            self.log_status(f"❌ Error selecting image: {e}")
            
    def select_mask_file(self):
        """Select mask file for analysis"""
        try:
            from tkinter import filedialog, Tk
            root = Tk()
            root.withdraw()
            
            file_path = filedialog.askopenfilename(
                title="Select Mask File",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*")]
            )
            
            root.destroy()
            
            if file_path:
                # Load and validate the mask
                test_mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if test_mask is not None:
                    self.analyzer.mask_path = file_path
                    
                    # Update UI
                    filename = file_path.split('/')[-1]  # Get just the filename
                    self.mask_file_label.setText(f"✅ {filename}")
                    
                    # Check if we can enable start button
                    self.check_start_ready()
                    
                    self.log_status(f"✅ Mask loaded: {test_mask.shape[1]}x{test_mask.shape[0]}")
                else:
                    self.log_status("❌ Could not load selected mask file")
            else:
                # Allow skipping mask selection
                self.analyzer.mask_path = None
                self.mask_file_label.setText("⚠️ No mask (optional)")
                self.check_start_ready()
                self.log_status("No mask file selected (optional)")
                
        except Exception as e:
            self.log_status(f"❌ Error selecting mask: {e}")
            
    def check_start_ready(self):
        """Check if analysis can be started"""
        image_ready = hasattr(self.analyzer, 'original_image') and self.analyzer.original_image is not None
        
        if image_ready:
            self.start_analysis_btn.setEnabled(True)
            self.log_status("🎯 Ready to start analysis!")
        else:
            self.start_analysis_btn.setEnabled(False)
            
    def start_analysis(self):
        """Start the analysis with selected files"""
        if not hasattr(self.analyzer, 'original_image') or self.analyzer.original_image is None:
            self.log_status("❌ No image loaded")
            return
            
        # Enable the tabbed interface
        self.tab_widget.setEnabled(True)
        
        # Disable file selection
        self.image_file_btn.setEnabled(False)
        self.mask_file_btn.setEnabled(False)
        self.start_analysis_btn.setEnabled(False)
        
        # Set zoom center
        self.analyzer.zoom_center_x = self.analyzer.width // 2
        self.analyzer.zoom_center_y = self.analyzer.height // 2
        
        self.log_status("🚀 Analysis started! Use the tabs to navigate through detection steps.")
        self.log_status("📋 Binary Detection: Step through the binary detection pipeline")
        self.log_status("🎨 Color Detection: Complete color sampling and detection")
        
        # Save current parameters immediately
        self.save_parameters()
        
    def load_images(self):
        """Load and display the original image (kept for compatibility)"""
        if self.analyzer.original_image is not None:
            self.display_image(self.analyzer.original_image)
            self.log_status("✅ Images loaded successfully")
            self.log_status(f"Image size: {self.analyzer.width} x {self.analyzer.height}")
        else:
            self.log_status("❌ No image loaded")
    
    def step1_grayscale(self):
        """Step 1: Convert to grayscale"""
        if self.analyzer.original_image is None:
            self.log_status("❌ No image loaded")
            return
            
        gray = cv2.cvtColor(self.analyzer.original_image, cv2.COLOR_BGR2GRAY)
        self.processing_images['gray'] = gray
        
        # Convert to 3-channel for display
        gray_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.display_image(gray_display)
        
        self.current_step = 1
        self.log_status("✅ Step 1: Converted to grayscale")
        
    def step2_dark_mask(self):
        """Step 2: Create dark mask"""
        if 'gray' not in self.processing_images:
            self.log_status("❌ Run Step 1 first")
            return
            
        gray = self.processing_images['gray']
        _, dark_mask = cv2.threshold(gray, self.analyzer.params.BINARY_DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        self.processing_images['dark_mask'] = dark_mask
        
        # Convert to 3-channel for display
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
        inpainted = cv2.inpaint(self.analyzer.original_image, dark_mask, self.analyzer.params.BINARY_INPAINT_RADIUS, cv2.INPAINT_TELEA)
        self.processing_images['inpainted'] = inpainted
        
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
        
        # Convert to 3-channel for display
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
        
        # Convert to 3-channel for display
        binary_display = cv2.cvtColor(binary_plate, cv2.COLOR_GRAY2BGR)
        self.display_image(binary_display)
        
        self.current_step = 5
        self.log_status(f"✅ Step 5: Applied binary threshold ({self.analyzer.params.BINARY_THRESHOLD})")
        
    def step6_detect_binary(self):
        """Step 6: Detect binary centroids with morphological operations"""
        if 'binary_plate' not in self.processing_images:
            self.log_status("❌ Run Step 5 first")
            return
            
        binary_plate = self.processing_images['binary_plate']
        
        # Apply morphological operations
        processed_binary = self.apply_morphological_operations(binary_plate)
        self.processing_images['processed_binary'] = processed_binary
        
        # Find centroids
        centroids = self.analyzer.find_binary_centroids(processed_binary)
        self.analyzer.binary_centroids = centroids
        self.processing_images['binary_centroids'] = centroids
        
        # Display original image with centroids
        result_img = self.analyzer.original_image.copy()
        for cx, cy in centroids:
            cv2.circle(result_img, (cx, cy), 8, (255, 0, 0), 2)  # Blue circles
            cv2.circle(result_img, (cx, cy), 2, (255, 0, 0), -1)
        
        self.display_image(result_img)
        
        self.current_step = 6
        self.binary_complete = True
        self.erosion_btn.setEnabled(True)  # Enable erosion visualization
        self.log_status(f"✅ Step 6: Detected {len(centroids)} binary centroids")
        self.log_status(f"   Erosion stages: {self.analyzer.params.BINARY_EROSION_STAGES}")
        self.log_status(f"   Kernel size: {self.kernel_size}x{self.kernel_size}")
        self.log_status(f"   Dilation iterations: {self.dilation_iterations}")
        self.check_continue_ready()
        
    def apply_morphological_operations(self, binary_image):
        """Apply morphological operations to binary image"""
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
        
        return inverted_binary
        
    def show_erosion_stages(self):
        """Show visualization of binary erosion stages"""
        if 'binary_plate' not in self.processing_images:
            self.log_status("❌ No binary image available for erosion visualization")
            return
            
        binary_plate = self.processing_images['binary_plate']
        
        # Apply mask filtering first (same as in detection)
        inverted_binary = cv2.bitwise_not(binary_plate)
        
        # Apply mask filtering if mask is available
        if hasattr(self.analyzer, 'mask_path') and self.analyzer.mask_path and self.analyzer.mask_path != "dummy_mask.png":
            try:
                mask = cv2.imread(self.analyzer.mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Resize mask to match image if needed
                    if mask.shape != inverted_binary.shape:
                        mask = cv2.resize(mask, (inverted_binary.shape[1], inverted_binary.shape[0]))
                    
                    # Apply mask filter - keep only white areas of mask
                    inverted_binary = cv2.bitwise_and(inverted_binary, mask)
            except Exception as e:
                pass  # Continue without mask filtering
        
        # Create kernel
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        
        # Calculate number of stages to show
        max_stages = min(self.analyzer.params.BINARY_EROSION_STAGES + 2, 8)
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Binary Erosion Stages Visualization (Kernel: {self.kernel_size}x{self.kernel_size})', fontsize=14)
        
        axes = axes.flatten()
        
        # Show original binary (after mask filtering)
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
        
        plt.tight_layout()
        plt.show()
        
        self.log_status("📊 Binary erosion stages visualization displayed")
        
    def color_step1_load_image(self):
        """Color Step 1: Load original image"""
        if self.analyzer.original_image is not None:
            self.color_processing_images['original'] = self.analyzer.original_image
            self.display_image(self.analyzer.original_image)
            self.color_current_step = 1
            self.log_status("✅ Color Step 1: Original image loaded")
        else:
            self.log_status("❌ No image available")
            
    def color_step2_sample_colors(self):
        """Color Step 2: Sample colors interactively"""
        self.log_status("🎨 Color Step 2: Starting color sampling...")
        self.log_status("Use the color sampling interface to collect samples")
        
        # Run color sampling
        self.analyzer.run_color_sampling()
        
        if self.analyzer.sample_masks:
            self.color_current_step = 2
            self.log_status(f"✅ Color Step 2: Collected {len(self.analyzer.sample_masks)} color samples")
        else:
            self.log_status("❌ No color samples collected")
            
    def color_step3_create_mask(self):
        """Color Step 3: Create color mask from samples"""
        if not self.analyzer.sample_masks:
            self.log_status("❌ Run Color Step 2 first")
            return
            
        # Collect sample pixels
        sample_pixels = []
        for mask in self.analyzer.sample_masks:
            pixels = self.analyzer.original_image[mask > 0]
            if len(pixels) > 0:
                sample_pixels.append(pixels)
        
        # Create color filter
        combined_mask = np.zeros(self.analyzer.original_image.shape[:2], dtype=np.uint8)
        
        for pixels in sample_pixels:
            mean_color = np.mean(pixels, axis=0)
            std_color = np.std(pixels, axis=0)
            
            # Create color range (mean ± 2*std)
            lower_bound = np.maximum(0, mean_color - 2 * std_color).astype(np.uint8)
            upper_bound = np.minimum(255, mean_color + 2 * std_color).astype(np.uint8)
            
            # Create mask for this color range
            mask = cv2.inRange(self.analyzer.original_image, lower_bound, upper_bound)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        self.color_processing_images['color_mask'] = combined_mask
        
        # Display the mask
        mask_display = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        self.display_image(mask_display)
        
        self.color_current_step = 3
        self.log_status("✅ Color Step 3: Created color mask from samples")
        
    def color_step4_blur(self):
        """Color Step 4: Apply Gaussian blur"""
        if 'color_mask' not in self.color_processing_images:
            self.log_status("❌ Run Color Step 3 first")
            return
            
        color_mask = self.color_processing_images['color_mask']
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(color_mask, (self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE, self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE), 0)
        self.color_processing_images['blurred'] = blurred
        
        # Display blurred mask
        blur_display = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        self.display_image(blur_display)
        
        self.color_current_step = 4
        self.log_status(f"✅ Color Step 4: Applied Gaussian blur (size: {self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE})")
        
    def color_step5_canny(self):
        """Color Step 5: Apply Canny edge detection"""
        if 'blurred' not in self.color_processing_images:
            self.log_status("❌ Run Color Step 4 first")
            return
            
        blurred = self.color_processing_images['blurred']
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD, self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD)
        self.color_processing_images['edges'] = edges
        
        # Display edges
        edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        self.display_image(edges_display)
        
        self.color_current_step = 5
        self.log_status(f"✅ Color Step 5: Applied Canny edge detection")
        self.log_status(f"   Low: {self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD}, High: {self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD}")
        
    def color_step6_erosion(self):
        """Color Step 6: Apply erosion"""
        if 'edges' not in self.color_processing_images:
            self.log_status("❌ Run Color Step 5 first")
            return
            
        edges = self.color_processing_images['edges']
        
        # Apply erosion
        kernel = np.ones((3, 3), np.uint8)
        if self.analyzer.params.COLOR_EROSION_STAGES > 0:
            eroded = cv2.erode(edges, kernel, iterations=self.analyzer.params.COLOR_EROSION_STAGES)
        else:
            eroded = edges
            
        self.color_processing_images['eroded'] = eroded
        
        # Display eroded edges
        eroded_display = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        self.display_image(eroded_display)
        
        self.color_current_step = 6
        self.color_erosion_viz_btn.setEnabled(True)  # Enable erosion visualization
        self.log_status(f"✅ Color Step 6: Applied erosion ({self.analyzer.params.COLOR_EROSION_STAGES} stages)")
        
    def color_step7_detect(self):
        """Color Step 7: Detect color centroids"""
        if 'eroded' not in self.color_processing_images:
            self.log_status("❌ Run Color Step 6 first")
            return
            
        eroded = self.color_processing_images['eroded']
        
        # Find centroids
        centroids = self.analyzer.find_circular_contours_with_centroids(eroded)
        self.analyzer.color_centroids = centroids
        self.color_processing_images['color_centroids'] = centroids
        
        # Display original image with all centroids
        result_img = self.analyzer.original_image.copy()
        
        # Draw binary centroids in blue
        for cx, cy in self.analyzer.binary_centroids:
            cv2.circle(result_img, (cx, cy), 8, (255, 0, 0), 2)
            cv2.circle(result_img, (cx, cy), 2, (255, 0, 0), -1)
        
        # Draw color centroids in red
        for cx, cy in centroids:
            cv2.circle(result_img, (cx, cy), 8, (0, 0, 255), 2)  # Red circles
            cv2.circle(result_img, (cx, cy), 2, (0, 0, 255), -1)
        
        self.display_image(result_img)
        
        self.color_current_step = 7
        self.color_complete = True
        self.log_status(f"✅ Color Step 7: Detected {len(centroids)} color centroids")
        self.check_continue_ready()
        
    def show_color_erosion_stages(self):
        """Show visualization of color erosion stages"""
        if 'edges' not in self.color_processing_images:
            self.log_status("❌ No edge image available for erosion visualization")
            return
            
        edges = self.color_processing_images['edges']
        
        # Create kernel
        kernel = np.ones((3, 3), np.uint8)
        
        # Calculate number of stages to show
        max_stages = min(self.analyzer.params.COLOR_EROSION_STAGES + 2, 8)
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Color Erosion Stages Visualization', fontsize=14)
        
        axes = axes.flatten()
        
        # Show original edges
        axes[0].imshow(edges, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original Edges')
        axes[0].axis('off')
        axes[0].set_aspect('equal')
        
        # Show erosion stages
        current_image = edges.copy()
        for i in range(1, max_stages):
            if i <= self.analyzer.params.COLOR_EROSION_STAGES:
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
        
        plt.tight_layout()
        plt.show()
        
        self.log_status("📊 Color erosion stages visualization displayed")
        
    def update_dark_threshold(self, value):
        """Update dark threshold parameter"""
        self.analyzer.params.BINARY_DARK_THRESHOLD = value
        self.dark_label.setText(str(value))
        self.save_parameters()
        
        # Re-run step 2 if we're past it
        if self.current_step >= 2:
            self.step2_dark_mask()
            
    def update_inpaint_radius(self, value):
        """Update inpaint radius parameter"""
        self.analyzer.params.BINARY_INPAINT_RADIUS = value
        self.inpaint_label.setText(str(value))
        self.save_parameters()
        
        # Re-run step 3 if we're past it
        if self.current_step >= 3:
            self.step3_inpainting()
            
    def update_binary_threshold(self, value):
        """Update binary threshold parameter"""
        self.analyzer.params.BINARY_THRESHOLD = value
        self.binary_label.setText(str(value))
        self.save_parameters()
        
        # Re-run step 5 if we're past it
        if self.current_step >= 5:
            self.step5_binary()
            
    def update_erosion_stages(self, value):
        """Update erosion stages parameter"""
        self.analyzer.params.BINARY_EROSION_STAGES = value
        self.erosion_label.setText(str(value))
        self.save_parameters()
        
        # Re-run step 6 if we're past it
        if self.current_step >= 6:
            self.step6_detect_binary()
            
        # Enable erosion visualization if we have a binary image
        if 'processed_binary' in self.processing_images:
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
        self.save_parameters()
        
        # Re-run operations if needed
        if self.current_step >= 6:
            self.step6_detect_binary()
            
    def update_dilation_iterations(self, value):
        """Update dilation iterations"""
        self.dilation_label.setText(str(value))
        self.dilation_iterations = value
        self.save_parameters()
        
        # Re-run operations if needed
        if self.current_step >= 6:
            self.step6_detect_binary()
    
    def update_min_diameter(self, value):
        """Update minimum diameter parameter"""
        self.analyzer.params.BINARY_MIN_DIAMETER = value
        self.min_diameter_label.setText(str(value))
        self.save_parameters()
        
        # Re-run step 6 if we're past it
        if self.current_step >= 6:
            self.step6_detect_binary()
        
    def update_max_diameter(self, value):
        """Update maximum diameter parameter"""
        self.analyzer.params.BINARY_MAX_DIAMETER = value
        self.max_diameter_label.setText(str(value))
        self.save_parameters()
        
        # Re-run step 6 if we're past it
        if self.current_step >= 6:
            self.step6_detect_binary()
        
    def update_circularity(self, value):
        """Update circularity threshold parameter"""
        threshold = value / 100.0
        self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD = threshold
        self.circularity_label.setText(f"{threshold:.2f}")
        self.save_parameters()
        
        # Re-run step 6 if we're past it
        if self.current_step >= 6:
            self.step6_detect_binary()
            
    def update_color_min_diameter(self, value):
        """Update color detection minimum diameter"""
        self.analyzer.params.COLOR_MIN_DIAMETER = value
        self.color_min_diameter_label.setText(str(value))
        self.save_parameters()
        
    def update_color_max_diameter(self, value):
        """Update color detection maximum diameter"""
        self.analyzer.params.COLOR_MAX_DIAMETER = value
        self.color_max_diameter_label.setText(str(value))
        self.save_parameters()
        
    def update_color_circularity(self, value):
        """Update color detection circularity threshold"""
        threshold = value / 100.0
        self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD = threshold
        self.color_circularity_label.setText(f"{threshold:.2f}")
        self.save_parameters()
        
    def update_color_erosion(self, value):
        """Update color detection erosion stages"""
        self.analyzer.params.COLOR_EROSION_STAGES = value
        self.color_erosion_label.setText(str(value))
        self.save_parameters()
        
    def update_blur_size(self, value):
        """Update Gaussian blur size"""
        # Ensure odd number for blur size
        if value % 2 == 0:
            value += 1
        self.blur_slider.setValue(value)
        self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE = value
        self.blur_label.setText(str(value))
        self.save_parameters()
        
    def update_canny_low(self, value):
        """Update Canny low threshold"""
        self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD = value
        self.canny_low_label.setText(str(value))
        self.save_parameters()
        
        # Ensure high threshold is greater than low threshold
        if self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD <= value:
            new_high = value + 10
            self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD = new_high
            self.canny_high_slider.setValue(new_high)
            self.canny_high_label.setText(str(new_high))
            
    def update_canny_high(self, value):
        """Update Canny high threshold"""
        self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD = value
        self.canny_high_label.setText(str(value))
        self.save_parameters()
        
        # Ensure high threshold is greater than low threshold
        if self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD >= value:
            self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD = max(1, value - 10)
            self.canny_low_slider.setValue(self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD)
            self.canny_low_label.setText(str(self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD))
            
    def update_merge_threshold(self, value):
        """Update centroid merge threshold"""
        self.analyzer.params.CENTROID_MERGE_THRESHOLD = value
        self.merge_threshold_label.setText(str(value))
        
    def display_image(self, cv_image):
        """Display OpenCV image in the QLabel"""
        if cv_image is None:
            return
            
        # Convert BGR to RGB
        if len(cv_image.shape) == 3:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Create QImage
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        label_size = self.image_label.size()
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        self.image_label.setPixmap(scaled_pixmap)
        
    def log_status(self, message):
        """Add message to status log"""
        self.status_text.append(message)
        self.status_text.ensureCursorVisible()
        
    def continue_analysis(self):
        """Continue with the complete analysis pipeline"""
        self.log_status("🚀 Continuing with complete analysis...")
        
        # Save final parameters
        self.save_parameters()
        
        # Create visualization with detected organoids
        self.analyzer.create_image_with_organoids()
        self.log_status(f"✅ Created visualization: {self.analyzer.image_with_organoids_path}")
        
        # Continue with well analysis if enabled
        if self.analyzer.params.ENABLE_WELL_ANALYSIS:
            self.analyzer.create_well_crops()
            self.analyzer.correlate_centroids_to_wells()
            
        # Save results
        self.analyzer.save_results_csv()
        
        # Show final summary
        total_binary = len(self.analyzer.binary_centroids)
        total_color = len(self.analyzer.color_centroids)
        total_organoids = total_binary + total_color
        
        self.log_status(f"\n📊 ANALYSIS COMPLETE!")
        self.log_status(f"   Binary detections: {total_binary}")
        self.log_status(f"   Color detections: {total_color}")
        self.log_status(f"   Total organoids: {total_organoids}")
        
        if hasattr(self.analyzer, 'well_crops') and self.analyzer.well_crops:
            self.log_status(f"   Wells analyzed: {len(self.analyzer.well_crops)}")
            
        # Close the debug GUI
        self.close()
        
    def closeEvent(self, event):
        """Handle window close event"""
        cv2.destroyAllWindows()
        event.accept()

    def load_parameters(self):
        """Load parameters from JSON file"""
        if os.path.exists(self.param_file):
            try:
                with open(self.param_file, 'r') as f:
                    params = json.load(f)
                
                # Load binary parameters
                if 'binary' in params:
                    bp = params['binary']
                    self.analyzer.params.BINARY_DARK_THRESHOLD = bp.get('dark_threshold', 80)
                    self.analyzer.params.BINARY_INPAINT_RADIUS = bp.get('inpaint_radius', 20)
                    self.analyzer.params.BINARY_THRESHOLD = bp.get('binary_threshold', 110)
                    self.analyzer.params.BINARY_MIN_DIAMETER = bp.get('min_diameter', 15)
                    self.analyzer.params.BINARY_MAX_DIAMETER = bp.get('max_diameter', 50)
                    self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD = bp.get('circularity', 0.75)
                    self.analyzer.params.BINARY_EROSION_STAGES = bp.get('erosion_stages', 4)
                    self.kernel_size = bp.get('kernel_size', 3)
                    self.dilation_iterations = bp.get('dilation_iterations', 0)
                
                # Load color parameters
                if 'color' in params:
                    cp = params['color']
                    self.analyzer.params.COLOR_MIN_DIAMETER = cp.get('min_diameter', 30)
                    self.analyzer.params.COLOR_MAX_DIAMETER = cp.get('max_diameter', 110)
                    self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD = cp.get('circularity', 0.6)
                    self.analyzer.params.COLOR_EROSION_STAGES = cp.get('erosion_stages', 9)
                    self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE = cp.get('blur_size', 3)
                    self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD = cp.get('canny_low', 50)
                    self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD = cp.get('canny_high', 150)
                
                # Load merge parameters
                if 'merge' in params:
                    mp = params['merge']
                    self.analyzer.params.CENTROID_MERGE_THRESHOLD = mp.get('threshold', 24)
                
                self.log_status("✅ Parameters loaded from file")
                
            except Exception as e:
                self.log_status(f"❌ Error loading parameters: {e}")
    
    def save_parameters(self):
        """Save current parameters to JSON file"""
        try:
            params = {
                'binary': {
                    'dark_threshold': self.analyzer.params.BINARY_DARK_THRESHOLD,
                    'inpaint_radius': self.analyzer.params.BINARY_INPAINT_RADIUS,
                    'binary_threshold': self.analyzer.params.BINARY_THRESHOLD,
                    'min_diameter': self.analyzer.params.BINARY_MIN_DIAMETER,
                    'max_diameter': self.analyzer.params.BINARY_MAX_DIAMETER,
                    'circularity': self.analyzer.params.BINARY_CIRCULARITY_THRESHOLD,
                    'erosion_stages': self.analyzer.params.BINARY_EROSION_STAGES,
                    'kernel_size': self.kernel_size,
                    'dilation_iterations': self.dilation_iterations
                },
                'color': {
                    'min_diameter': self.analyzer.params.COLOR_MIN_DIAMETER,
                    'max_diameter': self.analyzer.params.COLOR_MAX_DIAMETER,
                    'circularity': self.analyzer.params.COLOR_CIRCULARITY_THRESHOLD,
                    'erosion_stages': self.analyzer.params.COLOR_EROSION_STAGES,
                    'blur_size': self.analyzer.params.COLOR_GAUSSIAN_BLUR_SIZE,
                    'canny_low': self.analyzer.params.COLOR_CANNY_LOW_THRESHOLD,
                    'canny_high': self.analyzer.params.COLOR_CANNY_HIGH_THRESHOLD
                },
                'merge': {
                    'threshold': self.analyzer.params.CENTROID_MERGE_THRESHOLD
                }
            }
            
            with open(self.param_file, 'w') as f:
                json.dump(params, f, indent=2)
                
            self.log_status("✅ Parameters saved to file")
            
        except Exception as e:
            self.log_status(f"❌ Error saving parameters: {e}")

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