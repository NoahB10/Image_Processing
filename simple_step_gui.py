#!/usr/bin/env python3
"""
Simple Step-by-Step GUI for Binary Detection
Only processes what you click - no automatic processing
"""

import sys
import os
import cv2
import numpy as np
from tkinter import filedialog, Tk

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox, 
                                QDoubleSpinBox, QGroupBox, QGridLayout, QSplitter)
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap, QImage
    PYQT_AVAILABLE = True
except ImportError:
    print("❌ PyQt6 not available")
    sys.exit(1)

class SimpleStepGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Step-by-Step Binary Detection")
        self.setGeometry(100, 100, 1400, 800)
        
        # Image data
        self.original_image = None
        self.current_step_image = None
        
        # Parameters
        self.dark_threshold = 122
        self.inpaint_radius = 20
        self.binary_threshold = 163
        
        # Step results (only calculated when needed)
        self.grayscale = None
        self.dark_mask = None
        self.inpainted = None
        self.inpainted_gray = None
        self.binary_result = None
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Load button
        load_btn = QPushButton("Load Images")
        load_btn.clicked.connect(self.load_images)
        left_layout.addWidget(load_btn)
        
        # Step buttons
        step_group = QGroupBox("Processing Steps")
        step_layout = QVBoxLayout(step_group)
        
        self.step1_btn = QPushButton("Step 1: Convert to Grayscale")
        self.step1_btn.clicked.connect(self.step1_grayscale)
        self.step1_btn.setEnabled(False)
        step_layout.addWidget(self.step1_btn)
        
        self.step2_btn = QPushButton("Step 2: Create Dark Mask")
        self.step2_btn.clicked.connect(self.step2_dark_mask)
        self.step2_btn.setEnabled(False)
        step_layout.addWidget(self.step2_btn)
        
        self.step3_btn = QPushButton("Step 3: Apply Inpainting")
        self.step3_btn.clicked.connect(self.step3_inpainting)
        self.step3_btn.setEnabled(False)
        step_layout.addWidget(self.step3_btn)
        
        self.step4_btn = QPushButton("Step 4: Convert to Grayscale")
        self.step4_btn.clicked.connect(self.step4_inpaint_gray)
        self.step4_btn.setEnabled(False)
        step_layout.addWidget(self.step4_btn)
        
        self.step5_btn = QPushButton("Step 5: Apply Binary Threshold")
        self.step5_btn.clicked.connect(self.step5_binary)
        self.step5_btn.setEnabled(False)
        step_layout.addWidget(self.step5_btn)
        
        left_layout.addWidget(step_group)
        
        # Parameters
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)
        
        # Dark threshold
        dark_layout = QHBoxLayout()
        dark_layout.addWidget(QLabel("Dark Threshold:"))
        self.dark_slider = QSlider(Qt.Orientation.Horizontal)
        self.dark_slider.setRange(0, 255)
        self.dark_slider.setValue(self.dark_threshold)
        self.dark_spinbox = QSpinBox()
        self.dark_spinbox.setRange(0, 255)
        self.dark_spinbox.setValue(self.dark_threshold)
        self.dark_slider.valueChanged.connect(self.dark_spinbox.setValue)
        self.dark_spinbox.valueChanged.connect(self.dark_slider.setValue)
        self.dark_slider.valueChanged.connect(self.update_dark_threshold)
        dark_layout.addWidget(self.dark_slider)
        dark_layout.addWidget(self.dark_spinbox)
        param_layout.addLayout(dark_layout)
        
        # Inpaint radius
        inpaint_layout = QHBoxLayout()
        inpaint_layout.addWidget(QLabel("Inpaint Radius:"))
        self.inpaint_slider = QSlider(Qt.Orientation.Horizontal)
        self.inpaint_slider.setRange(1, 50)
        self.inpaint_slider.setValue(self.inpaint_radius)
        self.inpaint_spinbox = QSpinBox()
        self.inpaint_spinbox.setRange(1, 50)
        self.inpaint_spinbox.setValue(self.inpaint_radius)
        self.inpaint_slider.valueChanged.connect(self.inpaint_spinbox.setValue)
        self.inpaint_spinbox.valueChanged.connect(self.inpaint_slider.setValue)
        self.inpaint_slider.valueChanged.connect(self.update_inpaint_radius)
        inpaint_layout.addWidget(self.inpaint_slider)
        inpaint_layout.addWidget(self.inpaint_spinbox)
        param_layout.addLayout(inpaint_layout)
        
        # Binary threshold
        binary_layout = QHBoxLayout()
        binary_layout.addWidget(QLabel("Binary Threshold:"))
        self.binary_slider = QSlider(Qt.Orientation.Horizontal)
        self.binary_slider.setRange(0, 255)
        self.binary_slider.setValue(self.binary_threshold)
        self.binary_spinbox = QSpinBox()
        self.binary_spinbox.setRange(0, 255)
        self.binary_spinbox.setValue(self.binary_threshold)
        self.binary_slider.valueChanged.connect(self.binary_spinbox.setValue)
        self.binary_spinbox.valueChanged.connect(self.binary_slider.setValue)
        self.binary_slider.valueChanged.connect(self.update_binary_threshold)
        binary_layout.addWidget(self.binary_slider)
        binary_layout.addWidget(self.binary_spinbox)
        param_layout.addLayout(binary_layout)
        
        left_layout.addWidget(param_group)
        
        # Status
        self.status_label = QLabel("Click 'Load Images' to start")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        
        # Right panel - Image display
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 1px solid gray;")
        right_layout.addWidget(self.image_label)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
    def load_images(self):
        """Load images"""
        print("Loading images...")
        root = Tk()
        root.withdraw()
        
        image_path = filedialog.askopenfilename(
            title="Select Main Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if not image_path:
            return
            
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            self.status_label.setText("Failed to load image")
            return
            
        # Reset all step results
        self.grayscale = None
        self.dark_mask = None
        self.inpainted = None
        self.inpainted_gray = None
        self.binary_result = None
        
        # Show original image
        self.display_image(self.original_image)
        self.status_label.setText("Image loaded. Click Step 1 to start processing.")
        
        # Enable step 1
        self.step1_btn.setEnabled(True)
        self.step2_btn.setEnabled(False)
        self.step3_btn.setEnabled(False)
        self.step4_btn.setEnabled(False)
        self.step5_btn.setEnabled(False)
        
    def step1_grayscale(self):
        """Step 1: Convert to grayscale"""
        if self.original_image is None:
            return
            
        print("Step 1: Converting to grayscale...")
        self.grayscale = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Display grayscale
        gray_bgr = cv2.cvtColor(self.grayscale, cv2.COLOR_GRAY2BGR)
        self.display_image(gray_bgr)
        self.status_label.setText("Step 1 complete: Converted to grayscale")
        
        # Enable step 2
        self.step2_btn.setEnabled(True)
        
    def step2_dark_mask(self):
        """Step 2: Create dark mask"""
        if self.grayscale is None:
            return
            
        print(f"Step 2: Creating dark mask with threshold {self.dark_threshold}...")
        self.dark_mask = (self.grayscale < self.dark_threshold).astype(np.uint8) * 255
        
        # Display dark mask
        mask_bgr = cv2.cvtColor(self.dark_mask, cv2.COLOR_GRAY2BGR)
        self.display_image(mask_bgr)
        self.status_label.setText(f"Step 2 complete: Dark mask created (threshold: {self.dark_threshold})")
        
        # Enable step 3
        self.step3_btn.setEnabled(True)
        
    def step3_inpainting(self):
        """Step 3: Apply inpainting"""
        if self.dark_mask is None:
            return
            
        print(f"Step 3: Applying inpainting with radius {self.inpaint_radius}...")
        self.inpainted = cv2.inpaint(self.original_image, self.dark_mask, 
                                   inpaintRadius=self.inpaint_radius, 
                                   flags=cv2.INPAINT_TELEA)
        
        # Display inpainted
        self.display_image(self.inpainted)
        self.status_label.setText(f"Step 3 complete: Inpainting applied (radius: {self.inpaint_radius})")
        
        # Enable step 4
        self.step4_btn.setEnabled(True)
        
    def step4_inpaint_gray(self):
        """Step 4: Convert inpainted to grayscale"""
        if self.inpainted is None:
            return
            
        print("Step 4: Converting inpainted to grayscale...")
        self.inpainted_gray = cv2.cvtColor(self.inpainted, cv2.COLOR_BGR2GRAY)
        
        # Display inpainted grayscale
        gray_bgr = cv2.cvtColor(self.inpainted_gray, cv2.COLOR_GRAY2BGR)
        self.display_image(gray_bgr)
        self.status_label.setText("Step 4 complete: Inpainted image converted to grayscale")
        
        # Enable step 5
        self.step5_btn.setEnabled(True)
        
    def step5_binary(self):
        """Step 5: Apply binary threshold"""
        if self.inpainted_gray is None:
            return
            
        print(f"Step 5: Applying binary threshold {self.binary_threshold}...")
        _, self.binary_result = cv2.threshold(self.inpainted_gray, self.binary_threshold, 255, cv2.THRESH_BINARY)
        
        # Display binary result
        binary_bgr = cv2.cvtColor(self.binary_result, cv2.COLOR_GRAY2BGR)
        self.display_image(binary_bgr)
        self.status_label.setText(f"Step 5 complete: Binary threshold applied ({self.binary_threshold})")
        
    def update_dark_threshold(self, value):
        """Update dark threshold parameter"""
        self.dark_threshold = value
        # Clear dependent results
        self.dark_mask = None
        self.inpainted = None
        self.inpainted_gray = None
        self.binary_result = None
        
        # Disable dependent steps
        self.step3_btn.setEnabled(False)
        self.step4_btn.setEnabled(False)
        self.step5_btn.setEnabled(False)
        
    def update_inpaint_radius(self, value):
        """Update inpaint radius parameter"""
        self.inpaint_radius = value
        # Clear dependent results
        self.inpainted = None
        self.inpainted_gray = None
        self.binary_result = None
        
        # Disable dependent steps
        self.step4_btn.setEnabled(False)
        self.step5_btn.setEnabled(False)
        
    def update_binary_threshold(self, value):
        """Update binary threshold parameter"""
        self.binary_threshold = value
        # Clear dependent results
        self.binary_result = None
        
    def display_image(self, cv_image):
        """Display OpenCV image in the label"""
        if cv_image is None:
            return
            
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Get label size
        label_size = self.image_label.size()
        
        # Resize image to fit label while maintaining aspect ratio
        h, w, ch = rgb_image.shape
        aspect_ratio = w / h
        
        if label_size.width() / label_size.height() > aspect_ratio:
            new_height = label_size.height()
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = label_size.width()
            new_height = int(new_width / aspect_ratio)
        
        # Resize image
        resized = cv2.resize(rgb_image, (new_width, new_height))
        
        # Convert to QImage
        bytes_per_line = ch * new_width
        qt_image = QImage(resized.data, new_width, new_height, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Convert to QPixmap and set to label
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

def main():
    print("🚀 Launching Simple Step-by-Step GUI...")
    
    app = QApplication(sys.argv)
    
    gui = SimpleStepGUI()
    gui.show()
    
    print("✅ GUI opened! Click 'Load Images' to start.")
    
    app.exec()

if __name__ == "__main__":
    main() 