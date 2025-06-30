# Well Segmentation Tools

Drive Link: https://drive.google.com/drive/folders/1qWni3H6G9q5a0Jn-nBVRPilVw8qVFWvn?usp=sharing 


This folder contains specialized tools for well segmentation, mask creation, and image processing specifically designed for microplate analysis and cell culture imaging.

Important codes to remember
1. Code which splits up the image into individual wells is called: Mask_Cropper
2. Code which filters the wells by color to focus on the organoids is two stages
  1. The plate is inpainted to remove hard edges
  2. The whole plate is filtered into colors which are organoid specifc
  3. The plate is segmented into wells and threhold applied to isolate organoids
3. Code which filters the wells by color focusing on wells is called Circle_Droper_Filter
4. Code which takes all of the CSV's and images and combines them into detected wells is called Locations_Combine_Organoids 

WEll finder is a useful code to translate the coordinates from organoid detection over to abosolute coordinates

## Overview

The Well Segmentation toolkit provides comprehensive solutions for:
- Interactive mask creation and editing
- Color-based filtering and segmentation
- Multiple exposure image processing
- Background isolation and cleanup
- Well detection and analysis
- Batch processing of microplate images

## Core Tools

### Interactive Tools

#### `mask_cropper.py`
Interactive tool for cropping wells from images based on masks.
DOES NOT SEEM TO THRESHOLD THE IMAGES!
- **Purpose**: Extract individual wells from microplate images
- **Features**: Interactive selection, batch processing, mask-based cropping
- **Usage**: Load image and mask, select wells, export cropped regions
- **Output**: Individual well images with metadata

#### `interactive_mask_creator.py`
Advanced mask creation tool with interactive editing capabilities.
- **Purpose**: Create and edit complex masks for image processing
- **Features**: Drawing tools, mask editing, real-time preview
- **Usage**: Load image, create masks using various tools, save results
- **Output**: High-quality masks for downstream processing

#### `color_filter_from_samples.py`
Interactive color filtering tool with circular selection system.
- **Purpose**: Filter images based on color samples from user-selected areas
- **Features**: Circular sample selection, zoom/pan controls, iterative refinement
- **Controls**:
  - Right-click: Place circular sample areas
  - Left-click + drag: Pan around image
  - Mouse wheel: Zoom in/out
  - `[` / `]` keys: Adjust circle size
  - Space: Apply filter, `r`: Reset samples, `s`: Save results
- **Output**: Filtered images and binary masks

#### `Multiple exposure filtering.py`
Advanced tool for processing multiple exposure images with mask-based analysis.
- **Purpose**: Analyze multiple exposures to extract common features
- **Features**: Circular mask selection, color learning, AND operations, comprehensive visualization
- **Workflow**: Select sample areas → Learn colors per image → Filter entire images → AND operation → Binary result
- **Controls**: Similar to color_filter_from_samples.py with additional multi-image capabilities
- **Output**: Individual filtered images, final AND result, binary masks

### Processing Tools

#### `Background_Isolation.py`
This code will run inpainting twice and will need an edges file to filli n the second stage!
Background isolation and cleanup tool.
- **Purpose**: Isolate subjects from backgrounds
- **Features**: Automatic background detection, edge preservation
- **Usage**: Load image, configure parameters, process
- **Output**: Subject-isolated images

#### `Uniform_Masking.py`
Comprehensive uniform masking solution.
- **Purpose**: Apply consistent masking across image sets
- **Features**: Batch processing, mask standardization, quality control
- **Usage**: Load image set and reference mask, apply uniform processing
- **Output**: Consistently masked image set

### Utility Scripts

#### `Color_AVG.py`
Color analysis and averaging tool.
- **Purpose**: Analyze color distributions and calculate averages
- **Features**: Statistical analysis, color space conversions
- **Output**: Color statistics and analysis reports

#### `AI_Enhance.py`
AI-based image enhancement.
- **Purpose**: Improve image quality using AI algorithms
- **Features**: Noise reduction, contrast enhancement, detail preservation
- **Output**: Enhanced images with quality metrics

#### `Delete_Without_Organoids.py`
Automated organoid detection and cleanup.
- **Purpose**: Remove wells without organoids from datasets
- **Features**: Automatic detection, batch processing
- **Output**: Cleaned datasets with organoid-containing wells only

## Data Files

### Images
- `*.png` files: Various test images and masks
- `*.pp3` files: Processing parameter files
- `Fav_Translucent_Plate.png`: Reference plate image
- `Updated_final_adjusted_mask.png`: Current working mask

### Analysis Results
- `Well_objects.csv`: Well detection and analysis results
- `plate_report.html`: Comprehensive analysis report
- `Blue.csv`, `Green.csv`, `Red.csv`: Color channel analysis results

### Masks and References
- `uniform_squares_mask.png`: Standard grid mask
- `final_adjusted_mask.png`: Manually adjusted mask
- `calibrated_uniform_mask.png`: Calibrated mask for consistent processing

## Subtraction Trick Folder

### `Subtract_Images.py`
Advanced image masking and processing tool.
- **Purpose**: Apply masks to images with optional processing
- **Features**:
  - Interactive processing options dialog
  - Morphological dilation with adjustable parameters
  - Island size filtering (removes small black objects)
  - Optional mask inversion
  - Comprehensive visualization (3x3 or 2x3 grid displays)
- **Workflow**: Select images → Configure processing → Apply mask → Save results
- **Processing Options**:
  - **Mask Inversion**: Toggle between direct and inverted masking
  - **Dilation**: Expand features with customizable kernel size and iterations
  - **Island Filter**: Remove small black artifacts below threshold size
- **Output**: Masked images with detailed processing metadata

## Usage Guidelines

### Getting Started
1. **Environment Setup**: Ensure required libraries are installed (see `Workflow/libraries.txt`)
2. **Input Preparation**: Place raw images in appropriate subfolders
3. **Tool Selection**: Choose appropriate tool based on your analysis needs
4. **Processing**: Follow tool-specific workflows
5. **Quality Control**: Verify results before proceeding to next steps

### Best Practices
- Always backup original data before processing
- Use consistent naming conventions for outputs
- Document processing parameters in metadata files
- Validate results with known reference samples
- Keep processing logs for reproducibility

### Output Organization
- Individual tool outputs saved in working directory
- Batch processing results organized by timestamp
- Masks and metadata saved alongside processed images
- Analysis reports generated in HTML format for review

## Dependencies

Required libraries (see `Workflow/libraries.txt` for details):
- OpenCV (cv2): Image processing operations
- NumPy: Numerical computations
- Matplotlib: Visualization and plotting
- Tkinter: GUI interfaces
- Pandas: Data analysis (for CSV processing)

## Error Handling

All tools include comprehensive error handling:
- File validation before processing
- Memory management for large datasets
- Graceful handling of user cancellations
- Detailed error messages with troubleshooting hints

## Version Information

Tools are continuously updated based on research needs. Check individual file headers for version information and recent changes.

## Support

For issues or questions:
1. Check tool-specific help messages and documentation
2. Verify input file formats and requirements
3. Ensure all dependencies are properly installed
4. Review processing logs for error details

---

*This documentation follows the project standards outlined in `Workflow/rules.txt`. Last updated: 2025*