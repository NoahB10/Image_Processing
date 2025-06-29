import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image

# This script is used to crop the wells from the image based on the mask.
# It labels each well according to the grid layout.
#there is a threshold which can be modified or removed depending on if it is determined necessary the threshold helps with finding organoids.

class MaskCropper:
    def __init__(self, input_image_path, mask_path, output_folder="Well_Crops", target_size=(600, 600)):
        """
        Initialize the MaskCropper
        
        Args:
            input_image_path: Path to the original image
            mask_path: Path to the binary mask
            output_folder: Folder to save cropped images
            target_size: Target size for resized images (width, height). Set to None to keep original crop sizes.
        """
        self.input_image_path = input_image_path
        self.mask_path = mask_path
        self.output_folder = output_folder
        self.target_size = target_size
        
        # Create output folder
        Path(self.output_folder).mkdir(exist_ok=True)
        print(f"Output folder created/verified: {self.output_folder}")
        if self.target_size:
            print(f"Target size for cropped images: {target_size[0]}x{target_size[1]}")
        else:
            print("Keeping original crop sizes (no resizing)")
        
        # Load images
        self.load_images()
    
    def load_images(self):
        """Load the input image and mask"""
        # Load original image
        self.image = cv2.imread(self.input_image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from: {self.input_image_path}")
        binarize = False
        if binarize:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = (gray > 160).astype(np.uint8) * 255
            #self.image = (gray < 200).astype(np.uint8) * 255
    
        # Load mask
        self.mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask is None:
            raise ValueError(f"Could not load mask from: {self.mask_path}")
        
        # Ensure mask is binary
        self.mask = np.where(self.mask > 127, 255, 0).astype(np.uint8)
        
        # Check dimensions match
        img_h, img_w = self.image.shape[:2]
        mask_h, mask_w = self.mask.shape
        
        if img_h != mask_h or img_w != mask_w:
            print(f"Warning: Image size ({img_w}x{img_h}) != Mask size ({mask_w}x{mask_h})")
            print("Resizing mask to match image...")
            self.mask = cv2.resize(self.mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        print(f"Loaded image: {img_w}x{img_h}")
        print(f"Loaded mask: {mask_w}x{mask_h}")
        print(f"White pixels in mask: {np.sum(self.mask == 255)}")
    
    def find_connected_regions(self):
        """Find all connected white regions in the mask"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.mask, connectivity=8, ltype=cv2.CV_32S
        )
        
        regions = []
        
        # Skip label 0 (background)
        for label in range(1, num_labels):
            # Get statistics for this region
            x, y, w, h, area = stats[label]
            
            # Skip very small regions (noise)
            if area < 10:  # Minimum 10 pixels
                continue
            
            # Create region info
            region_info = {
                'label': label,
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': centroids[label]
            }
            
            regions.append(region_info)
            
        print(f"Found {len(regions)} connected regions")
        
        # Sort regions by position for grid layout
        regions_with_grid = self.assign_grid_positions(regions)
        
        return regions_with_grid, labels
    
    def assign_grid_positions(self, regions):
        """Assign grid positions based on column-first ordering (A1, B1, C1, ..., P1, A2, B2, etc.)"""
        if not regions:
            return regions
        
        # First, group regions into columns based on X coordinate
        x_tolerance = 100  # pixels tolerance for same column (increased for better grouping)
        columns = []
        
        # Sort by X coordinate first to process columns left to right
        regions_by_x = sorted(regions, key=lambda r: r['centroid'][0])
        
        for region in regions_by_x:
            region_x = region['centroid'][0]
            
            # Find which column this region belongs to
            assigned_to_column = False
            for column in columns:
                # Use average X of all regions in column for better grouping
                column_avg_x = sum(r['centroid'][0] for r in column) / len(column)
                
                if abs(region_x - column_avg_x) <= x_tolerance:
                    # This region belongs to this column
                    column.append(region)
                    assigned_to_column = True
                    break
            
            if not assigned_to_column:
                # Create new column
                columns.append([region])
        
        # Sort each column by Y coordinate (top to bottom)
        for column in columns:
            column.sort(key=lambda r: r['centroid'][1])
        
        # Assign grid labels: A1, B1, C1, ..., P1, then A2, B2, C2, ..., P2, etc.
        regions_with_grid = []
        
        for col_idx, column in enumerate(columns):
            col_number = col_idx + 1  # 1, 2, 3, etc.
            
            for row_idx, region in enumerate(column):
                row_letter = chr(ord('A') + row_idx)  # A, B, C, D, ..., P
                grid_label = f"{row_letter}{col_number}"
                
                # Add grid information to region
                region['grid_label'] = grid_label
                region['grid_row'] = row_idx
                region['grid_col'] = col_idx
                
                regions_with_grid.append(region)
        
        # Sort final list by column then row for consistent processing
        regions_with_grid.sort(key=lambda r: (r['grid_col'], r['grid_row']))
        
        # Calculate layout info
        num_columns = len(columns)
        max_rows = max(len(column) for column in columns) if columns else 0
        
        print(f"Grid layout: {max_rows} rows x {num_columns} columns")
        print(f"Column grouping tolerance: {x_tolerance} pixels")
        
        # Debug: Show column sizes
        print("Column details:")
        for i, column in enumerate(columns):
            avg_x = sum(r['centroid'][0] for r in column) / len(column)
            print(f"  Column {i+1}: {len(column)} wells, avg X = {int(avg_x)}")
        
        print("Grid assignments:")
        for region in regions_with_grid:
            centroid_x, centroid_y = region['centroid']
            print(f"  {region['grid_label']}: center at ({int(centroid_x)}, {int(centroid_y)})")
        
        return regions_with_grid
    
    def crop_region(self, region_info, region_mask):
        """Crop the image for a specific region"""
        x, y, w, h = region_info['bbox']
        
        # Add small padding around the region
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(self.image.shape[1], x + w + padding)
        y_end = min(self.image.shape[0], y + h + padding)
        
        # Just crop the image - no masking or black borders
        cropped_image = self.image[y_start:y_end, x_start:x_end]
        
        return cropped_image
    
    def remove_black_borders(self, masked_image, mask):
        """Remove black borders from masked image by cropping to actual content"""
        # Find where the mask has content (non-zero pixels)
        coords = np.column_stack(np.where(mask > 0))
        
        if len(coords) == 0:
            # No content found, return original
            return masked_image
        
        # Get bounding box of actual content
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add small padding to avoid cutting off edges
        padding = 2
        y_min = max(0, y_min - padding)
        x_min = max(0, x_min - padding)
        y_max = min(masked_image.shape[0] - 1, y_max + padding)
        x_max = min(masked_image.shape[1] - 1, x_max + padding)
        
        # Crop to just the content area
        tight_crop = masked_image[y_min:y_max+1, x_min:x_max+1]
        
        return tight_crop
    
    def resize_and_center_image(self, image):
        """
        Resize and center the image to target size with white background
        
        Args:
            image: Input image (numpy array from cv2)
            
        Returns:
            Resized and centered image with white background
        """
        # Convert cv2 image to PIL for easier handling
        if len(image.shape) == 3:
            # Color image (BGR to RGB conversion)
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            # Grayscale image
            pil_image = Image.fromarray(image)
        
        # Create white background
        new_img = Image.new('RGB', self.target_size, 'white')
        
        # Get original dimensions
        original_width, original_height = pil_image.size
        
        # Calculate scaling factor to fit image within target size while maintaining aspect ratio
        scale_factor = min(self.target_size[0] / original_width, self.target_size[1] / original_height)
        
        # Resize the original image if needed
        if scale_factor < 1:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Calculate position to center the image
        img_width, img_height = pil_image.size
        x = (self.target_size[0] - img_width) // 2
        y = (self.target_size[1] - img_height) // 2
        
        # Convert grayscale to RGB if needed
        if pil_image.mode == 'L':
            pil_image = pil_image.convert('RGB')
        
        # Paste the image onto the white background
        if pil_image.mode == 'RGBA' or 'transparency' in pil_image.info:
            new_img.paste(pil_image, (x, y), pil_image)  # Use alpha channel for transparency
        else:
            new_img.paste(pil_image, (x, y))
        
        # Convert back to cv2 format (RGB to BGR)
        result = cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)
        
        return result
    
    def save_crops(self):
        """
        Extract and save crops for each connected region - clean crops with no borders, resized to target size
        """
        regions, labels = self.find_connected_regions()
        
        if not regions:
            print("No regions found to crop!")
            return
        
        saved_count = 0
        
        for i, region in enumerate(regions):
            try:
                # Create mask for this specific region
                region_mask = np.where(labels == region['label'], 255, 0).astype(np.uint8)
                
                # Crop the region - just clean crop, no masking
                cropped_image = self.crop_region(region, region_mask)
                
                # Conditionally resize if target_size is specified
                if self.target_size:
                    # Resize and center the cropped image with white background
                    final_image = self.resize_and_center_image(cropped_image)
                    size_info = f"resized to {self.target_size[0]}x{self.target_size[1]}"
                else:
                    # Keep original crop size
                    final_image = cropped_image
                    h, w = cropped_image.shape[:2]
                    size_info = f"original size {w}x{h}"
                
                # Generate filename using grid label
                grid_label = region['grid_label']
                area = region['area']
                centroid_x, centroid_y = region['centroid']
                
                # Save the crop
                filename = f"Color_well_{grid_label}.png"
                filepath = os.path.join(self.output_folder, filename)
                cv2.imwrite(filepath, final_image)
                print(f"Saved crop: {filename} ({size_info})")
                saved_count += 1
                
                # Print region info
                x, y, w, h = region['bbox']
                original_size = f"{w}x{h}"
                if self.target_size:
                    print(f"Region {grid_label}: {original_size} -> {self.target_size[0]}x{self.target_size[1]} at ({x},{y}), area={area} pixels")
                else:
                    print(f"Region {grid_label}: {original_size} (kept original) at ({x},{y}), area={area} pixels")
                
            except Exception as e:
                print(f"Error processing region {i+1}: {e}")
                continue
        
        print(f"\nProcessing complete!")
        print(f"Total regions processed: {len(regions)}")
        print(f"Images saved: {saved_count}")
        if self.target_size:
            print(f"All images resized to: {self.target_size[0]}x{self.target_size[1]} with white background")
        else:
            print("All images kept at original crop sizes")
        print(f"Output folder: {self.output_folder}")
    
    def preview_regions(self):
        """Show a preview of detected regions"""
        regions, labels = self.find_connected_regions()
        
        # Create colored preview
        preview = self.image.copy()
        
        # Generate random colors for each region
        colors = []
        for i in range(len(regions)):
            color = (
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            )
            colors.append(color)
        
        # Draw bounding boxes and labels
        for i, region in enumerate(regions):
            x, y, w, h = region['bbox']
            color = colors[i]
            
            # Draw bounding box
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
            
            # Draw grid label
            grid_label = region['grid_label']
            label_text = f"Well {grid_label}"
            cv2.putText(preview, label_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show preview
        window_name = "Region Preview - Press any key to continue"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1200, preview.shape[1]), min(800, preview.shape[0]))
        cv2.imshow(window_name, preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return len(regions)

def main():
    """Main function to run the mask cropper"""
    # Input paths
    input_image_path = r"C:\Users\NoahB\Documents\HebrewU Bioengineering\Equipment\Camera\RPI\Well_Segmentation\Fav_Translucent_Plate.png"
    mask_path = r"C:\Users\NoahB\Documents\HebrewU Bioengineering\Equipment\Camera\RPI\Well_Segmentation\final_adjusted_mask.png"
        
    # Check if files exist
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found: {input_image_path}")
        return
    
    if not os.path.exists(mask_path):
        print(f"Error: Mask not found: {mask_path}")
        print("Please make sure you have created a mask using the interactive_mask_creator.py")
        return
    
    try:
        # Create cropper with target size (can be customized)
        target_size = (512, 512)  # Width x Height - can be changed as needed
        cropper = MaskCropper(input_image_path, mask_path, "wellfile", target_size)
        
        # Show preview of detected regions
        print("Showing preview of detected regions...")
        num_regions = cropper.preview_regions()
        
        if num_regions == 0:
            print("No regions detected in mask!")
            return
        
        # Ask user to proceed
        user_input = input(f"\nFound {num_regions} regions. Proceed with cropping and resizing to {target_size[0]}x{target_size[1]}? (y/n): ")
        if user_input.lower() != 'y':
            print("Cropping cancelled.")
            return
        
        # Save crops
        print("Extracting, resizing and saving crops...")
        cropper.save_crops()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 