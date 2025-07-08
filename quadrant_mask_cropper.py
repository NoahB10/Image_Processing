import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image

# This script is used to crop wells from a specific quadrant/section of the image based on the mask.
# It allows custom row and column ranges to be specified.

class QuadrantMaskCropper:
    def __init__(self, input_image_path, mask_path, row_range, col_range, output_folder="Quadrant_Crops", target_size=(512, 512)):
        """
        Initialize the QuadrantMaskCropper for processing specific quadrants
        
        Args:
            input_image_path: Path to the original image
            mask_path: Path to the binary mask
            row_range: List of row labels (e.g., ['m', 'l', 'k', 'j', 'i', 'h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'])
            col_range: List of column numbers (e.g., [10, 11] + list(range(12, 25)))
            output_folder: Folder to save cropped images
            target_size: Target size for resized images (width, height)
        """
        self.input_image_path = input_image_path
        self.mask_path = mask_path
        self.output_folder = output_folder
        self.target_size = target_size
        
        # Parse row range
        if isinstance(row_range, str):
            if '-' in row_range:
                start_char, end_char = row_range.split('-')
                start_ord = ord(start_char.lower())
                end_ord = ord(end_char.lower())
                if start_ord > end_ord:  # Reverse order like m-a
                    self.row_range = [chr(i) for i in range(start_ord, end_ord-1, -1)]
                else:  # Forward order like a-m
                    self.row_range = [chr(i) for i in range(start_ord, end_ord+1)]
            else:
                self.row_range = list(row_range.lower())
        else:
            self.row_range = [char.lower() for char in row_range]
        
        # Parse column range
        if isinstance(col_range, str):
            self.col_range = []
            parts = col_range.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    start_col, end_col = map(int, part.split('-'))
                    self.col_range.extend(range(start_col, end_col + 1))
                else:
                    self.col_range.append(int(part))
        else:
            self.col_range = col_range
        
        # Create output folder
        Path(self.output_folder).mkdir(exist_ok=True)
        print(f"Output folder created/verified: {self.output_folder}")
        print(f"Row range: {self.row_range}")
        print(f"Column range: {self.col_range}")
        if self.target_size:
            print(f"Target size for cropped images: {target_size[0]}x{target_size[1]}")
        else:
            print("Keeping original crop sizes (no resizing)")
        
        # Load images
        self.load_images()
    
    def load_images(self):
        """Load the input image and mask"""
        self.image = cv2.imread(self.input_image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from: {self.input_image_path}")
    
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
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            self.mask, connectivity=8, ltype=cv2.CV_32S
        )
        
        regions = []
        
        # Skip label 0 (background)
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            
            # Skip very small regions (noise)
            if area < 10:
                continue
            
            region_info = {
                'label': label,
                'bbox': (x, y, w, h),
                'area': area,
                'centroid': centroids[label]
            }
            
            regions.append(region_info)
            
        print(f"Found {len(regions)} connected regions")
        
        # Assign grid positions for the specified quadrant
        regions_with_grid = self.assign_quadrant_grid_positions(regions)
        
        return regions_with_grid, labels
    
    def assign_quadrant_grid_positions(self, regions):
        """Assign grid positions based on the specified row and column ranges"""
        if not regions:
            return regions
        
        # Group regions into columns based on X coordinate
        x_tolerance = 100
        columns = []
        
        regions_by_x = sorted(regions, key=lambda r: r['centroid'][0])
        
        for region in regions_by_x:
            region_x = region['centroid'][0]
            
            assigned_to_column = False
            for column in columns:
                column_avg_x = sum(r['centroid'][0] for r in column) / len(column)
                
                if abs(region_x - column_avg_x) <= x_tolerance:
                    column.append(region)
                    assigned_to_column = True
                    break
            
            if not assigned_to_column:
                columns.append([region])
        
        # Sort each column by Y coordinate (top to bottom)
        for column in columns:
            column.sort(key=lambda r: r['centroid'][1])
        
        # Map detected columns/rows to specified ranges
        regions_with_grid = []
        
        for col_idx, column in enumerate(columns):
            if col_idx >= len(self.col_range):
                break
                
            col_number = self.col_range[col_idx]
            
            for row_idx, region in enumerate(column):
                if row_idx >= len(self.row_range):
                    break
                    
                row_letter = self.row_range[row_idx]
                grid_label = f"{row_letter.upper()}{col_number}"
                
                region['grid_label'] = grid_label
                region['grid_row'] = row_idx
                region['grid_col'] = col_idx
                
                regions_with_grid.append(region)
        
        regions_with_grid.sort(key=lambda r: (r['grid_col'], r['grid_row']))
        
        print(f"Quadrant layout: {len(self.row_range)} rows x {len(self.col_range)} columns specified")
        print("Grid assignments for quadrant:")
        for region in regions_with_grid:
            centroid_x, centroid_y = region['centroid']
            print(f"  {region['grid_label']}: center at ({int(centroid_x)}, {int(centroid_y)})")
        
        return regions_with_grid
    
    def crop_region(self, region_info):
        """Crop the image for a specific region"""
        x, y, w, h = region_info['bbox']
        
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(self.image.shape[1], x + w + padding)
        y_end = min(self.image.shape[0], y + h + padding)
        
        cropped_image = self.image[y_start:y_end, x_start:x_end]
        return cropped_image
    
    def resize_and_center_image(self, image):
        """Resize and center the image to target size with white background"""
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        new_img = Image.new('RGB', self.target_size, 'white')
        
        original_width, original_height = pil_image.size
        scale_factor = min(self.target_size[0] / original_width, self.target_size[1] / original_height)
        
        if scale_factor < 1:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        img_width, img_height = pil_image.size
        x = (self.target_size[0] - img_width) // 2
        y = (self.target_size[1] - img_height) // 2
        
        if pil_image.mode == 'L':
            pil_image = pil_image.convert('RGB')
        
        new_img.paste(pil_image, (x, y))
        result = cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)
        
        return result
    
    def save_crops(self):
        """Extract and save crops for each region in the specified quadrant"""
        regions, labels = self.find_connected_regions()
        
        if not regions:
            print("No regions found to crop in the specified quadrant!")
            return
        
        saved_count = 0
        
        for region in regions:
            try:
                cropped_image = self.crop_region(region)
                
                if self.target_size:
                    final_image = self.resize_and_center_image(cropped_image)
                    size_info = f"resized to {self.target_size[0]}x{self.target_size[1]}"
                else:
                    final_image = cropped_image
                    h, w = cropped_image.shape[:2]
                    size_info = f"original size {w}x{h}"
                
                grid_label = region['grid_label']
                filename = f"Quadrant_well_{grid_label}.png"
                filepath = os.path.join(self.output_folder, filename)
                cv2.imwrite(filepath, final_image)
                print(f"Saved crop: {filename} ({size_info})")
                saved_count += 1
                
            except Exception as e:
                print(f"Error processing region: {e}")
                continue
        
        print(f"\nProcessing complete!")
        print(f"Total regions processed in quadrant: {len(regions)}")
        print(f"Images saved: {saved_count}")
        print(f"Output folder: {self.output_folder}")
    
    def preview_regions(self):
        """Show a preview of detected regions in the quadrant"""
        regions, labels = self.find_connected_regions()
        
        preview = self.image.copy()
        
        for i, region in enumerate(regions):
            x, y, w, h = region['bbox']
            color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
            
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
            
            grid_label = region['grid_label']
            label_text = f"Well {grid_label}"
            cv2.putText(preview, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        window_name = "Quadrant Region Preview - Press any key to continue"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1200, preview.shape[1]), min(800, preview.shape[0]))
        cv2.imshow(window_name, preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return len(regions)

def main():
    """Main function to run the quadrant mask cropper"""
    input_image_path = r"imx_519_Focus_6.jpg"
    mask_path = r"IMX519_Mask.png"
    
    # Define quadrant ranges as per your example
    # Rows: m, l, k, j, i, h, g, f, e, d, c, b, a (reverse alphabetical)
    row_range = ['m', 'l', 'k', 'j', 'i', 'h', 'g', 'f', 'e', 'd', 'c', 'b', 'a']
    
    # Columns: 10, 11, 12-24
    col_range = [10, 11] + list(range(12, 25))  # 12-24 inclusive
    
    # Check if files exist
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found: {input_image_path}")
        return
    
    if not os.path.exists(mask_path):
        print(f"Error: Mask not found: {mask_path}")
        print("Please make sure you have created a mask using the interactive_mask_creator.py")
        return
    
    try:
        target_size = (512, 512)
        cropper = QuadrantMaskCropper(
            input_image_path, 
            mask_path, 
            row_range, 
            col_range, 
            "Quadrant_Crops", 
            target_size
        )
        
        print("Showing preview of detected regions in quadrant...")
        num_regions = cropper.preview_regions()
        
        if num_regions == 0:
            print("No regions detected in the specified quadrant!")
            return
        
        user_input = input(f"\nFound {num_regions} regions in quadrant. Proceed with cropping and resizing to {target_size[0]}x{target_size[1]}? (y/n): ")
        if user_input.lower() != 'y':
            print("Cropping cancelled.")
            return
        
        print("Extracting and saving quadrant crops...")
        cropper.save_crops()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 