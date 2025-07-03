import cv2
import numpy as np
import os

class InteractiveMaskCreator:
    def __init__(self, template_image_path, existing_mask_path=None):
        # Load the template image at FULL RESOLUTION
        if isinstance(template_image_path, str) and os.path.exists(template_image_path):
            self.template = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
        elif isinstance(template_image_path, np.ndarray):
            # If it's already a numpy array (processed image)
            self.template = template_image_path
        else:
            # Create a default template if file doesn't exist
            self.template = np.zeros((512, 512), dtype=np.uint8)
            print(f"Template image not found, using default 512x512 black image")
        
        self.height, self.width = self.template.shape
        print(f"Loaded image at full resolution: {self.width}x{self.height}")
        
        # Load existing mask or create new one
        if existing_mask_path and os.path.exists(existing_mask_path):
            self.mask = cv2.imread(existing_mask_path, cv2.IMREAD_GRAYSCALE)
            if self.mask is None:
                print(f"Warning: Could not load mask from {existing_mask_path}, starting with blank mask")
                self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
            else:
                # Ensure mask is same size as template
                if self.mask.shape != (self.height, self.width):
                    print(f"Resizing mask from {self.mask.shape} to match template {(self.height, self.width)}")
                    self.mask = cv2.resize(self.mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                
                # Ensure mask is binary (0 or 255)
                self.mask = np.where(self.mask > 127, 255, 0).astype(np.uint8)
                
                white_pixels = np.sum(self.mask == 255)
                print(f"Loaded existing mask with {white_pixels} white pixels")
        else:
            # Create the mask (starts as all black/0's)
            self.mask = np.zeros((self.height, self.width), dtype=np.uint8)
            if existing_mask_path:
                print(f"Warning: Mask file '{existing_mask_path}' not found, starting with blank mask")
        
        # Display image (combination of template and current mask)
        self.display_img = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        
        # Cursor properties
        self.cursor_size = 210
        self.cursor_x = self.width // 2
        self.cursor_y = self.height // 2
        
        # Rotation and movement properties
        self.rotation_angle = 0
        self.offset_x = 0
        self.offset_y = 0
        
        # Window setup with resizable window for full resolution
        self.window_name = "Interactive Mask Creator - Full Resolution"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
        cv2.resizeWindow(self.window_name, min(1200, self.width), min(800, self.height))
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("Controls:")
        print("- Move mouse to position cursor")
        print("- [ ] to decrease/increase cursor size (fine adjustments)")
        print("- Click to add white area to mask")
        print("- WASD to move mask")
        print("- Q/E to rotate mask (1° increments)")
        print("- Enter or close window to finish")
        print("- Use mouse wheel or window resize to navigate large images")
        
        if existing_mask_path:
            print(f"- Editing existing mask: {existing_mask_path}")
        
    def mouse_callback(self, event, x, y, flags, param):
        # Update cursor position
        self.cursor_x = x
        self.cursor_y = y
        
        # Handle mouse click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_mask_area(x, y)
    
    def add_mask_area(self, center_x, center_y):
        """Add a white square area to the mask at the clicked position"""
        half_size = self.cursor_size // 2
        
        # Calculate square bounds
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(self.width, center_x + half_size)
        y2 = min(self.height, center_y + half_size)
        
        # Apply rotation if any
        if self.rotation_angle != 0:
            # Create a temporary mask for the rotated square
            temp_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.rectangle(temp_mask, (x1, y1), (x2, y2), 255, -1)
            
            # Apply rotation
            center = (self.width // 2, self.height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
            temp_mask = cv2.warpAffine(temp_mask, rotation_matrix, (self.width, self.height))
            
            # Add to main mask
            self.mask = cv2.bitwise_or(self.mask, temp_mask)
        else:
            # Simple rectangle without rotation
            cv2.rectangle(self.mask, (x1, y1), (x2, y2), 255, -1)
        
        print(f"Added mask area at ({center_x}, {center_y}) with size {self.cursor_size}")
    
    def update_display(self):
        """Update the display image showing template + mask + cursor"""
        # Start with template as base
        self.display_img = cv2.cvtColor(self.template, cv2.COLOR_GRAY2BGR)
        
        # Apply current mask as colored overlay (red for mask areas)
        mask_colored = np.zeros_like(self.display_img)
        mask_colored[:, :, 2] = self.mask  # Red channel
        
        # Blend template with mask overlay
        self.display_img = cv2.addWeighted(self.display_img, 0.7, mask_colored, 0.3, 0)
        
        # Draw cursor square
        half_size = self.cursor_size // 2
        cursor_x1 = max(0, self.cursor_x - half_size)
        cursor_y1 = max(0, self.cursor_y - half_size)
        cursor_x2 = min(self.width, self.cursor_x + half_size)
        cursor_y2 = min(self.height, self.cursor_y + half_size)
        
        # Draw cursor as green square outline
        cv2.rectangle(self.display_img, (cursor_x1, cursor_y1), (cursor_x2, cursor_y2), (0, 255, 0), 2)
        
        # Draw center crosshair
        cv2.line(self.display_img, (self.cursor_x-5, self.cursor_y), (self.cursor_x+5, self.cursor_y), (0, 255, 0), 1)
        cv2.line(self.display_img, (self.cursor_x, self.cursor_y-5), (self.cursor_x, self.cursor_y+5), (0, 255, 0), 1)
        
        # Show rotation angle and cursor size info
        info_text = f"Size: {self.cursor_size} | Rotation: {self.rotation_angle}°"
        cv2.putText(self.display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def handle_keyboard(self, key):
        """Handle keyboard input"""
        if key == ord('[') and self.cursor_size > 2:  # Changed from ( to [
            self.cursor_size -= 2  # Finer adjustments (2 pixels instead of 5)
            print(f"Cursor size: {self.cursor_size}")
        elif key == ord(']') and self.cursor_size < 300:  # Changed from ) to ]
            self.cursor_size += 2  # Finer adjustments (2 pixels instead of 5)
            print(f"Cursor size: {self.cursor_size}")
        elif key == ord('w'):
            self.offset_y -= 5
        elif key == ord('s'):
            self.offset_y += 5
        elif key == ord('a'):
            self.offset_x -= 5
        elif key == ord('d'):
            self.offset_x += 5
        elif key == ord('q'):
            self.rotation_angle = (self.rotation_angle - 1) % 360  # 1 degree increments
            print(f"Rotation: {self.rotation_angle}°")
        elif key == ord('e'):
            self.rotation_angle = (self.rotation_angle + 1) % 360  # 1 degree increments
            print(f"Rotation: {self.rotation_angle}°")
    
    def run(self):
        """Main loop for the interactive mask creator"""
        while True:
            self.update_display()
            cv2.imshow(self.window_name, self.display_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Exit conditions
            if key == 13 or key == 27:  # Enter or Escape
                break
            elif cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            
            # Handle other keyboard input
            if key != 255:  # Key was pressed
                self.handle_keyboard(key)
        
        cv2.destroyAllWindows()
        return self.mask
    
    def save_mask(self, output_path="mask_output.png"):
        """Save the created mask"""
        cv2.imwrite(output_path, self.mask)
        print(f"Mask saved to: {output_path}")
        return output_path

class MaskAdjuster:
    def __init__(self, original_image, mask):
        self.original_image = original_image
        self.mask = mask.copy()
        self.height, self.width = self.mask.shape
        
        # Selected region properties
        self.selected_region = None
        self.selected_region_coords = None
        self.is_dragging = False
        
        # Window setup
        self.window_name = "Mask Adjustment - Click white areas to move them"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, min(1200, self.width), min(800, self.height))
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n=== MASK ADJUSTMENT MODE ===")
        print("Controls:")
        print("- Click on white mask areas to select them")
        print("- Use ARROW KEYS to move selected area")
        print("- Hold SHIFT + ARROW KEYS for faster movement")
        print("- ESC to deselect current area")
        print("- Enter or close window to finish adjustments")
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.select_mask_region(x, y)
    
    def select_mask_region(self, x, y):
        """Select a connected white region at the clicked position"""
        if self.mask[y, x] == 255:  # Clicked on white area
            # Find connected component at this position
            temp_mask = np.zeros((self.height + 2, self.width + 2), dtype=np.uint8)
            cv2.floodFill(self.mask, temp_mask, (x, y), 128, loDiff=0, upDiff=0)
            
            # Extract the selected region
            self.selected_region = (self.mask == 128).astype(np.uint8) * 255
            
            # Get bounding box of selected region
            coords = np.column_stack(np.where(self.selected_region == 255))
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                self.selected_region_coords = (x_min, y_min, x_max, y_max)
                
                # Restore mask to original state (remove the 128 values)
                self.mask[self.mask == 128] = 255
                
                print(f"Selected region at ({x}, {y}) - Size: {x_max-x_min+1}x{y_max-y_min+1}")
            else:
                self.selected_region = None
                self.selected_region_coords = None
        else:
            # Clicked on black area, deselect
            self.selected_region = None
            self.selected_region_coords = None
            print("No mask area selected")
    
    def move_selected_region(self, dx, dy):
        """Move the selected region by dx, dy pixels"""
        if self.selected_region is None:
            return
        
        # Remove the selected region from current mask
        self.mask = self.mask - self.selected_region
        self.mask = np.clip(self.mask, 0, 255).astype(np.uint8)
        
        # Create moved region
        moved_region = np.zeros_like(self.selected_region)
        
        # Get current region coordinates
        coords = np.column_stack(np.where(self.selected_region == 255))
        
        # Apply movement
        for y, x in coords:
            new_x = x + dx
            new_y = y + dy
            
            # Keep within bounds
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                moved_region[new_y, new_x] = 255
        
        # Add moved region to mask
        self.mask = np.maximum(self.mask, moved_region)
        
        # Update selected region and coordinates
        self.selected_region = moved_region
        if self.selected_region_coords:
            x_min, y_min, x_max, y_max = self.selected_region_coords
            self.selected_region_coords = (x_min + dx, y_min + dy, x_max + dx, y_max + dy)
        
        print(f"Moved region by ({dx}, {dy})")
    
    def update_display(self):
        """Update the display showing masked image (AND operation)"""
        # Convert original to color if it's grayscale
        if len(self.original_image.shape) == 2:
            display_img = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2BGR)
        else:
            display_img = self.original_image.copy()
        
        # Apply mask using AND operation - only show image where mask is white
        mask_3channel = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        mask_normalized = mask_3channel.astype(np.float32) / 255.0
        
        # Apply mask to image (black where mask is 0, original image where mask is 255)
        self.display_img = (display_img.astype(np.float32) * mask_normalized).astype(np.uint8)
        
        # Highlight selected region with green border
        if self.selected_region is not None:
            # Find contours of selected region
            contours, _ = cv2.findContours(self.selected_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.display_img, contours, -1, (0, 255, 0), 3)  # Green border
        
        # Draw bounding box for selected region
        if self.selected_region_coords:
            x_min, y_min, x_max, y_max = self.selected_region_coords
            cv2.rectangle(self.display_img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)  # Yellow box
        
        # Show instructions
        info_text = "Masked Image | Green border: Selected area"
        cv2.putText(self.display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if self.selected_region is not None:
            status_text = "Use ARROW KEYS to move selected area (SHIFT for faster)"
            cv2.putText(self.display_img, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            status_text = "Click on visible areas to select and move them"
            cv2.putText(self.display_img, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def handle_keyboard(self, key):
        """Handle keyboard input for moving selected regions"""
        if self.selected_region is None:
            return
        
        # Determine movement step (1 pixel normal, 5 pixels with faster keys)
        step = 1
        
        dx, dy = 0, 0
        
        # Handle arrow keys and movement keys
        # Using ASCII codes and common arrow key codes
        if key == ord('w') or key == 2490368 or key == 65362:  # Up (w key, up arrow variants)
            dy = -step
            print(f"Moving up by {step}")
        elif key == ord('s') or key == 2621440 or key == 65364:  # Down (s key, down arrow variants)
            dy = step
            print(f"Moving down by {step}")
        elif key == ord('a') or key == 2424832 or key == 65361:  # Left (a key, left arrow variants)
            dx = -step
            print(f"Moving left by {step}")
        elif key == ord('d') or key == 2555904 or key == 65363:  # Right (d key, right arrow variants)
            dx = step
            print(f"Moving right by {step}")
        
        # Handle faster movement with different keys
        elif key == ord('W'):  # Shift+W for faster up
            dy = -5
            print(f"Moving up fast by 5")
        elif key == ord('S'):  # Shift+S for faster down
            dy = 5
            print(f"Moving down fast by 5")
        elif key == ord('A'):  # Shift+A for faster left
            dx = -5
            print(f"Moving left fast by 5")
        elif key == ord('D'):  # Shift+D for faster right
            dx = 5
            print(f"Moving right fast by 5")
        
        if dx != 0 or dy != 0:
            self.move_selected_region(dx, dy)
        elif key != 0xFFFFFF and key != 13 and key != 27:  # Debug: print unknown keys
            print(f"Key pressed: {key} (unknown)")
    
    def run(self):
        """Main loop for mask adjustment"""
        while True:
            self.update_display()
            cv2.imshow(self.window_name, self.display_img)
            
            # Handle keyboard input - use both waitKey and waitKeyEx
            key = cv2.waitKey(1) & 0xFF
            
            # Exit conditions
            if key == 13 or key == 27:  # Enter or Escape
                if key == 27:  # Escape - deselect current region
                    if self.selected_region is not None:
                        self.selected_region = None
                        self.selected_region_coords = None
                        print("Deselected region")
                        continue
                break
            elif cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            
            # Handle keyboard input for movement
            if key != 255:  # Key was pressed (255 = no key)
                self.handle_keyboard(key)
            
            # Also try special keys
            special_key = cv2.waitKeyEx(1)
            if special_key != -1 and special_key != key:
                self.handle_keyboard(special_key)
        
        cv2.destroyAllWindows()
        return self.mask

def edit_mask(template_path, existing_mask_path, output_path=None):
    """
    Edit an existing mask file
    
    Args:
        template_path: Path to the template image
        existing_mask_path: Path to the existing mask to edit
        output_path: Where to save the edited mask (defaults to same as existing_mask_path)
    """
    # Load and preprocess the template image
    template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
    
    if template_img is None:
        print(f"Error: Could not load image from {template_path}")
        return
    
    # Convert to grayscale and binarize
    gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Create the interactive mask creator
    creator = InteractiveMaskCreator(binary, existing_mask_path)
    
    # Run the interactive session
    final_mask = creator.run()
    
    # Save the mask
    if output_path is None:
        output_path = existing_mask_path  # Overwrite existing mask
    
    creator.save_mask(output_path)
    
    print(f"Mask editing complete!")
    print(f"White pixels: {np.sum(final_mask == 255)}")
    print(f"Black pixels: {np.sum(final_mask == 0)}")

def main():
    # Use your template image path
    template_path = r"imx_519_Focus_6.jpg"
    
    # Optional: Load existing mask to edit (set to None for new mask)
    # existing_mask_path = "created_mask.png"  # Uncomment to edit existing mask
    # existing_mask_path = "previous_mask.png"  # Or specify another mask file
    existing_mask_path = None
    # --- Load and preprocess the image ---
    # First load the image
    template_img = cv2.imread(template_path, cv2.IMREAD_COLOR)
    
    if template_img is None:
        print(f"Error: Could not load image from {template_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    
    # Binarize with threshold and OTSU
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    print(f"Preprocessed image to binary. Original size: {template_img.shape[:2]}")
    
    # === PHASE 1: Create/Edit Mask ===
    creator = InteractiveMaskCreator(gray, existing_mask_path)
    final_mask = creator.run()
    
    # Save the mask after creation phase
    output_path = creator.save_mask("created_mask.png")
    
    print(f"Mask creation complete!")
    print(f"White pixels: {np.sum(final_mask == 255)}")
    print(f"Black pixels: {np.sum(final_mask == 0)}")
    
    # === PHASE 2: Adjust Mask ===
    print("\n" + "="*50)
    print("Opening mask adjustment mode...")
    
    # Use original template image (not binary) for better visualization
    adjuster = MaskAdjuster(template_img, final_mask)
    adjusted_mask = adjuster.run()
    
    # Save the final adjusted mask
    cv2.imwrite("IMX519_Mask.png", adjusted_mask)
    print(f"\nFinal adjusted mask saved to: final_adjusted_mask.png")
    print(f"Final white pixels: {np.sum(adjusted_mask == 255)}")
    print(f"Final black pixels: {np.sum(adjusted_mask == 0)}")

if __name__ == "__main__":
    main() 