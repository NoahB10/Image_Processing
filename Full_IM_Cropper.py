import cv2
import numpy as np
import os
import json

class CropperGUI:
    def __init__(self, image_path, crop_size=(200, 200), output_folder="manual_crops", config_path="crop_config.json"):
        self.image_path = image_path
        self.crop_width, self.crop_height = crop_size
        self.output_folder = output_folder
        self.config_path = config_path
        os.makedirs(output_folder, exist_ok=True)

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")

        self.clone = self.image.copy()
        self.dragging = False

        # Load saved config or set default
        self.rect_start = self.load_or_default_config()
        self.window_name = "Cropper"
        self.run()

    def draw_rectangle(self, img):
        x, y = self.rect_start
        end_x = min(x + self.crop_width, img.shape[1])
        end_y = min(y + self.crop_height, img.shape[0])
        cv2.rectangle(img, (x, y), (end_x, end_y), (0, 255, 0), 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.offset_x = x - self.rect_start[0]
            self.offset_y = y - self.rect_start[1]
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            new_x = x - self.offset_x
            new_y = y - self.offset_y
            new_x = min(max(0, new_x), self.image.shape[1] - self.crop_width)
            new_y = min(max(0, new_y), self.image.shape[0] - self.crop_height)
            self.rect_start = (new_x, new_y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def save_crop(self):
        x, y = self.rect_start
        crop = self.image[y:y + self.crop_height, x:x + self.crop_width]
        filename = os.path.join(self.output_folder, f"crop_{x}_{y}.png")
        cv2.imwrite(filename, crop)
        print(f"✅ Saved crop at ({x}, {y}) to {filename}")

        # Save config
        self.save_config()

    def save_config(self):
        config = {
            "x": self.rect_start[0],
            "y": self.rect_start[1],
            "crop_width": self.crop_width,
            "crop_height": self.crop_height
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"📝 Saved crop config to {self.config_path}")

    def load_or_default_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            print(f"📂 Loaded saved crop config from {self.config_path}")
            self.crop_width = config.get("crop_width", self.crop_width)
            self.crop_height = config.get("crop_height", self.crop_height)
            return (config.get("x", 50), config.get("y", 50))
        else:
            print("🆕 No saved config found. Using default position (50, 50).")
            return (50, 50)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("🔧 Instructions:")
        print(" • Drag the rectangle with your mouse")
        print(" • Press 's' to save the crop and coordinates")
        print(" • Press 'q' to quit")

        while True:
            display = self.clone.copy()
            self.draw_rectangle(display)
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                print("👋 Exiting cropper.")
                break
            elif key == ord('s'):
                self.save_crop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = r"C:\Users\NoahB\Downloads\IMX519_Focus7.jpg" # Replace with your actual image
    crop_size = (2300, 1780)  # Adjust crop size as needed
    CropperGUI(image_path, crop_size)
