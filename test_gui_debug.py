#!/usr/bin/env python3
"""
Test script for the PyQt6 GUI debug interface
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Organoid_Analysis_Quadrant import WellOrganoidAnalyzer, PYQT_AVAILABLE

def test_gui_debug():
    """Test the PyQt6 GUI debug interface"""
    
    if not PYQT_AVAILABLE:
        print("❌ PyQt6 is not available. Please install it with:")
        print("   pip install PyQt6")
        return
    
    print("🧪 Testing PyQt6 GUI Debug Interface")
    print("=" * 50)
    
    # Create analyzer with debug mode enabled
    analyzer = WellOrganoidAnalyzer(debug_mode=True)
    
    # Load images (this will prompt for file selection)
    if not analyzer.load_images():
        print("❌ Failed to load images")
        return
    
    print("✅ Images loaded successfully")
    print(f"   Image size: {analyzer.width} x {analyzer.height}")
    
    # Run the dual detection with GUI debug
    print("\n🖥️  Starting GUI debug interface...")
    print("   - Adjust parameters using sliders")
    print("   - See real-time preview updates")
    print("   - Click 'Continue Analysis' when satisfied")
    
    analyzer.run_dual_detection()
    
    print("\n✅ GUI debug test completed!")
    print(f"   Binary centroids found: {len(analyzer.binary_centroids)}")
    print(f"   Color centroids found: {len(analyzer.color_centroids)}")
    print(f"   Total detections: {len(analyzer.binary_centroids) + len(analyzer.color_centroids)}")

if __name__ == "__main__":
    test_gui_debug() 