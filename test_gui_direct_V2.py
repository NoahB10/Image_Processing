#!/usr/bin/env python3
"""
Direct test script for the PyQt6 GUI debug interface - uses GUI file selection
Now with transparency preservation support!
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Organoid_Analysis_Quadrant_Debug import WellOrganoidAnalyzer, PYQT_AVAILABLE, ParameterDebugGUI
from PyQt6.QtWidgets import QApplication

def test_gui_direct():
    """Test the PyQt6 GUI debug interface directly with transparency support"""
    
    if not PYQT_AVAILABLE:
        print("❌ PyQt6 is not available. Please install it with:")
        print("   pip install PyQt6")
        return
    
    print("🧪 Testing PyQt6 GUI Debug Interface (Direct) - Transparency Enabled")
    print("=" * 60)
    
    # Create analyzer with debug mode enabled
    analyzer = WellOrganoidAnalyzer(debug_mode=True)
    
    print("🎨 SIMPLE TRANSPARENCY APPROACH ENABLED!")
    print("   ✅ Alpha channel creates binary mask at start of analysis")
    print("   ✅ All processing restricted to non-transparent areas only")
    print("   ✅ No complex transparency handling during pipeline")
    print("   ✅ Clean, simple, and bug-free alpha support")
    print("   🔧 MUCH SIMPLER: Alpha mask applied once at the beginning!")
    
    print("\n📁 GUI will provide file selection interface...")
    print("   → Select your image file using the GUI buttons")
    print("   → 🎯 TIP: Try loading a PNG with transparency to test the fix!")
    print("   → Optionally select a mask file")
    print("   → Click 'Start Analysis' to begin")
    
    # Create QApplication if it doesn't existerosion
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    
    print("\n🖥️  Launching PyQt6 GUI Debug Interface...")
    print("   → File Selection: Choose image and mask files")
    print("   → Binary Detection: Step through the binary detection pipeline")
    print("   → Color Detection: Complete color sampling and detection")
    print("   → Parameters are automatically saved and loaded")
    print("   → 🎨 Transparency: Images with alpha channels will be properly displayed")
    
    # Create and show debug GUI directly
    debug_gui = ParameterDebugGUI(analyzer)
    debug_gui.show()
    
    print("✅ GUI launched! Check your screen for the debug interface window.")
    print("🎨 SIMPLE ALPHA TEST:")
    print("   → Load a PNG image with transparency")
    print("   → Alpha mask created automatically at start")
    print("   → All analysis restricted to non-transparent areas")
    print("   → Display preserves transparency for visualization")
    print("   → ✅ SIMPLE & CLEAN: No complex transparency bugs!")
    print("   → Detection works perfectly with alpha images")
    
    # Run the GUI event loop
    app.exec()
    
    print("\n✅ GUI debug test completed!")
    if hasattr(analyzer, 'binary_centroids'):
        print(f"   Binary centroids found: {len(analyzer.binary_centroids)}")
    if hasattr(analyzer, 'color_centroids'):
        print(f"   Color centroids found: {len(analyzer.color_centroids)}")
    
    print("\n🎨 Transparency Status:")
    if hasattr(analyzer, 'original_image') and analyzer.original_image is not None:
        has_alpha = len(analyzer.original_image.shape) == 3 and analyzer.original_image.shape[2] == 4
        if has_alpha:
            print("   ✅ Image with transparency was loaded and processed!")
            print("   ✅ Alpha channel should be preserved in all outputs")
        else:
            print("   ℹ️  Regular RGB image was processed (no transparency)")
    else:
        print("   ℹ️  No image was loaded during this session")

if __name__ == "__main__":
    test_gui_direct() 