#!/usr/bin/env python3
"""
Direct test script for the PyQt6 GUI debug interface - uses GUI file selection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Organoid_Analysis_Quadrant_Debug import WellOrganoidAnalyzer, PYQT_AVAILABLE, ParameterDebugGUI
from PyQt6.QtWidgets import QApplication

def test_gui_direct():
    """Test the PyQt6 GUI debug interface directly"""
    
    if not PYQT_AVAILABLE:
        print("❌ PyQt6 is not available. Please install it with:")
        print("   pip install PyQt6")
        return
    
    print("🧪 Testing PyQt6 GUI Debug Interface (Direct)")
    print("=" * 50)
    
    # Create analyzer with debug mode enabled
    analyzer = WellOrganoidAnalyzer(debug_mode=True)
    
    print("📁 GUI will provide file selection interface...")
    print("   → Select your image file using the GUI buttons")
    print("   → Optionally select a mask file")
    print("   → Click 'Start Analysis' to begin")
    
    # Create QApplication if it doesn't exist
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    
    print("\n🖥️  Launching PyQt6 GUI Debug Interface...")
    print("   → File Selection: Choose image and mask files")
    print("   → Binary Detection: Step through the binary detection pipeline")
    print("   → Color Detection: Complete color sampling and detection")
    print("   → Parameters are automatically saved and loaded")
    
    # Create and show debug GUI directly
    debug_gui = ParameterDebugGUI(analyzer)
    debug_gui.show()
    
    print("✅ GUI launched! Check your screen for the debug interface window.")
    
    # Run the GUI event loop
    app.exec()
    
    print("\n✅ GUI debug test completed!")
    if hasattr(analyzer, 'binary_centroids'):
        print(f"   Binary centroids found: {len(analyzer.binary_centroids)}")
    if hasattr(analyzer, 'color_centroids'):
        print(f"   Color centroids found: {len(analyzer.color_centroids)}")

if __name__ == "__main__":
    test_gui_direct() 