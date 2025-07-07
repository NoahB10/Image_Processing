#!/usr/bin/env python3
"""
Direct launcher for the Organoid Analyzer PyQt6 GUI
Opens immediately without requiring image loading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Organoid_Analysis_Quadrant import WellOrganoidAnalyzer, PYQT_AVAILABLE, ParameterDebugGUI
from PyQt6.QtWidgets import QApplication

def launch_gui():
    """Launch the Organoid Analyzer GUI directly"""
    
    if not PYQT_AVAILABLE:
        print("❌ PyQt6 is not available. Please install it with:")
        print("   pip install PyQt6")
        return
    
    print("🚀 Launching Organoid Analyzer GUI...")
    print("=" * 50)
    
    # Create analyzer (no images loaded yet)
    analyzer = WellOrganoidAnalyzer(debug_mode=True)
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    print("🖥️  Opening GUI window...")
    print("   → Click 'Load Images' to select your images")
    print("   → Use step buttons to navigate binary detection stages")
    print("   → Adjust parameters with sliders for real-time updates")
    print("   → Click 'Sample Colors' for color detection")
    print("   → Click 'Continue Analysis' when done")
    
    # Create and show GUI
    gui = ParameterDebugGUI(analyzer)
    
    print("✅ GUI should be visible now!")
    print("   Window title: 'Organoid Analyzer - Parameter Debug Interface'")
    
    # Run the application
    app.exec()
    
    print("👋 GUI closed")

if __name__ == "__main__":
    launch_gui() 