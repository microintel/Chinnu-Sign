"""
Example script to run the GUI application
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("🎨 Starting ISL Detection GUI...")
    print("👋 Place your hand in front of the camera")
    print("📸 Press 'q' to quit")
    
    try:
        from app.gui.advanced_detector import ISLDetectorGUI
        import tkinter as tk
        
        # Create tkinter root window
        root = tk.Tk()
        
        # Initialize and run GUI
        gui = ISLDetectorGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("\n✅ GUI closed successfully")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
