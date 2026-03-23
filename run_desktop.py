import sys
from PySide6.QtWidgets import QApplication
from src.ui import MainWindow

def main():
    app = QApplication(sys.argv)

    xyz_path = "data/XYZ.xlsx"
    cmyk_path = "data/cmyk-10nm.xlsx"
    pink_path = "Dataset/pink-interpolated.xlsx"

    win = MainWindow(xyz_path=xyz_path, cmyk_path=cmyk_path, pink_path=pink_path)
    win.resize(1200, 750)
    win.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
