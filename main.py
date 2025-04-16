import sys
from PyQt5.QtWidgets import QApplication
from modern_gui import ModernGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ModernGUI()
    window.show()
    sys.exit(app.exec_()) 