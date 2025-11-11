from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

class LoadingDialog(QDialog):
    def __init__(self, message="Processing on HPC... please wait"):
        super().__init__()
        self.setWindowTitle("Running Job")
        self.setModal(True)
        self.setFixedSize(250, 150)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setFamilies([u"Inter"])
        font.setPointSize(10)
        label.setFont(font)
        layout.addWidget(label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # infinite loading
        layout.addWidget(self.progress)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False) # hide close button
