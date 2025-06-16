from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt6.QtGui import QPainter, QPen, QImage, QColor
from PyQt6.QtCore import Qt, QPoint
import sys
import numpy as np
from Wow import load_pretrained_model, predict_image

class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280) 
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        self.drawing = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            pen = QPen(Qt.GlobalColor.black, 20, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            current_point = event.position().toPoint()
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas = QPainter(self)
        canvas.drawImage(self.rect(), self.image, self.image.rect())

    def clear(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def get_image_array(self):
        image = self.image.scaled(28, 28).convertToFormat(QImage.Format.Format_Grayscale8)
        ptr = image.bits()
        ptr.setsize(28 * 28)
        arr = np.frombuffer(ptr, np.uint8).astype(np.float32)
        arr = ((255.0 - arr) - 127.5) / 127.5
        return arr


class DigitRecognizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MNIST Tahmin Paneli")
        self.canvas = Canvas()

        self.predict_button = QPushButton("Tahmin Et")
        self.clear_button = QPushButton("Temizle")
        self.result_label = QLabel("Sonuç: ")
        self.result_label.setStyleSheet("font-size: 24px;")

        self.predict_button.clicked.connect(self.predict)
        self.clear_button.clicked.connect(self.canvas.clear)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        self.model = load_pretrained_model()

    def predict(self):
        arr = self.canvas.get_image_array()
        prediction = predict_image(self.model, arr)
        self.result_label.setText(f"Tahmin Edilen Rakam: {prediction}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigitRecognizer()
    window.show()
    sys.exit(app.exec())