from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout
from PyQt6.QtGui import QPainter, QPen, QImage, QColor
from PyQt6.QtCore import Qt, QPoint
import sys
import numpy as np
from Wow import load_pretrained_model, predict_image
from Wow import save_custom_example
from scipy.ndimage import center_of_mass, shift
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(__file__)

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
        arr = 255.0 - arr  
        arr = (arr - 127.5) / 127.5  
        arr = arr.reshape(1, 784)
        return arr
    
def center_image(image_array):
    img = image_array.reshape(28, 28)
    cy, cx = center_of_mass(img)
    shift_y, shift_x = 14 - cy, 14 - cx
    shifted = shift(img, shift=(shift_y, shift_x), cval=0.0)
    return shifted.reshape(1, 784)

class DigitRecognizer(QWidget):
    def save_example(self):
        label_text = self.label_input.text()
        if not label_text.isdigit() or not (0 <= int(label_text) <= 9):
            self.result_label.setText("Geçerli bir sayı gir (0-9)")
            return

        image_array = center_image(self.canvas.get_image_array())
        save_custom_example(image_array, int(label_text))

        index = len(os.listdir(os.path.join(base_dir, "custom_data"))) // 2
        self.result_label.setText(f"{label_text} olarak veri kaydedildi. Toplam veri: {index}")
        self.canvas.clear()
        self.label_input.clear()
    
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

        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Doğru rakamı yaz (0-9)")
        self.save_button = QPushButton("Veri Setine Kaydet")
        self.save_button.clicked.connect(self.save_example)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.label_input)
        input_layout.addWidget(self.save_button)

        layout.addLayout(input_layout)

        self.setLayout(layout)

        self.model = load_pretrained_model()

    def predict(self):
        arr = self.canvas.get_image_array()
        arr = center_image(arr)  
        debug_image(arr)         

        prediction = predict_image(self.model, arr)
        self.result_label.setText(f"Tahmin Edilen Rakam: {prediction}")
        
def debug_image(arr):
    plt.imshow(arr.reshape(28, 28), cmap='gray')
    plt.title("GUI'den alınan input")
    plt.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigitRecognizer()
    window.show()
    sys.exit(app.exec())