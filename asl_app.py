import sys
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

# Load the trained model
model = tf.keras.models.load_model("asl_model.h5")

# Define class labels
CLASS_LABELS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
    'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

class ASLApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL_HackHive78")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #FCE79C;")  # **Lighter Yellow Background**

        # 📌 **Load and Position the Ontario Tech Logo (Centered at the TOP)**
        self.logo_label = QLabel(self)
        logo_pixmap = QPixmap("Screenshot 2025-02-09 023619.jpg")  # Load the logo
        self.logo_label.setPixmap(self.make_transparent(logo_pixmap, 0.1))  # **Make logo barely visible (0.1 opacity)**
        self.logo_label.setScaledContents(True)  # Ensure proper scaling
        self.logo_label.setFixedSize(400, 150)  # Adjust size
        self.logo_label.setAlignment(Qt.AlignCenter)
        
        # Position at the top center
        self.logo_label.setGeometry(
            (self.width() - self.logo_label.width()) // 2,  # Center X
            10,  # Fixed at the TOP (10px margin)
            self.logo_label.width(),
            self.logo_label.height()
        )

        # Initialize webcam and MediaPipe
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Video Feed Label
        self.video_label = QLabel(self)

        # Prediction Label
        self.prediction_label = QLabel("Sign 2 Speak ", self)
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 32px; color: black; font-weight: bold;")

        # Start/Stop Buttons
        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_recognition)
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_recognition)
        self.stop_button.setEnabled(False)

        # 📌 **Modify Layout to Keep UI Clean**
        layout = QVBoxLayout()
        layout.addWidget(self.logo_label, alignment=Qt.AlignHCenter)  # **Keep logo at the top center**
        layout.addWidget(self.video_label)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def make_transparent(self, pixmap, opacity):
        """Make the Ontario Tech logo more transparent."""
        transparent_pixmap = QPixmap(pixmap.size())
        transparent_pixmap.fill(Qt.transparent)
        painter = QPainter(transparent_pixmap)
        painter.setOpacity(opacity)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        return transparent_pixmap

    def start_recognition(self):
        """Start the recognition process."""
        self.timer.start(30)  # Update every 30ms
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_recognition(self):
        """Stop the recognition process."""
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Extract hand region
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

                    hand_img = frame[y_min:y_max, x_min:x_max]
                    if hand_img.size == 0:
                        continue

                    # Preprocess and predict
                    hand_img = cv2.resize(hand_img, (224, 224))
                    hand_img = hand_img / 255.0
                    hand_img = np.expand_dims(hand_img, axis=0)
                    prediction = model.predict(hand_img)
                    class_idx = np.argmax(prediction)
                    predicted_label = CLASS_LABELS[class_idx]

                    # Update prediction label
                    self.prediction_label.setText(f"Prediction: {predicted_label}")

                    # Save prediction to file
                    with open("predictions.txt", "a") as f:
                        f.write(f"{predicted_label}\n")

            # Display frame in PyQt
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def resizeEvent(self, event):
        """Ensure the Ontario Tech logo stays centered at the top when resizing the window."""
        self.logo_label.setGeometry(
            (self.width() - self.logo_label.width()) // 2,  # Center X
            10,  # Fixed at the TOP (10px margin)
            self.logo_label.width(),
            self.logo_label.height()
        )

    def closeEvent(self, event):
        # Release resources on close
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLApp()
    window.show()
    sys.exit(app.exec_())
