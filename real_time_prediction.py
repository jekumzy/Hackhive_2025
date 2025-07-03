import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model("asl_model.h5")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define class labels (adjust to match your dataset)
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
    'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand region
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1])
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # Preprocess hand image for prediction
            hand_img = cv2.resize(hand_img, (224, 224))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predict gesture
            prediction = model.predict(hand_img)
            class_idx = np.argmax(prediction)
            predicted_label = class_labels[class_idx]

            # Display prediction
            cv2.putText(frame, predicted_label, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('ASL Gesture Recognition', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()