import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import mediapipe as mp
import numpy as np
import time  # Add this line to import the time module
from preProcessor import redistribute_values, normalize_coordinates

# Load the trained model
model = torch.load("torch_model.pt")
model.eval()

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Open default camera
cap = cv2.VideoCapture(0)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Initialize variables for gesture detection
arr = []
frame_count = 0
min_length = 5

# Get the labels for gesture detection
with open("label_to_int.txt") as f:
    label_to_int = dict(json.load(f))

labels = {}
for lb,i in label_to_int.items():
    labels[i] = lb
    
is_found = False
frames = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        if not is_found:
            start_time = time.time_ns()
        is_found = True
        frames +=1
        # Extract hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]

        # Convert landmarks to list of tuples
        landmark_points = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) for l in hand_landmarks.landmark]

        # Convert landmarks to numpy array of type int32
        landmark_points_np = np.array(landmark_points, dtype=np.int32)

        # Draw bounding box around hand
        brect = cv2.boundingRect(cv2.convexHull(landmark_points_np))
        cv2.rectangle(frame, (brect[0], brect[1]), (brect[0] + brect[2], brect[1] + brect[3]), (0, 255, 0), 2)

        # Calculate center of bounding box
        center_x = brect[0] + brect[2] // 2
        center_y = brect[1] + brect[3] // 2
        arr.append([float(center_x), float(center_y)])

        # Display the coordinates of the center
        # cv2.putText(frame, f'Center: ({center_x}, {center_y})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Prepare the data for inference
    elif len(arr) >= min_length:
        print(frames*1e9/(time.time_ns()-start_time), "fps")
        is_found = False
        frames = 0
        data = np.array(arr)
        arr = []
        data = normalize_coordinates(data, frame_width, frame_height)
        data = redistribute_values(data, points=24)
        data = np.expand_dims(data, axis=0)
        data = torch.tensor(data, dtype=torch.float)
        
        # Predict the gesture
        prediction = model(data)[0]
        print(prediction)
        gesture_label = labels[np.argmax(prediction.detach().numpy())]
        
        # Display the predicted gesture
        cv2.putText(frame, f'Gesture: {gesture_label}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Hand Tracking', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
