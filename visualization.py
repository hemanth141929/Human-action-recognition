import torch
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch.nn as nn
from torchvision import transforms, models
import os

# --- CONFIGURATION ---
# Ensure this is the 'best_crime_model.pth' from your 14-class training
MODEL_PATH = "best_crime_model.pth" 

# MUST match the folders in your data directory exactly (Alphabetical)
CLASSES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
    'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting', 
    'Shoplifting', 'Stealing', 'Vandalism'
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_VIDEO = "./test_videos/test7.mp4" # Set to 0 for Webcam

# --- 1. LOAD MODELS ---
print(f"Loading models on {DEVICE}...")
yolo_model = YOLO('yolov8n.pt') 

# CHANGE THIS: Use r2plus1d_18 to match your checkpoint!
action_model = models.video.r2plus1d_18() 
action_model.fc = nn.Linear(action_model.fc.in_features, len(CLASSES))

try:
    # Use the correct path to your best model
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    action_model.load_state_dict(state_dict)
    print("Action model (R2+1D) loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nHELP: If this still fails, check if your CLASSES list has exactly 14 items.")
    exit()

action_model = action_model.to(DEVICE).eval()

# --- 2. TRANSFORMS ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Use a dictionary to keep separate buffers for different tracked people
track_buffers = {}

def main():
    cap = cv2.VideoCapture(TEST_VIDEO)
    
    # Global display states
    display_prediction = "Scanning..."
    display_confidence = 0
    
    print("Starting Surveillance Mode. Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 3. SPATIAL TRACKING (YOLO)
        results = yolo_model.track(frame, persist=True, classes=0, verbose=False)

        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box.astype(int)
                current_box_color = (0, 255, 0) # Default Green

                # Initialize buffer for new tracks
                if track_id not in track_buffers:
                    track_buffers[track_id] = deque(maxlen=16)

                # CROP & PREPROCESS
                try:
                    subject_crop = frame[y1:y2, x1:x2]
                    if subject_crop.size == 0: continue
                    transformed_frame = transform(subject_crop)
                    track_buffers[track_id].append(transformed_frame)
                except:
                    continue

                # 4. TEMPORAL CLASSIFICATION
                if len(track_buffers[track_id]) == 16:
                    input_tensor = torch.stack(list(track_buffers[track_id])).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = action_model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        conf, predicted = torch.max(probs, 1)
                        
                        raw_pred = CLASSES[predicted.item()]
                        raw_conf = conf.item() * 100

                    # Decision Logic: 60% threshold for Alert
                    if raw_pred != "Normal" and raw_conf > 60.0:
                        display_prediction = f"ALERT: {raw_pred}"
                        display_confidence = raw_conf
                        current_box_color = (0, 0, 255) # Red
                        # Screen Flash
                        cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 10)
                    else:
                        display_prediction = "CLEAR"
                        display_confidence = raw_conf if raw_pred == "Normal" else (100 - raw_conf)
                        current_box_color = (0, 255, 0) # Green

                # 5. DRAWING
                cv2.rectangle(frame, (x1, y1), (x2, y2), current_box_color, 2)
                label_text = f"ID:{track_id} {display_prediction} ({display_confidence:.1f}%)"
                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Crime Detection Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()