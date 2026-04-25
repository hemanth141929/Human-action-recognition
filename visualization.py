import torch
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch.nn as nn
from torchvision import transforms, models

# --- CONFIGURATION ---
MODEL_PATH = "action_model_ep5.pth" # Your saved model
CLASSES = ['ApplyEyeMakeup', 'Archery', 'Basketball', 'Biking', 'Bowling']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TEST_VIDEO = "./data/UCF-101/Biking/v_Biking_g01_c01.avi" 
TEST_VIDEO = 0 # Set to 0 for Webcam, or a path to a video file

# --- 1. LOAD MODELS ---
# A. Load Spatial Model (YOLOv8) for Human Tracking
print("Loading YOLO tracking model...")
yolo_model = YOLO('yolov8n.pt') # Uses Nano version for maximum speed

# B. Load Your Temporal Model (R3D_18)
print("Loading action recognition model...")
action_model = models.video.r3d_18()
action_model.fc = nn.Linear(action_model.fc.in_features, len(CLASSES))
action_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
action_model = action_model.to(DEVICE).eval()

# --- 2. TRANSFORMS & UTILS ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Queue to store the last 16 transformed frames for each subject
frame_buffer = deque(maxlen=16) 

def main():
    cap = cv2.VideoCapture(TEST_VIDEO)
    
    current_prediction = "Waiting..."
    confidence = 0
    
    print("Starting visualization. Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 3. SPATIAL TRACKING (YOLO)
        # track=True enables multi-object tracking
        # classes=0 limits detection to Humans
        results = yolo_model.track(frame, persist=True, classes=0, verbose=False)

        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy() # Bounding boxes
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            # Loop through detected humans
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box.astype(int)

                # CROP & PREPROCESS FRAME FOR ACTION MODEL
                # We crop the individual subject for better temporal analysis
                try:
                    subject_crop = frame[y1:y2, x1:x2]
                    transformed_frame = transform(subject_crop)
                    frame_buffer.append(transformed_frame)
                except:
                    # Occasional crop errors if subject is at the edge
                    continue

                # 4. TEMPORAL CLASSIFICATION (R3D_18)
                # Only classify if we have a full buffer of 16 frames
                if len(frame_buffer) == 16:
                    # Shape needed: [1, Channels, Frames, H, W]
                    input_tensor = torch.stack(list(frame_buffer)).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = action_model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        conf, predicted = torch.max(probs, 1)
                        
                    current_prediction = CLASSES[predicted.item()]
                    confidence = conf.item() * 100

                # 5. DRAWING & VISUALIZATION
                # Draw green bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create label text (Action + Confidence)
                label = f"ID:{track_id} {current_prediction} ({confidence:.1f}%)"
                
                # Draw black background for label
                cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 0, 0), -1)
                
                # Write label text above subject
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow("Human Action Recognition Tracking", frame)
        
        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()