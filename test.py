import torch
import cv2
import os
import torch.nn as nn
from torchvision import transforms, models

# --- CONFIGURATION ---
MODEL_PATH = "action_model_ep5.pth" # Use your best epoch file
TEST_VIDEO = "./data/UCF-101/Biking/v_Biking_g01_c01.avi" # Path to any video
CLASSES = ['ApplyEyeMakeup', 'Archery', 'Basketball', 'Biking', 'Bowling']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. LOAD MODEL ARCHITECTURE ---
model = models.video.r3d_18()
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval() # Set to evaluation mode

# --- 2. PREPROCESSING TRANSFORMS ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Read frames
    while len(frames) < 16: # We need 16 frames for the model
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))
    cap.release()

    if len(frames) < 16:
        print("Video too short!")
        return

    # Prepare tensor: [Batch, Channels, Frames, Height, Width]
    input_tensor = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)
        
    class_idx = predicted.item()
    prob = confidence[0][class_idx].item() * 100
    
    print("-" * 30)
    print(f"VIDEO: {os.path.basename(video_path)}")
    print(f"PREDICTION: {CLASSES[class_idx]}")
    print(f"CONFIDENCE: {prob:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    predict_video(TEST_VIDEO)