import torch
import cv2
import os
import torch.nn as nn
from torchvision import transforms, models

# --- CONFIGURATION ---
MODEL_PATH = "best_crime_model.pth" # Use the 'best' model from your training
TEST_VIDEO = "./test_videos/test7.mp4" 

# MUST match the folders in your data directory exactly
CLASSES = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
           'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting', 
           'Shoplifting', 'Stealing', 'Vandalism']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. LOAD MODEL ---
model = models.video.r2plus1d_18() # or r3d_18, whichever you used to train!
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
# Load the weights
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
except RuntimeError as e:
    print(f"ERROR: Model mismatch! Check architecture and class count.\n{e}")
    exit()

model = model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frames = []
    results = []

    print(f"Analyzing: {os.path.basename(video_path)}...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Preprocess and add to buffer
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(rgb_frame))

        # Once we have 16 frames, run inference
        if len(frames) == 16:
            input_tensor = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, predicted = torch.max(probs, 1)
                
                results.append((predicted.item(), conf.item()))
            
            # Clear buffer to get the NEXT 16 frames
            frames = []

    cap.release()

    if not results:
        print("Video too short for analysis.")
        return

    # --- AGGREGATION LOGIC ---
    # Find the most frequent prediction across all segments
    all_preds = [r[0] for r in results]
    final_idx = max(set(all_preds), key=all_preds.count)
    
    # Calculate average confidence for that prediction
    avg_conf = sum([r[1] for r in results if r[0] == final_idx]) / all_preds.count(final_idx)

    print("-" * 30)
    print(f"PREDICTION: {CLASSES[final_idx]}")
    print(f"AVERAGE CONFIDENCE: {avg_conf * 100:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    predict_video(TEST_VIDEO)