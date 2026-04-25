import os
import cv2
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# --- CONFIGURATION ---
VIDEO_DIR = './data/UCF-101/'
SPLIT_FILE = './data/trainlist01.txt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['ApplyEyeMakeup', 'Archery', 'Basketball', 'Biking', 'Bowling'] # Change or add as needed

# --- 1. DATASET CLASS ---
class UCFDataset(Dataset):
    def __init__(self, df, root, frames=16):
        self.df = df
        self.root = root
        self.frames = frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < 300: # Limit search to first 300 frames for speed
            ret, frame = cap.read()
            if not ret: break
            frames.append(self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames

    def __getitem__(self, idx):
        v_path = os.path.join(self.root, self.df.iloc[idx, 0])
        # Use 'label_id' which is the 3rd column (index 2) in our mini_df
        label = self.df.iloc[idx, 2] 

        all_frames = self._load_video(v_path)
        
        if not all_frames:
            return torch.zeros((3, self.frames, 112, 112)), torch.tensor(label)

        indices = torch.linspace(0, len(all_frames)-1, self.frames).long()
        sampled = [all_frames[i] for i in indices]
        
        # Ensure label is returned as a long tensor for the loss function
        return torch.stack(sampled).permute(1, 0, 2, 3), torch.tensor(label).long()

# --- 2. PREPARE DATA ---
print("Filtering classes and preparing data...")
raw_df = pd.read_csv(SPLIT_FILE, sep=' ', header=None, names=['path', 'label'])

# Update filtering logic to be exact
def get_class_name(path):
    return path.split('/')[0]

# Only keep rows where the class is EXACTLY in our list
mini_df = raw_df[raw_df['path'].apply(get_class_name).isin(CLASSES)].copy()
mini_df['label_id'] = mini_df['path'].apply(lambda x: CLASSES.index(x.split('/')[0]))

print(f"Data ready. Found {len(mini_df)} videos across {len(CLASSES)} classes.")

# Safety check: if mini_df is empty, the path or class names are wrong
if len(mini_df) == 0:
    print("ERROR: No videos found! Check if your CLASSES list matches the folder names in data/UCF-101/")
    exit()

train_loader = DataLoader(UCFDataset(mini_df, VIDEO_DIR), batch_size=4, shuffle=True, num_workers=0)

# --- 3. MODEL SETUP ---
print(f"Loading R3D_18 model on {DEVICE}...")
model = models.video.r3d_18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model = model.to(DEVICE)# Move to GPU & enable half-precision for 2x speed

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5) # Use 1e-5 to avoid 'nan'
criterion = nn.CrossEntropyLoss()

# --- 4. TRAINING LOOP ---
print("Starting Training...")
for epoch in range(10): # Try 10 epochs for better accuracy
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE) # No .half()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) # Now criterion is defined!
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        
        if i % 5 == 0:
            print(f"Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100*correct/total:.2f}%")

    print(f"--- Epoch {epoch+1} Finished | Avg Acc: {100*correct/total:.2f}% ---")
    torch.save(model.state_dict(), f"action_model_ep{epoch+1}.pth")

print("Project Built Successfully. Model saved locally.")