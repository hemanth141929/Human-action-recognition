import os
import cv2
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
VIDEO_DIR = './data/UCF_Crimes/' 
SPLIT_FILE = './data/Anomaly_Train.txt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  
EPOCHS = 20

# --- 1. DATASET CLASS ---
class UCFDataset(Dataset):
    def __init__(self, df, root, frames=16):
        self.df = df
        self.root = root
        self.frames = frames
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)), # Resize slightly larger
            transforms.RandomCrop((112, 112)), # Randomly crop to 112
            transforms.RandomHorizontalFlip(p=0.5), # Flip the video left-to-right
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Change lighting
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        v_path = os.path.join(self.root, self.df.iloc[idx, 0])
        label = self.df.iloc[idx, 2]

        cap = cv2.VideoCapture(v_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > self.frames:
            start_frame = torch.randint(0, total_frames - self.frames, (1,)).item()
        else:
            start_frame = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(self.frames):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(frame))
        cap.release()

        if len(frames) == 0:
            return torch.zeros(3, self.frames, 112, 112), torch.tensor(label).long()
            
        while len(frames) < self.frames:
            frames.append(frames[-1])

        return torch.stack(frames).permute(1, 0, 2, 3), torch.tensor(label).long()

# --- 2. THE MAIN EXECUTION FUNCTION ---
def train_model():
    print("Scanning directories for classes...")
    if not os.path.exists(VIDEO_DIR):
        print(f"ERROR: VIDEO_DIR {VIDEO_DIR} not found!")
        return

    CLASSES = sorted([f for f in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, f))])
    print(f"Detected {len(CLASSES)} classes: {CLASSES}")

    raw_df = pd.read_csv(SPLIT_FILE, sep=' ', header=None, names=['path', 'label'])
    def get_class_name(path): return path.split('/')[0]

    full_df = raw_df[raw_df['path'].apply(get_class_name).isin(CLASSES)].copy()
    full_df['label_id'] = full_df['path'].apply(lambda x: CLASSES.index(x.split('/')[0]))

    if len(full_df) == 0:
        print("ERROR: No matching videos found in dataframe. check SPLIT_FILE paths.")
        return

    train_df, val_df = train_test_split(full_df, test_size=0.15, stratify=full_df['label_id'])

    # num_workers=4 is safe now because of the __main__ protection
    train_loader = DataLoader(UCFDataset(train_df, VIDEO_DIR), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(UCFDataset(val_df, VIDEO_DIR), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Loading R3D_18 for {len(CLASSES)} classes...")
    model = models.video.r2plus1d_18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(DEVICE)

    weights_list = [4.0] * len(CLASSES)
    if 'Normal' in CLASSES:
        weights_list[CLASSES.index('Normal')] = 1.0 
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    print("Starting Training Loop...")
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Step {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / len(val_df)
        print(f"--- Epoch {epoch+1} Finished | Val Acc: {val_acc:.2f}% | Avg Loss: {running_loss/len(train_loader):.4f} ---")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_crime_model.pth")
            print("New Best Model Saved!")
        
        scheduler.step()
    
    print("Full Training Complete.")

# --- 3. THE PROTECTED ENTRY POINT ---
if __name__ == '__main__':
    train_model()