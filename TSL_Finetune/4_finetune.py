# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
import time 

# --- 1. CONFIGURATION ---
DATA_FOLDER = 'landmarks_json_tsl' 
SAVE_MODEL_PATH = "tsl_weights/tsl_finetuned.pth" 

# Hyperparameters
NUM_CLASSES_TSL = 92 
# NEW INPUT SIZE: Pose(33) + Hands(21+21) = 75 points * 3 coords = 225
INPUT_SIZE = 225 
HIDDEN_SIZE = 128      
NUM_LAYERS = 2
EPOCHS = 200 
BATCH_SIZE = 32
PATIENCE = 50 
LR_RATE = 1e-4

# Force CPU to avoid MPS NaN bugs
DEVICE = torch.device("cpu") 
print(f"Using device: {DEVICE}")
os.makedirs('tsl_weights', exist_ok=True) 

# --- 2. Model Architecture ---
class SignLangModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES_TSL):
        super(SignLangModel, self).__init__()
        
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4) # Increased Dropout slightly
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, 
            dropout=0.4
        )
        
        self.fc_out = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Handle NaNs in input
        x = torch.nan_to_num(x, nan=0.0)

        # Normalize (Standardization)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)
        
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        output, hidden = self.gru(x)
        
        final_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc_out(final_state)

# --- 3. Dataset Loader with AUGMENTATION & SLICING ---
class LandmarkDataset(Dataset):
    def __init__(self, data_list, augment=False):
        self.data_list = data_list
        self.augment = augment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path, label = self.data_list[idx]
        features = np.load(file_path)
        
        # Safety check for NaNs
        if np.isnan(features).any():
            features = np.nan_to_num(features, nan=0.0)

        # --- FEATURE SELECTION (CRITICAL STEP) ---
        # The .npy file has 1629 features (Pose + Face + Hands)
        # We only want Pose (0-99) and Hands (1503-1629)
        # Removing Face (99-1503) eliminates huge noise!
        pose = features[:, 0:99]
        # face = features[:, 99:1503] # SKIP THIS
        left_hand = features[:, 1503:1566]
        right_hand = features[:, 1566:1629]
        
        # Concatenate selected features (New shape: seq_len, 225)
        features = np.concatenate([pose, left_hand, right_hand], axis=1)

        # --- DATA AUGMENTATION ---
        if self.augment:
            features = self.augment_data(features)
            
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return features_tensor, label_tensor

    def augment_data(self, data):
        # 1. Random Scaling (Size change)
        scale = random.uniform(0.9, 1.1)
        data = data * scale
        
        # 2. Random Time Shift (move sequence slightly forward/back)
        if random.random() < 0.5:
            shift = random.randint(-2, 2)
            if shift > 0:
                data = np.pad(data, ((shift, 0), (0, 0)), mode='constant')[:-shift]
            elif shift < 0:
                data = np.pad(data, ((0, -shift), (0, 0)), mode='constant')[-shift:]
                
        # 3. Add Gaussian Noise (Simulate camera jitter)
        noise = np.random.normal(0, 0.005, data.shape)
        data = data + noise
        
        return data

def pad_collate(batch):
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    max_len = max([len(f) for f in features])
    padded_features = []
    for f in features:
        padding_needed = max_len - len(f)
        padded = torch.nn.functional.pad(f, (0, 0, 0, padding_needed), 'constant', 0)
        padded_features.append(padded)
    return torch.stack(padded_features), torch.stack(labels)

# --- 4. Main Training Function ---
def main():
    all_data = []
    class_map = {} 
    
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Folder '{DATA_FOLDER}' not found.")
        return

    for i, class_name in enumerate(sorted(os.listdir(DATA_FOLDER))):
        class_path = os.path.join(DATA_FOLDER, class_name)
        if os.path.isdir(class_path):
            class_map[class_name] = i
            for filename in os.listdir(class_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(class_path, filename)
                    all_data.append((file_path, i)) 
    
    if not all_data:
        print("Error: No data found.")
        return

    print(f"Total samples: {len(all_data)}")
    
    train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42, stratify=[item[1] for item in all_data])
    
    # Enable Augmentation ONLY for Training
    train_loader = DataLoader(LandmarkDataset(train_data, augment=True), batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(LandmarkDataset(val_data, augment=False), batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    
    print(f"Initializing Optimized GRU model (Input: {INPUT_SIZE})...")
    model = SignLangModel(num_classes=NUM_CLASSES_TSL).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    patience_counter = 0

    print("\nStarting Training...")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                print(f"⚠️ Loss is NaN at Epoch {epoch+1}. Skipping batch...")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_data)
        
        # Validation
        model.eval()
        corrects = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                corrects += torch.sum(preds == labels.data)

        val_acc = corrects.float() / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {(time.time() - start_time):.2f}s")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

if __name__ == "__main__":
    main()