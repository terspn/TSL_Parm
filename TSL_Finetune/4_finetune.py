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
PRETRAINED_MODEL_PATH = "../weights/msasl_pretrained.pth"
SAVE_MODEL_PATH = "tsl_weights/tsl_finetuned.pth" 

# Hyperparameters
NUM_CLASSES_TSL = 92 
INPUT_SIZE = 543 * 3 
HIDDEN_SIZE = 256
NUM_LAYERS = 2

EPOCHS = 200 
BATCH_SIZE = 32
PATIENCE = 50 

# Dual Learning Rate Strategy (Aggressive)
LR_HEAD = 5e-3 # 0.005: ใช้ LR สูงสำหรับ Head ใหม่
LR_BASE = 5e-5 # 0.00005: ใช้ LR ต่ำสำหรับ Base Model (MLP, LSTM)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
os.makedirs('tsl_weights', exist_ok=True) 

# --- 2. Model Architecture (เพิ่ม Dropout) ---
class SignLangModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=1000):
        super(SignLangModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5) # <-- NEW: เพิ่ม Dropout
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5) # <-- NEW: เพิ่ม Dropout
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc_final = nn.Linear(hidden_size * 2, num_classes) 

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_reshaped = x.view(-1, x.size(-1))
        
        out = self.fc1(x_reshaped)
        out = self.relu1(out)
        out = self.drop1(out) # <-- APPLY DROPOUT
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out) # <-- APPLY DROPOUT
        out = out.view(batch_size, seq_len, -1)
        
        lstm_out, (h_n, c_n) = self.lstm(out)
        
        final_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        output = self.fc_final(final_state)
        return output
# ------------------------------------------------------------------


# --- 3. Dataset Loader (ไม่เปลี่ยนแปลง) ---
class LandmarkDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path, label = self.data_list[idx]
        features = np.load(file_path)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return features_tensor, label_tensor

def pad_collate(batch):
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    max_len = max([len(f) for f in features])
    
    padded_features = []
    for f in features:
        padding_needed = max_len - len(f)
        padded = torch.nn.functional.pad(f, (0, 0, 0, padding_needed), 'constant', 0)
        padded_features.append(padded)
        
    padded_features_tensor = torch.stack(padded_features)
    labels_tensor = torch.stack(labels)
    
    return padded_features_tensor, labels_tensor


# --- 4. ฟังก์ชันหลักสำหรับ Fine-tuning ---
def main():
    # --- A. การโหลดข้อมูล ---
    all_data = []
    class_map = {} 
    
    for i, class_name in enumerate(sorted(os.listdir(DATA_FOLDER))):
        class_path = os.path.join(DATA_FOLDER, class_name)
        if os.path.isdir(class_path):
            class_map[class_name] = i
            for filename in os.listdir(class_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(class_path, filename)
                    all_data.append((file_path, i)) 
    
    if not all_data:
        print(f"Error: No data found in {DATA_FOLDER}. Please run 2_convert_multicore.py first.")
        return

    print(f"Total samples found: {len(all_data)}")
    print(f"Total classes found: {len(class_map)} (Expected: {NUM_CLASSES_TSL})")
    
    train_data, val_data = train_test_split(
        all_data, test_size=0.2, random_state=42, stratify=[item[1] for item in all_data]
    )
    
    train_dataset = LandmarkDataset(train_data)
    val_dataset = LandmarkDataset(val_data)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate
    )
    
    # --- B. โหลดโมเดลและปรับเปลี่ยน Head ---
    
    # 1. โหลดโมเดล Pre-trained (ใช้ num_classes=1000 ตาม MSASL)
    model = SignLangModel(num_classes=1000).to(DEVICE)
    try:
        pretrained_state = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE)
        
        # ลบ weights ของ fc_final ออกจาก state_dict
        pretrained_state.pop('fc_final.weight', None)
        pretrained_state.pop('fc_final.bias', None)
        
        # NOTE: ใช้ strict=False เพราะ fc_final ถูกลบออก
        model.load_state_dict(pretrained_state, strict=False) 
        print(f"Loaded Pre-trained weights from {PRETRAINED_MODEL_PATH} (except fc_final)")
    except FileNotFoundError:
        # หากไม่พบ Pre-trained weights จะขึ้นคำเตือนและเริ่มจากศูนย์
        print(f"Warning: Pre-trained model not found at {PRETRAINED_MODEL_PATH}. Starting from scratch.")

    # 2. เปลี่ยน Classification Head ให้เข้ากับ TSL 92 คลาส
    model.fc_final = nn.Linear(HIDDEN_SIZE * 2, NUM_CLASSES_TSL).to(DEVICE)
    
    # 3. ตั้งค่า Loss, Optimizer และ Scheduler
    criterion = nn.CrossEntropyLoss()
    
    # กำหนดกลุ่มพารามิเตอร์ (Dual Learning Rate)
    base_params = [param for name, param in model.named_parameters() if 'fc_final' not in name]
    head_params = [param for name, param in model.named_parameters() if 'fc_final' in name]

    # ใช้ LR ใหม่ที่ Aggressive
    optimizer = optim.Adam([
        {'params': base_params, 'lr': LR_BASE}, 
        {'params': head_params, 'lr': LR_HEAD} 
    ])

    # Scheduler 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
    )
    
    # Early Stopping
    best_val_acc = 0.0
    patience_counter = 0

    # --- C. Training Loop ---
    print("\nStarting Fine-tuning...")

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        model.train()
        running_loss = 0.0
        
        # 1. Training Phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        
        # 2. Validation Phase
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
        end_time = time.time()
        
        # แสดงผลลัพธ์
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {(end_time - start_time):.2f}s")
        
        # 3. Scheduler Step
        scheduler.step(val_acc)
        
        # 4. Early Stopping and Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"  --> Saved new best model with Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement.")
                break

    print(f"\nFine-tuning finished. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()