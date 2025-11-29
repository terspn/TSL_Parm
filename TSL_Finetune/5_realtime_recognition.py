# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import os
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import time

# --- 1. CONFIGURATION ---
MAX_SEQ_LEN = 60
MODEL_PATH = "tsl_weights/tsl_finetuned.pth" 
DEVICE = torch.device("cpu") # Force CPU to match training and avoid Mac MPS bugs
FONT_PATH = "Sarabun-Light.ttf"
FONT_SIZE = 50 
PRED_HISTORY_LEN = 60
RESET_DURATION_SECONDS = 3.5
CONFIDENCE_THRESHOLD = 0.60 

# *** CRITICAL UPDATE: NEW INPUT SIZE ***
# Pose(33*3) + Left Hand(21*3) + Right Hand(21*3) = 99 + 63 + 63 = 225
INPUT_SIZE = 225 

# --- 2. TSL 92 CLASS LABELS ---
TSL_LABELS = [
    "สวัสดี", "หิว", "หิวมั้ย", "ขอบคุณ", "ขอโทษ", "กินข้าวยัง", "กินข้าวแล้ว", "รัก", "กี่โมง", "คิดถึง", 
    "คุณชื่ออะไร", "ง่วง", "ฉันหูดี", "เหงา", "ฉันหูหนวก", "น่ารัก", "น่ารักมั้ย", "ปวดท้อง", "ปวดหัว", "ป่วย", 
    "มีความสุข", "ไม่ใช่", "ไม่เป็นไร", "ร้องไห้", "ลาก่อน", "แล้วเจอกัน", "เศร้า", 
    # Alphabets, Vowels, Tones (Index 27-91)
    "ก", "ข", "ค", "ฆ", "ง", "ฉ", "จ", "ช", "ซ", "ฌ", "ญ", "ฎ", "ฏ", "ฐ", "ฑ", "ฒ", "ด", "ต", 
    "ถ", "ท", "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ", "ฟ", "ภ", "ม", "ย", "ร", "ฤ", "ล", "ว", 
    "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ณ", "ฮ", "็", "่", "้", "๋", "์", "ฯ", "ะ", "ั", "า", 
    "ำ", "๊", "ึ", "ื", "ุ", "ู", "เ", "แ", "โ", "ใ", "ไ", "ิ", "ี"
]

# --- 3. Helper Functions ---
def put_thai_text(img, text, pos, font_path, font_size, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default() 
        font_size = 20
    draw.text(pos, text, font=font, fill=color)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def load_class_labels():
    index_to_label = {i: label for i, label in enumerate(TSL_LABELS)}
    return index_to_label, len(TSL_LABELS)

# --- 4. Model Architecture (MUST MATCH TRAINING) ---
class SignLangModel(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=128, num_layers=2, num_classes=92):
        super(SignLangModel, self).__init__()
        
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        
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
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.dropout(x)
        output, hidden = self.gru(x)
        final_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc_out(final_state)

# --- 5. Feature Extraction (UPDATED FOR 225 FEATURES) ---
def extract_features(results):
    # Total features: 33*3 (Pose) + 21*3 (LH) + 21*3 (RH) = 99 + 63 + 63 = 225
    frame_features = np.zeros(225, dtype=np.float32)
    offset = 0
    
    # 1. Pose (33 points -> 99 features)
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i < 33:
                frame_features[offset:offset+3] = [landmark.x, landmark.y, landmark.z]
                offset += 3
    else:
        offset += 33 * 3 
    
    # NOTE: We SKIP Face landmarks completely to match training!

    # 2. Left Hand (21 points -> 63 features)
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            frame_features[offset:offset+3] = [landmark.x, landmark.y, landmark.z]
            offset += 3
    else:
        # Fill with -1 or 0 for missing hand
        frame_features[offset : offset+(21*3)] = 0.0
        offset += 21 * 3

    # 3. Right Hand (21 points -> 63 features)
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            frame_features[offset:offset+3] = [landmark.x, landmark.y, landmark.z]
            offset += 3
    else:
        frame_features[offset : offset+(21*3)] = 0.0
        offset += 21 * 3
        
    return frame_features

# --- 6. Main Inference Loop ---
def main():
    # 1. Setup
    index_to_label, num_classes = load_class_labels()
    
    if num_classes != 92:
        print(f"FATAL ERROR: Found {num_classes} classes, expected 92.")
        return

    RECOGNITION_MODE = 'WORD'
    WORD_INDICES = list(range(0, 27))
    ALPHABET_INDICES = list(range(27, 92))
    
    print(f"Loaded {num_classes} classes. Mode: {RECOGNITION_MODE}")

    # 2. Initialize Model (hidden_size=128, input_size=225)
    model = SignLangModel(
        input_size=INPUT_SIZE,  # 225
        hidden_size=128,  
        num_layers=2, 
        num_classes=num_classes
    ).to(DEVICE)

    try:
        # Load weights
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state) 
        model.eval()
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. MediaPipe & Video
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    sequence_buffer = deque(maxlen=MAX_SEQ_LEN)
    prediction_history = deque(maxlen=PRED_HISTORY_LEN) 
    
    current_prediction = "Waiting..."
    prediction_prob = 0.0
    
    last_stable_time = time.time()
    last_stable_index = -1 

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1) 
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract features (Now size 225)
            features = extract_features(results)
            sequence_buffer.append(features)
            
            # Predict
            if len(sequence_buffer) == MAX_SEQ_LEN:
                input_tensor = np.array(list(sequence_buffer), dtype=np.float32)
                input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(input_tensor)
                    output_logits = logits.clone()
                    
                    # Filtering
                    if RECOGNITION_MODE == 'WORD':
                        for idx in ALPHABET_INDICES:
                            output_logits[0, idx] = -float('inf')
                    elif RECOGNITION_MODE == 'ALPHABET':
                        for idx in WORD_INDICES:
                            output_logits[0, idx] = -float('inf')
                    
                    prediction_history.append(output_logits.cpu().numpy().flatten())
                
                # Smoothing
                if len(prediction_history) == PRED_HISTORY_LEN:
                    mean_logits = np.mean(list(prediction_history), axis=0)
                    probabilities = torch.softmax(torch.tensor(mean_logits).unsqueeze(0), dim=1)
                    pred_prob, pred_index = torch.max(probabilities, 1)
                    current_index = pred_index.item()
                    
                    # Logic
                    if pred_prob.item() < CONFIDENCE_THRESHOLD:
                        current_prediction = "..."
                        prediction_prob = pred_prob.item()
                        last_stable_index = -1 
                    elif current_index == last_stable_index and last_stable_index != -1:
                        elapsed_time = time.time() - last_stable_time
                        if elapsed_time >= RESET_DURATION_SECONDS:
                            current_prediction = "✅ READY"
                            sequence_buffer.clear()
                            prediction_history.clear()
                            last_stable_index = -1
                        else:
                            current_prediction = f"{index_to_label.get(last_stable_index, 'UNKNOWN')} ({RESET_DURATION_SECONDS - elapsed_time:.1f}s)"
                            prediction_prob = pred_prob.item()
                    else:
                        current_prediction = index_to_label.get(current_index, "UNKNOWN")
                        prediction_prob = pred_prob.item()
                        last_stable_index = current_index
                        last_stable_time = time.time()

            # Draw
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            image = put_thai_text(image, f'โหมด: {RECOGNITION_MODE}', (20, 20), FONT_PATH, 30, (255, 255, 0))
            image = put_thai_text(image, f'คำศัพท์: {current_prediction}', (20, 60), FONT_PATH, 50, (0, 255, 0))
            cv2.putText(image, f'Prob: {prediction_prob:.2f}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow('TSL Recognition', image)

            # Key controls
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'): # Toggle Mode
                RECOGNITION_MODE = 'ALPHABET' if RECOGNITION_MODE == 'WORD' else 'WORD'
                print(f"Switched mode to: {RECOGNITION_MODE}")
                sequence_buffer.clear()
                prediction_history.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()