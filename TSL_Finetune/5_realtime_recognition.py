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

# --- 1. CONFIG สำหรับ Real-time (แก้ไข Confidence Threshold) ---
MAX_SEQ_LEN = 60
MODEL_PATH = "tsl_weights/tsl_finetuned.pth" 
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
FONT_PATH = "Sarabun-Light.ttf"
FONT_SIZE = 50 
PRED_HISTORY_LEN = 60
RESET_DURATION_SECONDS = 3.5

# แก้ไข: ลดเกณฑ์ความเชื่อมั่นลงเพื่อเปิดโอกาสให้โมเดลทำนาย
# CONFIDENCE_THRESHOLD = 0.40
CONFIDENCE_THRESHOLD = 0.05

# ----------------------------------

# --- 2. TSL 92 CLASS LABELS ---
TSL_LABELS = [
    "สวัสดี", "หิว", "หิวมั้ย", "ขอบคุณ", "ขอโทษ", "กินข้าวยัง", "กินข้าวแล้ว", "รัก", "กี่โมง", "คิดถึง", 
    "คุณชื่ออะไร", "ง่วง", "ฉันหูดี", "เหงา", "ฉันหูหนวก", "น่ารัก", "น่ารักมั้ย", "ปวดท้อง", "ปวดหัว", "ป่วย", 
    "มีความสุข", "ไม่ใช่", "ไม่เป็นไร", "ร้องไห้", "ลาก่อน", "แล้วเจอกัน", "เศร้า", 
    # พยัญชนะ สระ และวรรณยุกต์ (Index 27-91)
    "ก", "ข", "ค", "ฆ", "ง", "ฉ", "จ", "ช", "ซ", "ฌ", "ญ", "ฎ", "ฏ", "ฐ", "ฑ", "ฒ", "ด", "ต", 
    "ถ", "ท", "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ", "ฟ", "ภ", "ม", "ย", "ร", "ฤ", "ล", "ว", 
    "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ณ", "ฮ", "็", "่", "้", "๋", "์", "ฯ", "ะ", "ั", "า", 
    "ำ", "๊", "ึ", "ื", "ุ", "ู", "เ", "แ", "โ", "ใ", "ไ", "ิ", "ี"
]
# ------------------------------------------------------------------

# --- 3. ฟังก์ชันช่วย: วาดภาษาไทยด้วย PIL (ไม่เปลี่ยนแปลง) ---
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

def get_input_size():
    return 543 * 3

def load_class_labels():
    index_to_label = {i: label for i, label in enumerate(TSL_LABELS)}
    return index_to_label, len(TSL_LABELS)

# --- 4. Model Architecture (เพิ่ม Dropout และแก้ fc2) ---
# class SignLangModel(nn.Module):
#     def __init__(self, input_size=543 * 3, hidden_size=256, num_layers=2, num_classes=92):
#         super(SignLangModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size * 2)
#         self.relu1 = nn.ReLU()
#         self.drop1 = nn.Dropout(0.5) # <-- NEW DROPOUT
#         self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
#         self.relu2 = nn.ReLU()
#         self.drop2 = nn.Dropout(0.5) # <-- NEW DROPOUT
        
#         self.lstm = nn.LSTM(
#             input_size=hidden_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True
#         )
        
#         self.fc_final = nn.Linear(hidden_size * 2, num_classes) 

#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()
#         x_reshaped = x.view(-1, x.size(-1))
        
#         # MLP layers 
#         out = self.fc1(x_reshaped)
#         out = self.relu1(out)
#         out = self.drop1(out) # <-- APPLY DROPOUT
#         out = self.fc2(out) # <-- แก้ไขแล้ว: ใช้อินพุต 'out'
#         out = self.relu2(out)
#         out = self.drop2(out) # <-- APPLY DROPOUT
#         out = out.view(batch_size, seq_len, -1)
        
#         lstm_out, (h_n, c_n) = self.lstm(out)
        
#         final_state = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
#         output = self.fc_final(final_state)
#         return output
#  -----------------------------------------------------------------------
class SignLangModel(nn.Module):
    def __init__(self, 
                 input_size=543*3, 
                 hidden_size=256, 
                 num_layers=2, 
                 num_classes=92):
        super(SignLangModel, self).__init__()

        # --- MLP Encoder ---
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.4)

        # --- LSTM ---
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # --- LayerNorm after LSTM ---
        self.norm = nn.LayerNorm(hidden_size * 2)

        # --- Multi-Head Self Attention ---
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )

        # --- Final Classifier ---
        self.fc_final = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        # --- 0) Normalize landmark features ---
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

        batch_size, seq_len, _ = x.size()
        x_reshaped = x.view(-1, x.size(-1))

        # --- 1) MLP Encoder ---
        out = self.fc1(x_reshaped)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        out = out.view(batch_size, seq_len, -1)

        # --- 2) LSTM ---
        lstm_out, _ = self.lstm(out)

        # --- 3) LayerNorm ---
        lstm_out = self.norm(lstm_out)

        # --- 4) Self-Attention ---
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # --- 5) Average pooling over time ---
        final_state = torch.mean(attn_out, dim=1)

        # --- 6) Classifier ---
        output = self.fc_final(final_state)
        return output


# --- 5. ฟังก์ชันสกัด Landmark (ไม่เปลี่ยนแปลง) ---
def extract_features(results):
    frame_features = np.zeros(get_input_size(), dtype=np.float32)
    offset = 0
    
    # ... (ส่วนการสกัด Landmark เหมือนเดิม) ...
    # 1. Pose
    if results.pose_landmarks:
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if i < 33:
                frame_features[offset] = landmark.x
                frame_features[offset + 1] = landmark.y
                frame_features[offset + 2] = landmark.z
                offset += 3
    else:
        offset += 33 * 3 
    
    # 2. Face
    if results.face_landmarks:
        for i, landmark in enumerate(results.face_landmarks.landmark):
            if i < 468:
                frame_features[offset] = landmark.x
                frame_features[offset + 1] = landmark.y
                frame_features[offset + 2] = landmark.z
                offset += 3
    else:
        offset += 468 * 3 
        
    # 3) Left Hand
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            frame_features[offset] = landmark.x
            frame_features[offset+1] = landmark.y
            frame_features[offset+2] = landmark.z
            offset += 3
    else:
        # Padding for missing left hand (use -1 instead of 0)
        frame_features[offset : offset+(21*3)] = -1.0
        offset += 21 * 3

    # 4) Right Hand
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            frame_features[offset] = landmark.x
            frame_features[offset+1] = landmark.y
            frame_features[offset+2] = landmark.z
            offset += 3
    else:
        # Padding for missing right hand
        frame_features[offset : offset+(21*3)] = -1.0
        offset += 21 * 3

        
    return frame_features
# -----------------------------------------------------------


def main():
    # 1. โหลด Label และ Model
    index_to_label, num_classes = load_class_labels()
    
    if num_classes != 92:
        print(f"FATAL ERROR: Found {num_classes} classes, expected 92. Check TSL_LABELS list.")
        return

    # *** ตั้งค่าโหมดการรู้จำ ***
    RECOGNITION_MODE = 'WORD'
    WORD_INDICES = list(range(0, 27))
    ALPHABET_INDICES = list(range(27, 92))
    
    print(f"Loaded {num_classes} TSL classes for recognition. Initial Mode: {RECOGNITION_MODE}")

    model = SignLangModel(num_classes=num_classes).to(DEVICE)
    try:
        # model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        model.eval() # <-- IMPORTANT: ปิด Dropout เสมอในการทดสอบ
        print(f"Successfully loaded model weights from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Did you run 4_finetune.py?")
        return

    # 2. เตรียม MediaPipe และ Video Input
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    sequence_buffer = deque(maxlen=MAX_SEQ_LEN)
    prediction_history = deque(maxlen=PRED_HISTORY_LEN) 
    
    current_prediction = "Waiting for sign..."
    prediction_prob = 0.0
    
    # Tracking Time and Index สำหรับ Timer Reset
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
            
            # 3. ประมวลผล Frame ด้วย MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            print("POSE:", results.pose_landmarks is not None,
            "LH:", results.left_hand_landmarks is not None,
            "RH:", results.right_hand_landmarks is not None)

            # 4. สกัด Landmark และเพิ่มเข้า Buffer
            features = extract_features(results)
            sequence_buffer.append(features)
            
            # 5. ทำนายเมื่อ Buffer เต็ม
            if len(sequence_buffer) == MAX_SEQ_LEN:
                
                input_tensor = np.array(list(sequence_buffer), dtype=np.float32)
                input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = model(input_tensor)
                    output_logits = logits.clone()
                    
                    # Filtering/Masking (ใช้โหมดที่ตั้งไว้)
                    if RECOGNITION_MODE == 'WORD':
                        for idx in ALPHABET_INDICES:
                            output_logits[0, idx] = -float('inf')
                    elif RECOGNITION_MODE == 'ALPHABET':
                        for idx in WORD_INDICES:
                            output_logits[0, idx] = -float('inf')
                    
                    prediction_history.append(output_logits.cpu().numpy().flatten())

                # try:
                #     print("Pred:", current_index, "Prob:", pred_prob.item())
                # except:
                #     print("Pred: None (no prediction yet)")

                
                # 6. ทำ Temporal Smoothing และ Timer Logic
                if len(prediction_history) == PRED_HISTORY_LEN:
                    
                    mean_logits = np.mean(list(prediction_history), axis=0)
                    probabilities = torch.softmax(torch.tensor(mean_logits).unsqueeze(0), dim=1)
                    pred_prob, pred_index = torch.max(probabilities, 1)
                    
                    current_index = pred_index.item()
                    
                    # === Timer Reset Logic + Confidence Check (ปรับปรุงแล้ว) ===
                    
                    # 1. ตรวจสอบว่าคำทำนายมีความน่าเชื่อถือสูงพอหรือไม่
                    if pred_prob.item() < CONFIDENCE_THRESHOLD:
                        current_prediction = "No Sign Detected" # ไม่ถึงเกณฑ์
                        prediction_prob = pred_prob.item()
                        last_stable_index = -1 

                    # 2. ถ้าคำทำนายมีความน่าเชื่อถือและยังคงเดิม
                    elif current_index == last_stable_index and last_stable_index != -1:
                        elapsed_time = time.time() - last_stable_time
                        
                        if elapsed_time >= RESET_DURATION_SECONDS:
                            # เกินเวลาที่กำหนด: รีเซตระบบ
                            current_prediction = "✅ 5s ELAPSED! READY"
                            prediction_prob = 0.0
                            
                            sequence_buffer.clear()
                            prediction_history.clear()
                            last_stable_index = -1
                            last_stable_time = time.time()
                        else:
                            # ยังไม่ครบเวลา: แสดงคำทำนายเดิมและเวลาที่เหลือ
                            current_prediction = f"{index_to_label.get(last_stable_index, 'UNKNOWN')} ({RESET_DURATION_SECONDS - elapsed_time:.1f}s left)"
                            prediction_prob = pred_prob.item()

                    # 3. ถ้าคำทำนายเปลี่ยน (หรือคำแรกที่น่าเชื่อถือ)
                    else:
                        thai_label = index_to_label.get(current_index, "UNKNOWN")
                        current_prediction = thai_label
                        prediction_prob = pred_prob.item()
                        
                        # รีเซต Timer/เริ่มต้น
                        last_stable_index = current_index
                        last_stable_time = time.time()
                    
                    # === END Timer Reset LOGIC ===

            # 7. วาด Landmark และแสดงผลลัพธ์
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # แสดงผลภาษาไทยโดยใช้ PIL
            image = put_thai_text(
                img=image,
                text=f'โหมด: {RECOGNITION_MODE} (Words)',
                pos=(20, 20),
                font_path=FONT_PATH,
                font_size=FONT_SIZE - 20,
                color=(255, 255, 0)
            )
            image = put_thai_text(
                img=image,
                text=f'คำศัพท์: {current_prediction}',
                pos=(20, 50),
                font_path=FONT_PATH,
                font_size=FONT_SIZE,
                color=(0, 255, 0)
            )
            
            # แสดง Prob และ Buffer
            cv2.putText(image, f'Prob: {prediction_prob:.2f}', (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Buffer: {len(sequence_buffer)}/{MAX_SEQ_LEN}', (20, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('TSL Real-time Recognition', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()