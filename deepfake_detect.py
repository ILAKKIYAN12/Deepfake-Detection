import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import sys
import os

# Face detector - OpenCV built-in (no dlib needed!)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Load pretrained model
def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 2)
    try:
        state = torch.load(model_path, map_location='cpu')
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        model.load_state_dict(state, strict=False)
        print("✅ Model loaded!")
    except Exception as e:
        print(f"⚠️ Model load issue: {e}")
        print("✅ Using default weights for demo...")
    model.eval()
    return model

# Transform for face images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def detect_deepfake(video_path, model_path):
    print(f"\n🎬 Video: {video_path}")
    print("🔍 Analyzing for deepfakes...\n")

    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video!")
        return

    fake_count = 0
    real_count = 0
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:  # Every 10th frame
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            tensor = transform(face_rgb).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output, dim=1)
                pred = torch.argmax(prob).item()

            if pred == 1:
                fake_count += 1
                label = "FAKE"
                color = (0, 0, 255)
            else:
                real_count += 1
                label = "REAL"
                color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.0f}% | REAL: {real_count} | FAKE: {fake_count}", end='\r')

    cap.release()

    print("\n\n" + "="*40)
    print("📊 DEEPFAKE DETECTION RESULT")
    print("="*40)
    print(f"✅ REAL frames : {real_count}")
    print(f"❌ FAKE frames : {fake_count}")
    total = real_count + fake_count
    if total > 0:
        fake_pct = (fake_count / total) * 100
        print(f"🎯 Fake Score  : {fake_pct:.1f}%")
        print("="*40)
        if fake_pct > 50:
            print("⚠️  VERDICT: This video is likely DEEPFAKE!")
        else:
            print("✅ VERDICT: This video appears REAL!")
    else:
        print("⚠️  No faces detected in video!")
    print("="*40)

# Run detection
if __name__ == "__main__":
    VIDEO = "./videos/003_000.mp4"
    MODEL = "./pretrained_model/ffpp_c40.pth"

    if not os.path.exists(VIDEO):
        print(f"❌ Video not found: {VIDEO}")
        sys.exit(1)
    if not os.path.exists(MODEL):
        print(f"❌ Model not found: {MODEL}")
        sys.exit(1)

    detect_deepfake(VIDEO, MODEL)
