import base64
import os
import sys

import cv2
import torch
import torchvision.transforms as transforms

from network.models import model_selection

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

_model_cache = {}


def load_model(model_path):
    path = os.path.abspath(model_path)
    if path in _model_cache:
        return _model_cache[path]

    model = model_selection(modelname="xception", num_out_classes=2, dropout=0.5)
    kwargs = {"map_location": "cpu"}
    try:
        state = torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        state = torch.load(path, **kwargs)

    if isinstance(state, dict) and "model" in state and not any(
        k.startswith("model.") for k in state
    ):
        state = state["model"]

    model.load_state_dict(state, strict=False)
    model.eval()
    _model_cache[path] = model
    return model


def _encode_jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def detect_deepfake(video_path, model_path, frame_stride=10, max_sampled_frames=80):
    """Analyze a video and return a result dict (used by CLI and the web app)."""
    result = {
        "success": False,
        "real_count": 0,
        "fake_count": 0,
        "total_faces": 0,
        "fake_score": 0.0,
        "verdict": "UNKNOWN",
        "message": "",
        "preview": None,
        "frames_sampled": 0,
        "total_frames": 0,
    }

    if not os.path.exists(video_path):
        result["message"] = "Video file was not found."
        return result

    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["message"] = "Could not open this video. Try MP4 or AVI."
        return result

    fake_count = 0
    real_count = 0
    frame_count = 0
    sampled = 0
    preview = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    result["total_frames"] = total_frames
    stride = max(1, int(frame_stride))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % stride != 0:
            continue
        if sampled >= max_sampled_frames:
            break
        sampled += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y : y + h, x : x + w]
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
                color = (72, 72, 255)
            else:
                real_count += 1
                label = "REAL"
                color = (96, 196, 120)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, max(22, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
            preview = frame

        if total_frames > 0:
            progress = (frame_count / total_frames) * 100
            print(
                f"Progress: {progress:.0f}% | REAL: {real_count} | FAKE: {fake_count}",
                end="\r",
            )

    cap.release()

    total = real_count + fake_count
    result["success"] = True
    result["real_count"] = real_count
    result["fake_count"] = fake_count
    result["total_faces"] = total
    result["frames_sampled"] = sampled
    result["preview"] = _encode_jpeg(preview) if preview is not None else None

    if total > 0:
        fake_pct = (fake_count / total) * 100
        result["fake_score"] = round(fake_pct, 1)
        if fake_pct > 50:
            result["verdict"] = "DEEPFAKE"
            result["message"] = "This video is likely a deepfake."
        else:
            result["verdict"] = "REAL"
            result["message"] = "This video appears to be real."
    else:
        result["message"] = "No faces were detected in the sampled frames."

    return result


def print_report(result):
    print("\n\n" + "=" * 40)
    print("DEEPFAKE DETECTION RESULT")
    print("=" * 40)
    print(f"REAL faces : {result['real_count']}")
    print(f"FAKE faces : {result['fake_count']}")
    if result["total_faces"] > 0:
        print(f"Fake score : {result['fake_score']:.1f}%")
        print("=" * 40)
        print(f"VERDICT: {result['verdict']} — {result['message']}")
    else:
        print(result["message"])
    print("=" * 40)


if __name__ == "__main__":
    VIDEO = "./videos/003_000.mp4"
    MODEL = "./pretrained_model/ffpp_c40.pth"

    if not os.path.exists(VIDEO):
        print(f"Video not found: {VIDEO}")
        sys.exit(1)
    if not os.path.exists(MODEL):
        print(f"Model not found: {MODEL}")
        sys.exit(1)

<<<<<<< HEAD
    detect_deepfake(VIDEO, MODEL)
=======
    print(f"\nVideo: {VIDEO}")
    print("Analyzing for deepfakes...\n")
    report = detect_deepfake(VIDEO, MODEL)
    print_report(report)
    if not report["success"]:
        sys.exit(1)
>>>>>>> df049ba (Updated files and added new components)
