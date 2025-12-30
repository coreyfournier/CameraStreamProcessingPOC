import argparse
import time
import cv2
import numpy as np

# Common MobileNet-SSD classes
CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def parse_args():
    p = argparse.ArgumentParser(description="Detect objects from Synology Surveillance Station RTSP stream with OpenCV.")
    p.add_argument("--rtsp", required=True, help="RTSP URL from Synology Surveillance Station camera.")
    p.add_argument("--proto", required=True, help="Path to MobileNet-SSD deploy prototxt.")
    p.add_argument("--model", required=True, help="Path to MobileNet-SSD Caffe model.")
    p.add_argument("--out", required=True, help="Output video file path, e.g., output.mp4.")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    p.add_argument("--width", type=int, default=640, help="Resize width for processing.")
    p.add_argument("--height", type=int, default=360, help="Resize height for processing.")
    p.add_argument("--fps", type=float, default=20.0, help="Output video FPS.")
    return p.parse_args()

def main():
    args = parse_args()

    net = cv2.dnn.readNetFromCaffe(args.proto, args.model)

    cap = cv2.VideoCapture(args.rtsp, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Failed to open RTSP stream. Verify URL, credentials, and network access.")

    # Initialize video writer after reading first frame to get size
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    last_ts = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # RTSP streams may stall; small backoff then retry
                time.sleep(0.1)
                continue

            # Resize for processing for speed
            frame_resized = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_LINEAR)

            blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (args.width, args.height), 127.5, swapRB=True)
            net.setInput(blob)
            detections = net.forward()

            # Lazy-init writer with actual frame size of output (resized frame)
            if writer is None:
                writer = cv2.VideoWriter(args.out, fourcc, args.fps, (args.width, args.height))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open writer for output file: {args.out}")

            # Draw detections
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < args.conf:
                    continue

                idx = int(detections[0, 0, i, 1])
                if idx < 0 or idx >= len(CLASS_NAMES):
                    continue

                box = detections[0, 0, i, 3:7] * np.array([args.width, args.height, args.width, args.height])
                (startX, startY, endX, endY) = box.astype("int")

                label = f"{CLASS_NAMES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame_resized, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame_resized, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            writer.write(frame_resized)

            # Basic rate-limiting for CPU
            now = time.time()
            elapsed = now - last_ts
            target_dt = 1.0 / args.fps
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
            last_ts = now

    finally:
        if writer:
            writer.release()
        cap.release()

if __name__ == "__main__":
    main()