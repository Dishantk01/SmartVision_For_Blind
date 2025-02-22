import cv2
from ultralytics import YOLO
import torch

# Load the YOLOv8x model without the 'device' parameter.
model = YOLO("yolo11x.pt")

# Move the underlying PyTorch model to GPU if available.
if torch.mps.is_available():
    model.to(device="mps")
elif torch.cuda.is_available():
    model.to(device="cuda:0")
    print("Using GPU for inference.")
else:
    print("GPU not available. Using CPU.")


def read(frame):

    if len(frame) != 0:

        # Run YOLO object detection on the current frame.
        results = model(frame, verbose=False)[0]
        objects = []  # To store names of detected objects in this frame

        # Iterate over detections (if any) and process them.
        if results.boxes is not None:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = model.names[int(cls)]
                objects.append(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("", frame)
        cv2.waitKey(1)

        return objects
