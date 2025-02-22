import torch
import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection

# Check if CUDA is available
device = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Load model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm").to(device)


def read(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV format) to RGB

    # Prepare inputs
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to COCO API format
    target_sizes = torch.tensor([image.shape[:2]], device=device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.75)[0]
    objects = []

    # Print detected objects
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        objects.append(model.config.id2label[label.item()])
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

    return objects
