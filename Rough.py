import torch
import torchvision.transforms as t
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


def load_model():
    device_ = torch.device("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model_ = fasterrcnn_resnet50_fpn(weights=weights).to(device_)
    model_.eval()
    category_names_ = weights.meta["categories"]
    return model_, device_, category_names_


def preprocess_image(image, device_):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = t.Compose([
        t.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device_), image


def plot_detections(image, outputs, threshold=0.5):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(image)

    for idx, box in enumerate(outputs["boxes"]):
        score = outputs["scores"][idx].item()
        if score < threshold:
            continue

        x_min, y_min, x_max, y_max = box.detach().cpu().numpy()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min, f"{outputs['labels'][idx].item()} ({score:.2f})",
                bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')

    plt.show()


model, device, category_names = load_model()


def read(frame):
    image_tensor, image = preprocess_image(frame, device)

    with torch.no_grad():
        outputs = model(image_tensor)[0]

    detected_labels = [category_names[outputs['labels'][idx].item() - 1] for idx, score in enumerate(outputs["scores"])
                       if score.item() >= 0.5]

    print(detected_labels)

    return detected_labels
