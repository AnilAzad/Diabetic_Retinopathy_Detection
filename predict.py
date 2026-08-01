# ─────────────────────────────────────────────
#  predict.py  —  Grade a single fundus image
#  Run: python predict.py --image path/to/image.png
# ─────────────────────────────────────────────

# 1. IMPORTS
import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE   = 380
MODEL_PATH = "best_model.pth"   # created by train.py

LABELS = {
    0: "No DR (Healthy)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR (Urgent)"
}

# ─────────────────────────────────────────────
# 3. PREPROCESSING FUNCTION
# ─────────────────────────────────────────────
def preprocess_fundus(img_path, img_size=IMG_SIZE):
    """Ben Graham's preprocessing: enhances retinal vessel contrast."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), img_size // 30), -4,
        128
    )
    return img

# ─────────────────────────────────────────────
# 4. MODEL  (same definition as train.py)
# ─────────────────────────────────────────────
class RetinopathyModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=False, num_classes=0
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# ─────────────────────────────────────────────
# 5. GRAD-CAM HEATMAP
# ─────────────────────────────────────────────
def get_gradcam(model, img_tensor, original_img_np):

    """
    Generate Grad-CAM heatmap.
    """

    # Target layer
    target_layer = [model.backbone.blocks[-1][-1]]

    # Create CAM object
    cam = GradCAM(
        model=model,
        target_layers=target_layer
    )

    # Generate grayscale CAM
    grayscale_cam = cam(
        input_tensor=img_tensor.unsqueeze(0)
    )[0]

    # Resize original image to match CAM size
    resized_img = cv2.resize(
        original_img_np,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize image
    resized_img = resized_img.astype(np.float32) / 255.0

    # Overlay heatmap
    visualization = show_cam_on_image(
        resized_img,
        grayscale_cam,
        use_rgb=True
    )

    return visualization

# ─────────────────────────────────────────────
# 6. INFERENCE TRANSFORM
# ─────────────────────────────────────────────
infer_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DR grade from a fundus image")
    parser.add_argument("--image",  required=True,          help="Path to fundus image (.png or .jpg)")
    parser.add_argument("--output", default="heatmap.png",  help="Where to save the Grad-CAM heatmap")
    parser.add_argument("--model",  default=MODEL_PATH,     help="Path to trained model weights")
    args = parser.parse_args()

    # Check model file exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            "Run train.py first to generate best_model.pth"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model from {args.model} ...")
    model = RetinopathyModel()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess image
    print(f"Processing image: {args.image}")
    img_np    = preprocess_fundus(args.image)
    img_tensor = infer_transform(image=img_np)["image"].to(device)

    # Predict
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    grade      = int(probs.argmax())
    confidence = float(probs[grade]) * 100

    # Print result
    print("\n" + "─" * 40)
    print(f"  Grade     : {grade} — {LABELS[grade]}")
    print(f"  Confidence: {confidence:.1f}%")
    print("─" * 40)
    print("\nAll probabilities:")
    for g, label in LABELS.items():
        bar = "█" * int(probs[g] * 30)
        print(f"  Grade {g} ({label[:20]:<20}) {probs[g]*100:5.1f}%  {bar}")

    # Save Grad-CAM heatmap
    print(f"\nGenerating Grad-CAM heatmap ...")
    heatmap = get_gradcam(model, img_tensor, img_np)
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, heatmap_bgr)
    print(f"Heatmap saved to: {args.output}")