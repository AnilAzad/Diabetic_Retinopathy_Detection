# ─────────────────────────────────────────────
#  app.py  —  Gradio web app
#  Run: python app.py
#  Then open: http://localhost:7860
# ─────────────────────────────────────────────

# 1. IMPORTS
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import gradio as gr
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE   = 380
MODEL_PATH = "best_model.pth"   # created by train.py

LABELS = {
    0: ("No DR",             "✅ Healthy retina — no signs of diabetic retinopathy."),
    1: ("Mild DR",           "🟡 Early stage — microaneurysms present. Monitor closely."),
    2: ("Moderate DR",       "🟠 Moderate damage. Referral to ophthalmologist recommended."),
    3: ("Severe DR",         "🔴 Severe damage. Urgent ophthalmology referral needed."),
    4: ("Proliferative DR",  "🚨 Advanced stage. Immediate treatment required to prevent blindness.")
}

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_fundus_pil(pil_image, img_size=IMG_SIZE):
    """Convert PIL image and apply Ben Graham contrast enhancement."""
    img = np.array(pil_image.convert("RGB"))
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
# 5. GRAD-CAM
# ─────────────────────────────────────────────
def get_gradcam(model, img_tensor, original_img_np):

    target_layer = [model.backbone.blocks[-1][-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layer
    )

    grayscale_cam = cam(
        input_tensor=img_tensor.unsqueeze(0)
    )[0]

    # Resize original image to match CAM size
    resized_img = cv2.resize(
        original_img_np,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize
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
# 7. LOAD MODEL  (runs once at startup)
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Please run train.py first to generate best_model.pth"
    )

print(f"Loading model on {device} ...")
model = RetinopathyModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("Model loaded. Starting app...")

# ─────────────────────────────────────────────
# 8. PREDICTION FUNCTION  (called on each upload)
# ─────────────────────────────────────────────
def predict(pil_image):
    if pil_image is None:
        return None, {}

    # Preprocess
    img_np     = preprocess_fundus_pil(pil_image)
    img_tensor = infer_transform(image=img_np)["image"].to(device)

    # Predict
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    grade = int(probs.argmax())

    # Grad-CAM heatmap
    heatmap     = get_gradcam(model, img_tensor, img_np)
    heatmap_pil = Image.fromarray(heatmap)

    # Format label output for Gradio
    label_name, description = LABELS[grade]
    confidence_dict = {
        f"Grade {g} — {LABELS[g][0]}": float(probs[g])
        for g in range(5)
    }

    return heatmap_pil, confidence_dict

# ─────────────────────────────────────────────
# 9. GRADIO INTERFACE
# ─────────────────────────────────────────────
with gr.Blocks(title="Diabetic Retinopathy Screener") as demo:
    gr.Markdown("""
    # 🩺 Diabetic Retinopathy Screener
    Upload a retinal fundus image to detect diabetic retinopathy severity.
    The model classifies it into one of 5 grades (0 = Healthy → 4 = Proliferative DR).

    > ⚠️ **Disclaimer:** This tool is for screening assistance only, not clinical diagnosis.
    > Always consult a qualified ophthalmologist.
    """)

    with gr.Row():
        with gr.Column():
            input_img  = gr.Image(type="pil", label="Upload Fundus Image")
            submit_btn = gr.Button("Analyse Image", variant="primary")

        with gr.Column():
            heatmap_out = gr.Image(label="Grad-CAM Heatmap (regions that influenced prediction)")
            label_out   = gr.Label(label="DR Grade Probabilities", num_top_classes=5)

    submit_btn.click(fn=predict, inputs=input_img, outputs=[heatmap_out, label_out])

    gr.Markdown("""
    ### Grade Guide
    | Grade | Label | Action |
    |-------|-------|--------|
    | 0 | No DR | Routine annual check |
    | 1 | Mild DR | Monitor every 6 months |
    | 2 | Moderate DR | Referral recommended |
    | 3 | Severe DR | Urgent referral |
    | 4 | Proliferative DR | Immediate treatment |
    """)

if __name__ == "__main__":
    demo.launch(
    share=False,
    inbrowser=True
)   # set share=True to get a public link