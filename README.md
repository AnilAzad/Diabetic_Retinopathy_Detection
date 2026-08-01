# 🩺 Diabetic Retinopathy Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-IDRiD-orange)
![Deployment](https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-yellow?logo=huggingface)

An end-to-end deep learning system for automated grading of **Diabetic Retinopathy (DR)** from retinal fundus images.
Built with **EfficientNet-B3 transfer learning**, **Grad-CAM explainability**, and deployed as a live web app on Hugging Face Spaces.

> 🎯 Achieves **~82% validation accuracy** and **0.87 Quadratic Weighted Kappa** — surpassing the clinical screening threshold of 0.85.

---

## 📌 What is Diabetic Retinopathy?

Diabetic retinopathy (DR) is a diabetes complication that damages blood vessels in the retina. It is the **leading cause of preventable blindness** worldwide — affecting 1 in 3 diabetics among 537 million patients globally. Early detection is critical: **90% of blindness cases can be prevented** if caught in time.

This model screens fundus images and classifies them into 5 severity grades:

| Grade | Label              | Description                              | Recommended Action          |
|-------|--------------------|------------------------------------------|-----------------------------|
| 0     | No DR              | Healthy retina                           | Routine annual check        |
| 1     | Mild DR            | Microaneurysms only                      | Monitor every 6 months      |
| 2     | Moderate DR        | More than mild, less than severe         | Referral recommended        |
| 3     | Severe DR          | Extensive haemorrhages                   | Urgent referral             |
| 4     | Proliferative DR   | Neovascularisation — vision at risk      | Immediate treatment         |

---

## 🗂️ Project Structure

```
Diabetic_Retinopathy_Detection/
├── data/
│   ├── train_images/        # Fundus images for training (~400 images)
│   ├── test_images/         # Fundus images for testing
│   ├── train.csv            # Labels: id_code, diagnosis (0–4)
│   └── test.csv             # Test image filenames
├── train.py                 # Train the model → saves best_model.pth
├── predict.py               # Grade a single image from terminal
├── app.py                   # Gradio web app (run locally or deploy)
├── requirements.txt         # All dependencies
└── README.md
```

---

## 📦 Dataset — IDRiD (Indian Diabetic Retinopathy Image Dataset)

| Property       | Details                                      |
|----------------|----------------------------------------------|
| Size           | 516 labelled fundus images                   |
| Grades         | 5 DR severity levels (0–4)                   |
| Origin         | Nanded, Maharashtra, India                   |
| License        | CC BY 4.0 — Free for research & projects     |
| Extra Labels   | Pixel-level lesion annotations included      |
| Download       | [idrid.grand-challenge.org](https://idrid.grand-challenge.org/Data/) |

```bash
# After downloading, place files like this:
data/
├── train_images/    ← all .jpg fundus images
├── train.csv        ← columns: id_code, diagnosis
└── test.csv
```

---

## 🧠 Model Architecture

```
Input Image (380×380×3)
        ↓
Ben Graham Preprocessing (contrast enhancement)
        ↓
albumentations Augmentation Pipeline
        ↓
EfficientNet-B3 Backbone (pretrained on ImageNet, via timm)
        ↓
Custom Classification Head:
    Dropout(0.3) → Linear(1536→256) → ReLU → Dropout(0.2) → Linear(256→5)
        ↓
Softmax → DR Grade (0–4) + Confidence
        ↓
Grad-CAM Heatmap (explainability overlay)
```

**Training configuration:**
- Loss: `CrossEntropyLoss` with inverse-frequency class weights
- Optimiser: `AdamW` (lr=3e-4, weight_decay=1e-4)
- LR Schedule: Cosine Annealing over 15 epochs
- Batch size: 16
- Input size: 380×380

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone [https://github.com/AnilAzad/Diabetic_Retinopathy_Detection.git]
cd Diabetic_Retinopathy_Detection

# 2. Install dependencies
pip install torch torchvision timm albumentations grad-cam opencv-python gradio pandas scikit-learn
```

> **No GPU on your laptop?** Use [Kaggle Notebooks](https://www.kaggle.com/code) — free T4 GPU, 30 hrs/week. Upload the files and run the same commands in notebook cells.

---

## 🚀 Quickstart

### Step 1 — Train the model

```bash
python train.py
```

What happens:
- Loads IDRiD fundus images and CSV labels
- Applies Ben Graham contrast enhancement + augmentation
- Fine-tunes EfficientNet-B3 for 15 epochs
- Prints loss, accuracy, and QWK after each epoch
- Saves the best model as `best_model.pth`

⏱️ Training time: ~30–45 minutes on a free Kaggle T4 GPU.

---

### Step 2 — Predict on a single image

```bash
python predict.py --image data/test_images/sample.jpg
```

Example output:
```
Grade     : 2 — Moderate DR
Confidence: 76.4%

All probabilities:
  Grade 0 (No DR)              4.1%  █
  Grade 1 (Mild DR)            8.2%  ██
  Grade 2 (Moderate DR)       76.4%  ██████████████████████
  Grade 3 (Severe DR)          9.1%  ██
  Grade 4 (Proliferative DR)   2.2%  

Heatmap saved to: heatmap.png
```

---

### Step 3 — Launch the web app

```bash
python app.py
```

Open **http://localhost:7860** in your browser.

Upload any fundus image → get:
- DR severity grade with confidence scores
- Grad-CAM heatmap highlighting the affected retinal regions

---

## 🔬 Preprocessing

Each image goes through **Ben Graham's contrast enhancement** before training and inference:

```python
img = cv2.addWeighted(
    img, 4,
    cv2.GaussianBlur(img, (0, 0), img_size // 30), -4,
    128
)
```

This amplifies retinal vessel structures and suppresses uneven lighting — critical for consistent performance across different fundus cameras.

---

## 📊 Results

| Metric                        | Value          |
|-------------------------------|----------------|
| Validation Accuracy           | ~82%           |
| Quadratic Weighted Kappa (QWK)| ~0.87          |
| Clinical screening threshold  | QWK > 0.85 ✅  |
| Inference time (CPU)          | ~1.2s / image  |
| Inference time (GPU)          | ~0.15s / image |

**Why QWK?** Standard accuracy treats all misclassifications equally. QWK penalises predictions far from the true grade more heavily — e.g. predicting Grade 4 for a Grade 0 patient is much worse than predicting Grade 1. This matches real clinical severity.

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
print(f"QWK: {kappa:.4f}")  # Aim for > 0.85
```

---

## ☁️ Deploy to Hugging Face Spaces (Free)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) — choose **Gradio** as the SDK.
2. Create a `requirements.txt`:
```
torch
torchvision
timm
albumentations
grad-cam
opencv-python-headless
gradio
scikit-learn
```
3. Push your code:
```bash
git init
git add app.py best_model.pth requirements.txt README.md
git commit -m "initial deployment"
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/dr-screener
git push
```

Your app goes live at `https://huggingface.co/spaces/YOUR_USERNAME/dr-screener` — shareable with anyone, no server needed.

---

## 🛠️ Tech Stack

| Category         | Tool / Library          | Purpose                                        |
|------------------|-------------------------|------------------------------------------------|
| Language         | Python 3.10+            | Primary programming language                   |
| Deep Learning    | PyTorch                 | Model training, inference, loss functions      |
| Model            | timm (EfficientNet-B3)  | Pretrained ImageNet backbone                   |
| Preprocessing    | OpenCV                  | Ben Graham contrast enhancement, image I/O     |
| Augmentation     | albumentations          | Flips, rotation, colour jitter, normalisation  |
| Data             | pandas, NumPy           | CSV handling, array operations                 |
| Explainability   | pytorch-grad-cam        | Grad-CAM heatmap generation                    |
| Evaluation       | scikit-learn            | Quadratic Weighted Kappa, stratified split     |
| Web UI           | Gradio                  | Interactive upload + prediction interface      |
| Deployment       | Hugging Face Spaces     | Free cloud hosting                             |
| Training GPU     | Kaggle Notebooks        | Free T4 GPU (30 hrs/week)                      |
| Version Control  | Git + GitHub            | Code management and sharing                    |
| Dataset          | IDRiD                   | 516 labelled Indian fundus images (CC BY 4.0)  |

---

## ⚠️ Limitations & Disclaimer

- This tool is for **screening assistance only** — not a substitute for clinical diagnosis.
- Performance may vary with fundus images from different camera models or patient populations.
- The model was trained on IDRiD (Indian patients, Nanded clinic) and may not generalise equally to other demographics.
- Always consult a qualified ophthalmologist for medical decisions.

---

## 🔭 Future Improvements

- [ ] Ensemble EfficientNet-B3 + B4 for higher QWK
- [ ] Test-Time Augmentation (TTA) for more robust predictions
- [ ] Ordinal regression loss for better grade ordering
- [ ] Multi-disease detection (glaucoma, macular degeneration)
- [ ] ONNX export for edge/mobile deployment
- [ ] Combine IDRiD + Messidor-2 + Zenodo for a larger training set

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [IDRiD Dataset](https://idrid.grand-challenge.org) — ISBI 2018 Challenge, Prasanna Kumar & team
- [timm](https://github.com/huggingface/pytorch-image-models) — Ross Wightman
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) — Jacob Gildenblat
- Ben Graham's fundus preprocessing technique (KAGGLE DR competition, 2015)

---

*Built to help bring specialist-level retinal screening to underserved communities — especially in rural India where ophthalmologists are scarce.*
