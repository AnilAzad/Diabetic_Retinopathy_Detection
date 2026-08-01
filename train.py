# ─────────────────────────────────────────────
#  train.py  —  Train the DR detection model
#  Run: python train.py
# ─────────────────────────────────────────────

# 1. IMPORTS
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

# ─────────────────────────────────────────────
# 2. CONFIG  (change these if needed)
# ─────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS      = 1
BATCH_SIZE  = 16
LR          = 3e-4
IMG_SIZE    = 380
TRAIN_CSV   = "/Users/anilkumarazad/MyFiles/Project/Diabetic Retinopathy Detection/Data/train_1.csv"
TRAIN_IMGS  = "/Users/anilkumarazad/MyFiles/Project/Diabetic Retinopathy Detection/Data/train_images"

print(f"Using device: {DEVICE}")

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
# 4. AUGMENTATION PIPELINES
# ─────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# ─────────────────────────────────────────────
# 5. DATASET CLASS
# ─────────────────────────────────────────────
class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["id_code"] + ".png")
        img      = preprocess_fundus(img_path)

        if self.transform:
            img = self.transform(image=img)["image"]

        label = torch.tensor(row["diagnosis"], dtype=torch.long)
        return img, label

# ─────────────────────────────────────────────
# 6. MODEL
# ─────────────────────────────────────────────
class RetinopathyModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features  # 1536 for B3
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
# 7. TRAINING FUNCTIONS
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (preds.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds        = model(imgs)
            loss         = criterion(preds, labels)
            total_loss  += loss.item()
            correct     += (preds.argmax(1) == labels).sum().item()
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return total_loss / len(loader), correct / len(loader.dataset), kappa

# ─────────────────────────────────────────────
# 8. MAIN TRAINING BLOCK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Load CSV
    df = pd.read_csv(TRAIN_CSV)
    print(f"Dataset size: {len(df)} images")
    print(f"Grade distribution:\n{df['diagnosis'].value_counts().sort_index()}\n")

    # Train / validation split (stratified)
    train_df, val_df = train_test_split(
        df, test_size=0.15, stratify=df["diagnosis"], random_state=42
    )

    # DataLoaders
    train_ds     = RetinopathyDataset(train_df, TRAIN_IMGS, transform=train_transform)
    val_ds       = RetinopathyDataset(val_df,   TRAIN_IMGS, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Class weights to handle imbalance
    class_counts  = df["diagnosis"].value_counts().sort_index().values
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(DEVICE)

    # Model, loss, optimizer, scheduler
    model     = RetinopathyModel(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    best_kappa = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc          = train_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc, val_kappa = val_epoch(model, val_loader, criterion)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.3f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.3f}  Acc: {val_acc:.3f}  QWK: {val_kappa:.4f}"
        )

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Saved best model  (QWK: {val_kappa:.4f})\n")

    print(f"\nTraining complete. Best QWK: {best_kappa:.4f}")
    print("Model saved as: best_model.pth")