# ---------------- train_effnet.py ----------------
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import medmnist
from medmnist import INFO
from model_efficientnet import EfficientNetB0

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
DATASET = 'retinamnist'
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
LR = 2e-4
NUM_WORKERS = 0   # Windows must use 0
OUT_DIR = "outputs"
CKPT_NAME = "best_effnet.pth"
PRETRAINED = True

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# DATA LOADING
# -------------------------------------------------
def get_dataloaders(dataset_name=DATASET):
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])

    # Training transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Evaluation transforms
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = DataClass(split="train", transform=train_transform, download=True)
    val_ds   = DataClass(split="val",   transform=eval_transform,  download=True)
    test_ds  = DataClass(split="test",  transform=eval_transform,  download=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, info


# -------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds = []
    all_targets = []

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device).float()
        labels = labels.squeeze().long().to(device)  # FIXED: MedMNIST labels shape (N,1)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        all_preds.extend(outputs.argmax(1).cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    return np.mean(losses), acc


# -------------------------------------------------
# VALIDATION FUNCTION
# -------------------------------------------------
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    targets = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device).float()
            labels = labels.squeeze().long().to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            losses.append(loss.item())
            preds.extend(outputs.argmax(1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    acc = accuracy_score(targets, preds)
    return np.mean(losses), acc


# -------------------------------------------------
# MAIN TRAINING LOOP
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, info = get_dataloaders()
    num_classes = len(info["label"])

    model = EfficientNetB0(num_classes=num_classes, pretrained=PRETRAINED).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_val_acc = 0.0

    # Training
    for epoch in range(1, EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_acc)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Time: {(time.time()-start):.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "info": info
            }, os.path.join(OUT_DIR, CKPT_NAME))
            print("✔ Saved BEST model\n")

    # Final Test Accuracy
    ckpt_path = os.path.join(OUT_DIR, CKPT_NAME)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print("\n===================================================")
    print(f" FINAL TEST ACCURACY: {test_acc:.4f}")
    print("===================================================")


if __name__ == "__main__":
    main()
