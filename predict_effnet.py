# predict_effnet.py
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model_efficientnet import EfficientNetB0
import os
import medmnist
from medmnist import INFO

IMAGE_SIZE = 224
CKPT_PATH = os.path.join("outputs", "best_effnet.pth")

# same preprocessing as training (evaluation)
_eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(checkpoint_path=CKPT_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train model first with train_effnet.py")
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get('num_classes', None)
    if num_classes is None:
        # fallback
        info = ckpt.get('info', None)
        if info:
            num_classes = len(info['label'])
        else:
            num_classes = 5
    model = EfficientNetB0(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    info = ckpt.get('info', None)
    return model, device, info

def predict_image(img_pil, model, device, info=None):
    # img_pil: PIL image
    x = _eval_transform(img_pil).unsqueeze(0).to(device).float()
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred = int(outputs.argmax(1).cpu().numpy()[0])

    label_names = None
    if info is None:
        try:
            info = INFO['retinamnist']
        except Exception:
            info = None
    if info and 'label' in info:
        label_desc = info.get('label', None)
        if isinstance(label_desc, dict):
            # medmnist often uses dict mapping '0'->"class"
            label_names = [label_desc.get(str(i), str(i)) for i in range(len(probs))]
        elif isinstance(label_desc, list):
            label_names = label_desc

    return pred, float(probs[pred]), probs, label_names
