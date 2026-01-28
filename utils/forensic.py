import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# -------------------------
# ELA MAP
# -------------------------
def ela_map(path, quality=90):
    original = Image.open(path).convert("RGB")
    buffer = BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer)

    diff = np.abs(np.array(original) - np.array(recompressed))
    diff = np.mean(diff, axis=2)

    return cv2.resize(diff, (224, 224)).astype(np.uint8)

# -------------------------
# NOISE MAP
# -------------------------
def noise_map(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = cv2.absdiff(gray, blur)

    return cv2.resize(noise, (224, 224))

# -------------------------
# CNN INPUT TENSOR
# -------------------------
def build_tensor(path):
    ela = ela_map(path)
    noise = noise_map(path)

    # normalize per-channel (IMPORTANT for stability)
    ela = ela / (np.max(ela) + 1e-6)
    noise = noise / (np.max(noise) + 1e-6)

    return np.stack([ela, noise, noise], axis=-1).astype(np.float32)

# -------------------------
# FORENSIC REASONS (IMPROVED)
# -------------------------
def forensic_reasons(path):
    ela = ela_map(path)
    noise = noise_map(path)

    ela_mean = float(np.mean(ela))
    noise_var = float(np.var(noise))

    reasons = []

    # ELA interpretation (soft thresholds)
    if ela_mean > 22:
        reasons.append(
            "Strong compression inconsistencies detected, often caused by digital manipulation or AI synthesis."
        )
    elif ela_mean > 14:
        reasons.append(
            "Moderate compression variation detected. This may indicate editing or recompression."
        )
    else:
        reasons.append(
            "Uniform compression pattern consistent with natural camera images."
        )

    # Noise interpretation (soft thresholds)
    if noise_var < 25:
        reasons.append(
            "Very low noise variance detected, which is commonly associated with AI-generated or heavily smoothed images."
        )
    elif noise_var < 60:
        reasons.append(
            "Reduced noise variance detected. Possible mild processing or enhancement."
        )
    else:
        reasons.append(
            "Natural sensor noise pattern consistent with real-world photography."
        )

    return reasons

# -------------------------
# HEATMAP GENERATION (NEW)
# -------------------------
def generate_heatmap(path, save_path):
    ela = ela_map(path)
    heat = cv2.normalize(ela, None, 0, 255, cv2.NORM_MINMAX)
    heat = heat.astype(np.uint8)

    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    original = cv2.imread(path)
    original = cv2.resize(original, (224, 224))

    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)
