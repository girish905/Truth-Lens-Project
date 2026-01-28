import cv2
import numpy as np
from PIL import Image, ExifTags
from scipy.stats import entropy
import os

def integrity_report(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"Error": "Image could not be loaded."}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # -----------------------------
    # 1. Sharpness (Blur Detection)
    # -----------------------------
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_result = "Good" if sharpness > 150 else "Low (Possibly Blurry)"

    # -----------------------------
    # 2. Noise Analysis
    # -----------------------------
    noise = np.var(gray)
    noise_result = "Natural" if noise > 500 else "Artificial / Smoothed"

    # -----------------------------
    # 3. Resolution Check
    # -----------------------------
    resolution_result = f"{w} x {h}"

    # -----------------------------
    # 4. JPEG Compression Artifacts
    # -----------------------------
    dct = cv2.dct(np.float32(gray) / 255.0)
    high_freq_energy = np.mean(np.abs(dct[10:, 10:]))
    compression_result = "High Compression Detected" if high_freq_energy < 0.01 else "Low Compression"

    # -----------------------------
    # 5. Color Channel Consistency
    # -----------------------------
    b, g, r = cv2.split(img)
    channel_diff = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))
    color_result = "Normal" if channel_diff < 50 else "Unusual Channel Differences"

    # -----------------------------
    # 6. Edge Density (Tampering Clue)
    # -----------------------------
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (w * h)
    edge_result = "Normal" if 0.01 < edge_density < 0.15 else "Unusual Edge Distribution"

    # -----------------------------
    # 7. Histogram Uniformity
    # -----------------------------
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / hist.sum()
    hist_entropy = entropy(hist_norm)
    histogram_result = "Natural Distribution" if hist_entropy > 4.5 else "Flat / Edited Histogram"

    # -----------------------------
    # 8. EXIF Metadata Check
    # -----------------------------
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if exif_data:
            exif_result = "Metadata Present"
        else:
            exif_result = "No Metadata (Possibly Stripped)"
    except:
        exif_result = "Metadata Not Available"

    # -----------------------------
    # 9. File Size vs Resolution
    # -----------------------------
    file_size_kb = os.path.getsize(image_path) / 1024
    pixels = w * h
    size_ratio = file_size_kb / pixels
    size_result = "Unusual Compression Ratio" if size_ratio < 0.0005 else "Normal"

    return {
        "Sharpness": sharpness_result,
        "Noise Pattern": noise_result,
        "Resolution": resolution_result,
        "Compression Artifacts": compression_result,
        "Color Channel Consistency": color_result,
        "Edge Density": edge_result,
        "Histogram Entropy": histogram_result,
        "EXIF Metadata": exif_result,
        "File Size Analysis": size_result
    }
