import cv2
import numpy as np
import os

def generate_heatmap(image_path, filename):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Laplacian(gray, cv2.CV_64F)
    heat = cv2.normalize(np.abs(edges),None,0,255,cv2.NORM_MINMAX)

    heatmap = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img,0.6,heatmap,0.4,0)

    out = f"static/heatmaps/{filename}"
    cv2.imwrite(out, overlay)
    return out
