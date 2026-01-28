from flask import Flask, request, render_template, url_for
import tensorflow as tf
from utils.forensic import build_tensor, forensic_reasons, generate_heatmap
from utils.integrity import integrity_report   # NEW
from utils.video import analyze_video, extract_thumbnail  # NEW for video detection
import os, time, uuid
import numpy as np
import mimetypes  # NEW for file type detection

app = Flask(__name__)
model = tf.keras.models.load_model("truthlens_model.h5")

UPLOAD_DIR = "static/uploads"
HEATMAP_DIR = "static/heatmap"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)

# Global storage for reports (in production, use a database)
reports = {}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")
        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        # Detect file type
        mime_type, _ = mimetypes.guess_type(file.filename)
        is_video = mime_type and mime_type.startswith('video/')

        if is_video:
            # Handle video
            filename = f"{int(time.time())}.mp4"
            filepath = os.path.join(UPLOAD_DIR, filename)
            file.save(filepath)

            # Analyze video
            verdict, score, reasons = analyze_video(filepath)

            # If no anomalies detected and verdict is Real, set high confidence
            if verdict == "Real" and score == 0:
                score = 100

            # Confidence band
            if score >= 90:
                confidence = "Highly Confident"
            elif score >= 70:
                confidence = "Likely"
            else:
                confidence = "Low Confidence"

            # Warning for low confidence
            warning = None
            if confidence == "Low Confidence":
                warning = "⚠ This result has low confidence. Human verification is recommended."

            # Extract thumbnail
            thumbnail_filename = f"thumb_{filename}.jpg"
            thumbnail_path = os.path.join(UPLOAD_DIR, thumbnail_filename)
            extract_thumbnail(filepath, thumbnail_path)

            # Generate heatmap from thumbnail
            heatmap_filename = f"heatmap_{int(time.time())}.jpg"
            heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
            generate_heatmap(thumbnail_path, heatmap_path)

            # Store report data
            report_id = str(uuid.uuid4())
            reports[report_id] = {
                'result': verdict,
                'score': score,
                'confidence': confidence,
                'warning': warning,
                'reasons': reasons,
                'video_path': f"/{filepath}",
                'thumbnail_path': f"/{thumbnail_path}",
                'heatmap_path': f"/{heatmap_path}"
            }

            # Generate shareable URL
            share_url = url_for('view_report', report_id=report_id, _external=True)

            return render_template(
                "index.html",
                result=verdict,
                score=score,
                confidence=confidence,
                warning=warning,
                reasons=reasons,
                video_path=filepath,
                thumbnail_path=thumbnail_path,
                share_url=share_url
            )
        else:
            # Handle image (existing logic)
            filename = f"{int(time.time())}.jpg"
            filepath = os.path.join(UPLOAD_DIR, filename)
            file.save(filepath)

            # Build tensor
            tensor = build_tensor(filepath)

            # Predict
            pred = float(model.predict(tensor[np.newaxis, ...])[0][0])

            # Verdict + score logic (FIXED)
            if pred > 0.72:
                verdict = "Fake"
                score = round(pred * 100, 2)
            elif pred < 0.35:
                verdict = "Real"
                score = round((1 - pred) * 100, 2)
            else:
                verdict = "Uncertain"
                score = round(max(pred, 1 - pred) * 100, 2)

            # Confidence band
            if score >= 90:
                confidence = "Highly Confident"
            elif score >= 70:
                confidence = "Likely"
            else:
                confidence = "Low Confidence"

            # Warning for low confidence
            warning = None
            if confidence == "Low Confidence":
                warning = "⚠ This result has low confidence. Human verification is recommended."

            # Forensic explanations (your existing logic)
            reasons = forensic_reasons(filepath)

            # Integrity report (NEW)
            try:
                integrity = integrity_report(filepath)
            except Exception as e:
                integrity = {"Error": f"Integrity analysis failed: {str(e)}"}

            # Generate heatmap
            heatmap_filename = f"heatmap_{filename}"
            heatmap_path = os.path.join(HEATMAP_DIR, heatmap_filename)
            generate_heatmap(filepath, heatmap_path)

            # Store report data
            report_id = str(uuid.uuid4())
            reports[report_id] = {
                'result': verdict,
                'score': score,
                'confidence': confidence,
                'warning': warning,
                'reasons': reasons,
                'integrity': integrity,
                'image_path': f"/{filepath}",
                'heatmap_path': f"/{heatmap_path}"
            }

            # Generate shareable URL
            share_url = url_for('view_report', report_id=report_id, _external=True)

            return render_template(
                "index.html",
                result=verdict,
                score=score,
                confidence=confidence,
                warning=warning,
                reasons=reasons,
                integrity=integrity,
                image_path=filepath,
                heatmap=heatmap_path,
                share_url=share_url
            )

    return render_template("index.html")

@app.route("/report/<report_id>")
def view_report(report_id):
    if report_id not in reports:
        return "Report not found", 404

    report = reports[report_id]
    return render_template("report.html", **report)

if __name__ == "__main__":
    app.run(debug=True)
