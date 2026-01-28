import cv2
import numpy as np
import os

def analyze_video(video_path):
    """
    Analyze video for signs of AI generation or fakeness based on slow frames and slow motion.
    Returns: verdict ('Real' or 'Fake'), score (0-100), reasons (list of strings)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error", 0, ["Unable to open video file."]

    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sample frames (e.g., every 20th frame to speed up)
    sample_rate = max(1, total_frames // 100)  # Increased to 100 frames for better motion analysis

    prev_frame = None
    duplicate_count = 0
    motion_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % sample_rate != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        if prev_frame is not None:
            # Check for duplicate frames (slow frames)
            diff = cv2.absdiff(prev_frame, gray)
            mean_diff = np.mean(diff)
            if mean_diff < 5:  # Threshold for duplicate
                duplicate_count += 1

            # Optical flow for motion
            flow = cv2.calcOpticalFlowFarneback(prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_motion = np.mean(mag)
            motion_values.append(avg_motion)

        prev_frame = gray

    cap.release()

    if not frames:
        return "Error", 0, ["No frames extracted from video."]

    # Calculate metrics
    duplicate_ratio = duplicate_count / len(frames) if frames else 0
    avg_motion = np.mean(motion_values) if motion_values else 0
    motion_variance = np.var(motion_values) if motion_values else 0  # New: Variance in motion for inconsistent slow movements

    # Rules for fake detection
    score = 0
    reasons = []

    # Slow frames: High duplicate ratio indicates fake (e.g., AI generated with repeated frames)
    if duplicate_ratio > 0.3:
        score += 50
        reasons.append(f"High duplicate frame ratio ({duplicate_ratio:.2%}) detected, common in AI-generated videos.")
    elif duplicate_ratio > 0.1:
        score += 20
        reasons.append(f"Moderate duplicate frames ({duplicate_ratio:.2%}), possible manipulation.")

    # Slow motion: Low average motion indicates slow motion or fake
    if avg_motion < 0.3:  # Lower threshold for high sensitivity to slow motion
        score += 50  # Increased weight for strong slow motion indicator
        reasons.append(f"Very low average motion ({avg_motion:.2f}) detected, highly indicative of slow motion or AI synthesis.")
    elif avg_motion < 0.7:  # Adjusted threshold for moderate slow motion
        score += 25  # Increased weight
        reasons.append(f"Low average motion ({avg_motion:.2f}) detected, indicative of slow motion or AI synthesis.")
    elif avg_motion < 1.2:  # Slightly higher for reduced motion
        score += 15
        reasons.append(f"Reduced motion ({avg_motion:.2f}), may suggest processing.")

    # High variance in slow motion: Inconsistent slow movements may indicate AI synthesis
    if motion_variance > 0.1 and avg_motion < 1.0:  # Only apply if motion is generally low
        score += 15
        reasons.append(f"High variance in slow motion ({motion_variance:.2f}) detected, potentially indicating AI manipulation.")

    # Additional rules based on FPS
    if fps < 24:
        score += 20
        reasons.append(f"Low frame rate ({fps:.1f} FPS), often associated with fake videos.")
    
    # Cap score at 100
    score = min(score, 100)

    # Verdict
    if score > 60:
        verdict = "Fake"
    elif score < 30:
        verdict = "Real"
    else:
        verdict = "Uncertain"

    if not reasons:
        reasons.append("No significant anomalies detected.")

    return verdict, round(score, 2), reasons

def extract_thumbnail(video_path, save_path):
    """
    Extract a thumbnail from the video for display.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    # Get middle frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_path, frame)
    cap.release()
    return ret
