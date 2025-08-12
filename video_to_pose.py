"""
Video → Pose Data (2D) quick-start

What this does
- Reads a local 2D video (MP4/MOV)
- Runs MediaPipe Pose on every frame
- Saves:
  1) poses.json  (per-frame keypoints + confidences)
  2) poses.csv   (tidy table for spreadsheets)
  3) overlay.mp4 (skeletal overlay for quality check)

Requirements (install locally first):
  pip install opencv-python mediapipe numpy pandas tqdm

Usage:
  python video_to_pose.py --video /path/to/dancers.mp4 --outdir ./out

Tip: use a clear, well-lit video for better results.
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("MediaPipe is required. Install with: pip install mediapipe")

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 33 landmark names from MediaPipe Pose (world_landmarks differ; we use image coords here)
LANDMARK_NAMES = [
    'nose','left_eye_inner','left_eye','left_eye_outer','right_eye_inner','right_eye','right_eye_outer',
    'left_ear','right_ear','mouth_left','mouth_right','left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_pinky','right_pinky','left_index','right_index','left_thumb','right_thumb',
    'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_heel','right_heel',
    'left_foot_index','right_foot_index'
]

@dataclass
class Landmark:
    name: str
    x: float
    y: float
    z: float
    visibility: float


def draw_skeleton(frame, landmarks, visibility_thresh=0.5):
    h, w = frame.shape[:2]
    # Convert normalized coords to pixel coords
    pts = {}
    for lm in landmarks:
        if lm.visibility < visibility_thresh:
            continue
        pts[lm.name] = (int(lm.x * w), int(lm.y * h))

    # Simple connections (subset of full MediaPipe connections for clarity)
    pairs = [
        ('left_shoulder','right_shoulder'),
        ('left_shoulder','left_elbow'),('left_elbow','left_wrist'),
        ('right_shoulder','right_elbow'),('right_elbow','right_wrist'),
        ('left_shoulder','left_hip'),('right_shoulder','right_hip'),
        ('left_hip','right_hip'),
        ('left_hip','left_knee'),('left_knee','left_ankle'),
        ('right_hip','right_knee'),('right_knee','right_ankle'),
        ('nose','left_eye'),('nose','right_eye')
    ]

    # Draw joints
    for p in pts.values():
        cv2.circle(frame, p, 3, (255, 255, 255), -1)

    # Draw bones
    for a,b in pairs:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)

    return frame


def process_video(video_path, outdir, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    os.makedirs(outdir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    overlay_path = os.path.join(outdir, 'overlay.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

    frames_data = []  # for JSON
    rows = []          # for CSV

    with mp_pose.Pose(
        model_complexity=model_complexity,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pbar = tqdm(total=total if total>0 else None, desc='Processing')
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            landmarks_list = []
            if result.pose_landmarks:
                for i, lm in enumerate(result.pose_landmarks.landmark):
                    name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f'lm_{i}'
                    landmarks_list.append(Landmark(name, lm.x, lm.y, lm.z, lm.visibility))

                # Draw overlay
                draw_skeleton(frame, landmarks_list)

                # Collect rows for CSV
                row = {'frame': frame_idx}
                for lm in landmarks_list:
                    row[f'{lm.name}_x'] = lm.x
                    row[f'{lm.name}_y'] = lm.y
                    row[f'{lm.name}_z'] = lm.z
                    row[f'{lm.name}_v'] = lm.visibility
                rows.append(row)

            # Append JSON entry (even if empty landmarks)
            frames_data.append({
                'frame': frame_idx,
                'timestamp_sec': frame_idx / fps,
                'landmarks': [asdict(lm) for lm in landmarks_list]
            })

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    writer.release()

    # Save JSON
    json_path = os.path.join(outdir, 'poses.json')
    with open(json_path, 'w') as f:
        json.dump({
            'video': os.path.basename(video_path),
            'fps': fps,
            'width': width,
            'height': height,
            'frames': frames_data
        }, f)

    # Save CSV (aligned columns)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, 'poses.csv')
    df.to_csv(csv_path, index=False)

    print(f"Saved overlay: {overlay_path}")
    print(f"Saved JSON:    {json_path}")
    print(f"Saved CSV:     {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path to input video (MP4/MOV/etc.)')
    parser.add_argument('--outdir', default='./out', help='Where outputs will be written')
    parser.add_argument('--model_complexity', type=int, default=1, choices=[0,1,2], help='0=fast, 2=accurate')
    parser.add_argument('--min_det', type=float, default=0.5, help='Min detection confidence')
    parser.add_argument('--min_track', type=float, default=0.5, help='Min tracking confidence')
    args = parser.parse_args()

    process_video(
        video_path=args.video,
        outdir=args.outdir,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_track,
    )

if __name__ == '__main__':
    main()

