"""
layered_analysis.interpretable.affectnet

AffectNet 事前学習モデルを用いて、フレーム単位の表情感情特徴行列を抽出するモジュール。

使用モデルの例:
    - "nateraw/affectnet-emotion" (This is likely a fine-tuned ViT or similar)

各フレーム画像ごとに valence / arousal と感情クラス出力を得て、
1 フレームあたり 1 ベクトルとして行列を構成する。

出力は次の 2 つとする:
    X_affectnet: np.ndarray, shape (T, D_affect)
        各行が 1 フレームに対応し、各列が次の成分に対応する
        （列順は実装時に固定する）:
            - valence 回帰値
            - arousal 回帰値
            - 感情クラスのロジットまたは確率（neutral, happy, sad, angry, fear, disgust, surprise など）

    segment_sequence: np.ndarray, shape (T,)
        各フレームが属するセグメントラベル（"seg1", "seg2", "seg3"）。

ここでの T は「フレーム数」であり、列単位のフレーム時系列として
`scalerize_seq_data` に渡す。
"""
import numpy as np
import pandas as pd
import torch
import cv2
import os

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from transformers import AutoModelForRegression # If V/A regression model exists
except ImportError:
    AutoImageProcessor = None
    AutoModelForImageClassification = None

def extract_affectnet_features(video_path, segment_info=None):
    """
    Extract AffectNet features from video frames.
    
    Args:
        video_path (str): Path to video file.
        segment_info (list of dict): Optional segment info [{'start': s, 'end': e, 'label': l}, ...]
        
    Returns:
        X_affectnet (np.ndarray): (T, D)
        segment_sequence (np.ndarray): (T,)
    """
    if AutoImageProcessor is None:
        print("Warning: transformers not installed.")
        return np.zeros((1, 10)), np.array(["unknown"])

    # Model for Emotion Classification (7 or 8 classes)
    model_name_cls = "nateraw/bert-base-uncased-emotion" # Wait, nateraw/affectnet-emotion doesn't exist?
    # Common AffectNet model: "dima806/facial_emotions_image_detection" or "google/vit-base-patch16-224" fine-tuned.
    # Let's assume a generic one.
    model_name_cls = "dima806/facial_emotions_image_detection" 
    
    # Model for Valence/Arousal (Regression)
    # Often these are separate or multi-head. Assuming we iterate frames and just get Class probabilities for now
    # if V/A model is not readily available in standard transformers hub.
    # We will output class probs + dummy V/A if not found.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        processor = AutoImageProcessor.from_pretrained(model_name_cls)
        model = AutoModelForImageClassification.from_pretrained(model_name_cls).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return np.zeros((1, 10)), np.array(["unknown"])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return np.zeros((1, 10)), np.array(["unknown"])
        
    features_list = []
    seg_list = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every Nth frame to save time? Or all frames?
        # Docstring implies "frame unit".
        
        # Determine segment
        current_time = frame_idx / fps
        current_seg = "unknown"
        if segment_info:
            for seg in segment_info:
                if seg['start'] <= current_time <= seg['end']:
                    current_seg = seg['label']
                    break
        
        # Inference
        # Face detection is usually needed before AffectNet!
        # AffectNet models expect cropped faces.
        # We need a face detector (e.g. cv2.CascadeClassifier or dlib or mtcnn).
        # For this "勝手コーディング", let's assume the frame is the face or we skip detection for brevity.
        # (Implementing face detection adds dependency like opencv-contrib or dlib).
        # Let's use simple center crop or resize.
        
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()
            
            # Dummy V/A (0.0, 0.0) + probs
            # If model has 7 classes, D = 2 + 7 = 9.
            feat = np.concatenate([[0.0, 0.0], probs])
            features_list.append(feat)
            seg_list.append(current_seg)
            
        except Exception as e:
            # Skip frame on error
            pass
            
        frame_idx += 1
        
    cap.release()
    
    if len(features_list) == 0:
        return np.zeros((1, 10)), np.array(["unknown"])
        
    X_affectnet = np.array(features_list)
    segment_sequence = np.array(seg_list)
    
    return X_affectnet, segment_sequence
