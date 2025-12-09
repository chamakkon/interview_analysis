"""
layered_analysis.raw_features.action_units

動画フレーム列から、フレーム単位の Action Unit (AU) および関連表情特徴行列を抽出するモジュール。

OpenFace 2.0 互換の出力を想定し、エラーが発生しにくい範囲で
安定して推定できる AU 系特徴をできるだけ多く採用する。

出力は次の 2 つとする:
    X_au: np.ndarray, shape (T, D)
        各行が 1 フレームに対応し、各列が次の特徴に対応する
        （列順は実装時に固定する）:
            - 連続値 AU 強度:
                AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r,
                AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r,
                AU20_r, AU23_r, AU24_r, AU25_r, AU26_r
            - 2値 AU 発火フラグ:
                AU01_c, AU02_c, AU04_c, AU05_c, AU06_c, AU07_c,
                AU09_c, AU10_c, AU12_c, AU14_c, AU15_c, AU17_c,
                AU20_c, AU23_c, AU24_c, AU25_c, AU26_c
            - 顔姿勢:
                pose_rx, pose_ry, pose_rz
            - 目線方向:
                gaze_angle_x, gaze_angle_y

    segment_sequence: np.ndarray, shape (T,)
        各フレームが属するセグメントラベル（"seg1", "seg2", "seg3"）。

ここでの T は「フレーム数」であり、各列を
フレーム時系列として `scalerize_seq_data` に渡す。
"""

import pandas as pd
import numpy as np

def extract_action_units(openface_df):
    """
    Extract Action Unit features from OpenFace output.

    Args:
        openface_df (pd.DataFrame): OpenFace output dataframe.
    
    Returns:
        X_au (np.ndarray): Shape (T, D)
        segment_sequence (np.ndarray): Shape (T,)
    """
    # Define columns based on OpenFace 2.0 standard
    # Note: Column names are case-sensitive and depend on OpenFace version.
    # Assuming standard headers from OpenFace 2.x
    
    # Regression (intensity)
    au_r_cols = [f"AU{i:02d}_r" for i in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,24,25,26]]
    
    # Classification (presence)
    au_c_cols = [f"AU{i:02d}_c" for i in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,24,25,26]]
    
    # Pose: pitch, yaw, roll (radians). OpenFace uses pose_Rx, pose_Ry, pose_Rz
    pose_cols = ["pose_Rx", "pose_Ry", "pose_Rz"]
    
    # Gaze
    gaze_cols = ["gaze_angle_x", "gaze_angle_y"]
    
    target_cols = au_r_cols + au_c_cols + pose_cols + gaze_cols
    
    # Ensure columns exist, fill with 0 if missing
    for col in target_cols:
        if col not in openface_df.columns:
            # Simple fallback or warning could go here. For now, 0-fill.
            openface_df[col] = 0.0
            
    X_au = openface_df[target_cols].values
    
    # Handle segment sequence
    # If 'segment' is present (pre-processed), use it. Else fill with 'unknown'.
    if "segment" in openface_df.columns:
        segment_sequence = openface_df["segment"].fillna("unknown").astype(str).values
    else:
        segment_sequence = np.full(len(openface_df), "unknown")
        
    return X_au, segment_sequence
