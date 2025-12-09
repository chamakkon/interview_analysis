"""
layered_analysis.embedding_vector.openface_vector

OpenFace 2.0 の出力から、フレーム単位の表情埋め込みベクトル行列を構築するモジュール。

OpenFace 出力 CSV から埋め込みとして扱う列を選択し、
1 フレームあたり 1 ベクトルとして行列を構成する。

出力は次の 2 つとする:
    X_openface: np.ndarray, shape (T, D_openface)
        各行が 1 フレームに対応し、各列が採用する OpenFace 由来の
        埋め込み次元（顔ランドマークから導出した形状特徴など）に対応する。

    segment_sequence: np.ndarray, shape (T,)
        各フレームが属するセグメントラベル（"seg1", "seg2", "seg3"）。

ここでの T は「フレーム数」であり、列単位のフレーム時系列として
`scalerize_seq_data` に渡す。
"""

import pandas as pd
import numpy as np

def extract_openface_vector(openface_df):
    """
    Extract embedding-like vector from OpenFace output.
    Uses non-rigid shape parameters (p_0 to p_33) as the embedding.

    Args:
        openface_df (pd.DataFrame): OpenFace output dataframe.
    
    Returns:
        X_openface (np.ndarray): Shape (T, D)
        segment_sequence (np.ndarray): Shape (T,)
    """
    # OpenFace outputs non-rigid shape parameters p_0 through p_33.
    # These represent the face shape deformation due to expression/identity,
    # often used as a low-dimensional face representation.
    p_cols = [f"p_{i}" for i in range(34)]
    
    # Ensure columns exist
    for col in p_cols:
        if col not in openface_df.columns:
            openface_df[col] = 0.0
            
    X_openface = openface_df[p_cols].values
    
    if "segment" in openface_df.columns:
        segment_sequence = openface_df["segment"].fillna("unknown").astype(str).values
    else:
        segment_sequence = np.full(len(openface_df), "unknown")
        
    return X_openface, segment_sequence
