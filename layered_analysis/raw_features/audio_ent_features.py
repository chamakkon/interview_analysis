"""
layered_analysis.raw_features.audio_ent_features

2 話者の音声特徴と発話区間情報から、発話単位の線形エントレインメント特徴行列を抽出するモジュール。

計算する特徴量およびライブラリ構成はローカルの `linear_ent.py` と同一とする。
`linear_ent.py` で定義されたエントレインメント関連特徴
（F0 差分、対数パワー差分、相関係数ベースの特徴、正規化距離指標など）を、
発話単位で算出する。

出力は次の 2 つとする:
    X_ent: np.ndarray, shape (T, D)
        各行が 1 発話に対応し、各列が linear_ent.py で定義された
        エントレインメント特徴量に対応する。

    segment_sequence: np.ndarray, shape (T,)
        各発話が属するセグメントラベル（"seg1", "seg2", "seg3"）。

ここでの T は「発話数」であり、列単位で
`scalerize_seq_data` に渡してスカラー化する。
"""
import numpy as np
import pandas as pd
import itertools

def extract_audio_ent_features(df):
    """
    Extract linear entrainment features (turn-level).
    Computes absolute difference of features between current turn and previous turn of the partner.
    """
    # Feature names definition (copied from linear_ent.py)
    nested_feature_names = ["F0final_sma", "pcm_loudness_sma", "F0final_sma_de", "pcm_loudness_sma_de",
                            [f"pcm_fftMag_mfcc_sma[{i}]" for i in range(15)],
                            [f"lspFreq_sma[{i}]" for i in range(8)],
                            [f"logMelFreqBand_sma[{i}]" for i in range(8)],
                            "jitterLocal_sma", "jitterDDP_sma", "shimmerLocal_sma"]
    feature_names = list(itertools.chain.from_iterable(elem if isinstance(elem, list) else [elem] for elem in nested_feature_names))
    columns = list(itertools.chain.from_iterable([[name+"_mean", name+"_median", name+"_std", name+"_percentile_1", name+"_percntile_99", name+"_range"] for name in feature_names]))

    X_ent_list = []
    
    # Iterate turns
    for i in range(len(df)):
        current_row = df.iloc[i]
        current_spk = current_row.get('speaker', '')
        
        # Find previous turn of OTHER speaker
        prev_row = None
        for j in range(i-1, -1, -1):
            if df.iloc[j].get('speaker', '') != current_spk:
                prev_row = df.iloc[j]
                break
        
        if prev_row is None:
            # No previous partner turn, fill 0
            diffs = [0.0] * len(columns)
        else:
            # Calculate absolute difference
            diffs = []
            for col in columns:
                if col in df.columns:
                    val_curr = float(current_row[col])
                    val_prev = float(prev_row[col])
                    if pd.isna(val_curr) or pd.isna(val_prev):
                        diffs.append(0.0)
                    else:
                        diffs.append(abs(val_curr - val_prev))
                else:
                    diffs.append(0.0)
                    
        X_ent_list.append(diffs)
        
    X_ent = np.array(X_ent_list)
    
    if 'segment' in df.columns:
        segment_sequence = df['segment'].values
    else:
        segment_sequence = np.full(len(df), "unknown")

    return X_ent, segment_sequence
