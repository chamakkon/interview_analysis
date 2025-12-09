"""
layered_analysis.raw_features.word_count_features

テキストと発話区間の時間情報から、発話単位の speech rate 系特徴行列を抽出するモジュール。

各発話区間について、時間あたりの発話量を表す特徴を計算する。

出力は次の 2 つとする:
    X_speech: np.ndarray, shape (T, D)
        各行が 1 発話に対応し、各列が次の特徴に対応する
        （列順は実装時に固定する）:
            - speech_rate_token:
                発話内のトークン数（形態素または単語）を
                発話継続時間 [秒] で割った値
            - speech_rate_content_token:
                発話内の内容語トークン数を発話継続時間 [秒] で割った値
            - speech_rate_char:
                発話内の文字数を発話継続時間 [秒] で割った値
            - segment_duration:
                発話継続時間 [秒]

    segment_sequence: np.ndarray, shape (T,)
        各発話が属するセグメントラベル（"seg1", "seg2", "seg3"）。

ここでの T は「発話数」であり、各列を
発話時系列として `scalerize_seq_data` に渡す。
"""

import numpy as np
import pandas as pd

def extract_word_count_features(df):
    """
    Extract word count features from transcript dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with columns 'text', 'start', 'end', 'segment'.
    
    Returns:
        X_speech (np.ndarray): Shape (T, D)
        segment_sequence (np.ndarray): Shape (T,)
    """
    feature_list = []
    segment_list = []

    for _, row in df.iterrows():
        text = str(row.get('text', ''))
        start = float(row.get('start', 0))
        end = float(row.get('end', 0))
        segment = row.get('segment', 'unknown')
        
        duration = end - start
        if duration <= 0:
            duration = 1e-6 # Avoid division by zero

        # Simple tokenization (whitespace)
        tokens = text.strip().split()
        num_tokens = len(tokens)
        num_chars = len(text)
        
        # Approximate content tokens: words with length >= 4
        num_content_tokens = sum(1 for t in tokens if len(t) >= 4)

        speech_rate_token = num_tokens / duration
        speech_rate_content = num_content_tokens / duration
        speech_rate_char = num_chars / duration
        
        feature_list.append([speech_rate_token, speech_rate_content, speech_rate_char, duration])
        segment_list.append(segment)

    X_speech = np.array(feature_list)
    segment_sequence = np.array(segment_list)
    
    return X_speech, segment_sequence
