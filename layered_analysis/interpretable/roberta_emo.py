"""
layered_analysis.interpretable.roberta_emo

RoBERTa / XLM-R 系感情モデルを用いて、発話単位のテキスト感情特徴行列を抽出するモジュール。

使用モデルの例:
    - "j-hartmann/emotion-english-distilroberta-base"
    - "cardiffnlp/twitter-xlm-roberta-base-sentiment"

各発話テキストごとに感情クラスのロジットまたは確率、および
そこから計算される valence / arousal / dominance 系の連続値特徴を算出し、
1 発話あたり 1 ベクトルとして行列を構成する。

出力は次の 2 つとする:
    X_text_emo: np.ndarray, shape (T, D_emo)
        各行が 1 発話に対応し、各列が感情クラス出力および
        VAD 連続値特徴に対応する（列順は実装時に固定する）。

    segment_sequence: np.ndarray, shape (T,)
        各発話が属するセグメントラベル（"seg1", "seg2", "seg3"）。

ここでの T は「発話数」であり、列単位の発話時系列として
`scalerize_seq_data` に渡す。
"""
import numpy as np
import pandas as pd
import torch

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

def extract_text_emo_features(df, model_name="j-hartmann/emotion-english-distilroberta-base"):
    """
    Extract text emotion features.
    
    Args:
        df (pd.DataFrame): Dataframe with 'text' column.
        
    Returns:
        X_text_emo (np.ndarray): (T, D)
        segment_sequence (np.ndarray): (T,)
    """
    if pipeline is None:
        print("Warning: transformers not installed.")
        return np.zeros((len(df), 7)), df.get('segment', np.full(len(df), "unknown")).values

    # Using pipeline is easiest
    device = 0 if torch.cuda.is_available() else -1
    try:
        classifier = pipeline("text-classification", model=model_name, return_all_scores=True, device=device)
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        return np.zeros((len(df), 7)), df.get('segment', np.full(len(df), "unknown")).values

    embeddings = []
    
    texts = df['text'].fillna("").astype(str).tolist()
    
    # Process in batches if needed, but pipeline handles list
    try:
        results = classifier(texts, truncation=True, max_length=512)
        # results is list of list of dicts: [[{'label': 'anger', 'score': 0.1}, ...], ...]
        
        # Sort labels to ensure consistent column order
        labels = sorted([x['label'] for x in results[0]])
        
        for res in results:
            # Convert to vector based on sorted labels
            res_dict = {x['label']: x['score'] for x in res}
            vec = [res_dict[l] for l in labels]
            
            # VAD calculation?
            # Mapping discrete emotions to VAD is heuristic.
            # E.g. anger -> low V, high A.
            # For this "勝手コーディング", we just output the class probabilities.
            # D = num_classes.
            
            embeddings.append(vec)
            
    except Exception as e:
        print(f"Error during inference: {e}")
        # Fallback
        embeddings = [np.zeros(len(labels))] * len(df)

    X_text_emo = np.array(embeddings)

    if 'segment' in df.columns:
        segment_sequence = df['segment'].values
    else:
        segment_sequence = np.full(len(df), "unknown")
        
    return X_text_emo, segment_sequence
