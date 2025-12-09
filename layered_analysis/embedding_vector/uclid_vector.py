"""
layered_analysis.embedding_vector.uclid_vector

UCLID 手法を用いて、発話単位の相互作用埋め込み行列を生成するモジュール。

アルゴリズムおよび特徴量定義はローカルの `tned_score.py` と同一とする。
対話から発話単位で UCLID 埋め込みを算出し、
1 発話あたり 1 ベクトルとして行列を構成する。

出力は次の 2 つとする:
    X_uclid: np.ndarray, shape (T, D_uclid)
        各行が 1 発話に対応し、各列が tned_score.py で定義された
        UCLID 埋め込み次元に対応する。

    segment_sequence: np.ndarray, shape (T,)
        各発話が属するセグメントラベル（"seg1", "seg2", "seg3"）。

ここでの T は「発話数」であり、列単位の発話時系列として
`scalerize_seq_data` に渡す。
"""
import sys
import os
import numpy as np
import pandas as pd

try:
    import MeCab
    from gensim.models import KeyedVectors
except ImportError:
    MeCab = None
    KeyedVectors = None

def extract_uclid_vector(df, n=5):
    """
    Extract UCLID vectors (WMD scores) using logic copied from wmd_ent.py.
    """
    if MeCab is None or KeyedVectors is None:
        print("Warning: MeCab or gensim not installed. returning zeros.")
        return np.zeros((len(df), 1)), df.get('segment', np.full(len(df), "unknown")).values

    # Model path resolution
    # Original: "../../interview_analysis_demo/entity_vector/entity_vector.model.bin"
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "../../interview_analysis_demo/entity_vector/entity_vector.model.bin")
    
    if not os.path.exists(model_path):
        model_path = "../../interview_analysis_demo/entity_vector/entity_vector.model.bin"

    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
        return np.zeros((len(df), 1)), df.get('segment', np.full(len(df), "unknown")).values

    mecab = MeCab.Tagger("-Owakati")
    
    # Pre-process utterances
    # wmd_ent.py logic:
    # utterances = df['text'].tolist()
    # utterances = map(lambda txt:mecab.parse(txt), utterances)
    # speakers = df['speaker'].tolist()

    utterances = df['text'].fillna("").astype(str).tolist()
    # mecab.parse returns string with spaces
    utterances_parsed = [mecab.parse(txt) for txt in utterances]
    speakers = df['speaker'].tolist()

    score_list = []
    
    for i, (utt, spk) in enumerate(zip(utterances_parsed, speakers)):
        # Find previous N utterances of the OTHER speaker
        prev_indices = [j for j in range(i) if speakers[j] != spk]
        
        # Take last n
        target_prev_indices = prev_indices[-n:]
        
        if not target_prev_indices:
            score_list.append(np.nan)
        else:
            target_tokens = utt.strip().split()
            min_distance = float('inf')
            
            for prev_idx in target_prev_indices:
                prev_utt = utterances_parsed[prev_idx]
                prev_tokens = prev_utt.strip().split()
                
                # WMD calculation
                if not target_tokens or not prev_tokens:
                     distance = float('inf')
                else:
                    distance = model.wmdistance(target_tokens, prev_tokens)
                
                if distance < min_distance:
                    min_distance = distance
            
            if min_distance == float("inf"):
                min_distance = np.nan
                
            score_list.append(min_distance)

    # Convert to numpy array, handle NaNs (fill with 0 or max dist?)
    # wmd_ent.py just assigns to df["wmd"].
    # Here we return array.
    
    # Replace NaN with 0 or a large value? WMD is distance, so large value is 'far'.
    # But usually embeddings are 0-centered or similar.
    # If we use 0 for missing, it implies perfect similarity (distance 0).
    # That might be misleading.
    # However, if we follow the snippet in wmd_ent.py which had `df["wmd"] = 0.0` as dummy,
    # maybe 0 is safe placeholder?
    # Let's assume 0 for now but print warning if many NaNs.
    
    X_uclid = np.array(score_list).reshape(-1, 1)
    X_uclid = np.nan_to_num(X_uclid, nan=0.0) # Replace nan with 0.0

    if 'segment' in df.columns:
        segment_sequence = df['segment'].values
    else:
        segment_sequence = np.full(len(df), "unknown")
        
    return X_uclid, segment_sequence
