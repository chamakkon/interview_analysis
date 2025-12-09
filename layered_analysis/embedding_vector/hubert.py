"""
layered_analysis.embedding_vector.hubert

HuBERT 系自己教師あり音声モデルを用いて、発話単位の音声埋め込み行列を抽出するモジュール。

使用モデルの例:
    - "facebook/hubert-large-ls960-ft"

各発話区間について HuBERT のフレーム埋め込みを取得し、
発話内フレーム方向に平均 pooling することで、発話単位のベクトルを得る。

出力は次の 2 つとする:
    X_hubert: np.ndarray, shape (T, D_model)
        各行が 1 発話に対応し、各列が選択した層の隠れ状態次元
        （例: 1024 次元）に対応する。

    segment_sequence: np.ndarray, shape (T,)
        各発話が属するセグメントラベル（"seg1", "seg2", "seg3"）。

ここでの T は「発話数」であり、列単位に取り出すことで
発話時系列として `scalerize_seq_data` に渡せる。
"""
import numpy as np
import pandas as pd
import torch
import librosa
import os

try:
    from transformers import Wav2Vec2Processor, HubertModel
except ImportError:
    Wav2Vec2Processor = None
    HubertModel = None

def extract_hubert_features(df, model_name="facebook/hubert-large-ls960-ft"):
    """
    Extract HuBERT embeddings.
    
    Args:
        df (pd.DataFrame): Dataframe with 'audio' column (path to wav).
        model_name (str): HuggingFace model name.
        
    Returns:
        X_hubert (np.ndarray): (T, D)
        segment_sequence (np.ndarray): (T,)
    """
    if HubertModel is None:
        print("Warning: transformers not installed. returning zeros.")
        return np.zeros((len(df), 1024)), df.get('segment', np.full(len(df), "unknown")).values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = HubertModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading HuBERT model: {e}")
        return np.zeros((len(df), 1024)), df.get('segment', np.full(len(df), "unknown")).values
        
    embeddings = []
    
    for _, row in df.iterrows():
        audio_path = row.get('audio', '')
        if not os.path.exists(audio_path):
            # Try relative path? or skip
            embeddings.append(np.zeros(model.config.hidden_size))
            continue
            
        # Load audio
        # HuBERT expects 16kHz
        try:
            speech, sr = librosa.load(audio_path, sr=16000)
            input_values = processor(speech, return_tensors="pt", sampling_rate=sr).input_values.to(device)
            
            with torch.no_grad():
                outputs = model(input_values)
            
            # outputs.last_hidden_state: (batch, seq_len, hidden_size)
            hidden_states = outputs.last_hidden_state
            
            # Mean pooling over time
            pooled = torch.mean(hidden_states, dim=1).cpu().numpy().flatten()
            embeddings.append(pooled)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            embeddings.append(np.zeros(model.config.hidden_size))
            
    X_hubert = np.array(embeddings)
    
    if 'segment' in df.columns:
        segment_sequence = df['segment'].values
    else:
        segment_sequence = np.full(len(df), "unknown")
        
    return X_hubert, segment_sequence
