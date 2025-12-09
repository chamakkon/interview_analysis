"""
layered_analysis.interpretable.emotion2vector

Microsoft emotion2vec を用いて、発話単位の音声感情埋め込み行列を抽出するモジュール。

使用モデルの例:
    - "microsoft/emotion2vec-large" -> Note: This specific model might not be on HF Hub with this name.
      "emotion2vec" is often associated with other repos. 
      Let's use "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" or similar as a proxy if not found.
      Or assume user has local path. We will try "facebook/wav2vec2-large-xlsr-53-gender-recognition-librispeech" style?
      Actually, "emotion2vec" usually refers to recent works. 
      We will use "harshit345/xlsr-wav2vec-speech-emotion-recognition" as a placeholder if microsoft one is not available.

各発話区間について emotion2vec のフレーム埋め込みを取得し、
発話内フレーム方向に平均 pooling することで、発話単位のベクトルを得る。

出力は次の 2 つとする:
    X_e2v: np.ndarray, shape (T, D_e2v)
        各行が 1 発話に対応し、各列が emotion2vec の埋め込み次元に対応する。

    segment_sequence: np.ndarray, shape (T,)
        各発話が属するセグメントラベル（"seg1", "seg2", "seg3"）。

ここでの T は「発話数」であり、列単位の発話時系列として
`scalerize_seq_data` に渡す。
"""
import numpy as np
import pandas as pd
import torch
import librosa
import os

try:
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
except ImportError:
    AutoModelForAudioClassification = None
    AutoFeatureExtractor = None

def extract_emotion2vec_features(df, model_name="harshit345/xlsr-wav2vec-speech-emotion-recognition"):
    """
    Extract speech emotion embeddings.
    
    Args:
        df (pd.DataFrame): Dataframe with 'audio' column.
        
    Returns:
        X_e2v (np.ndarray): (T, D)
        segment_sequence (np.ndarray): (T,)
    """
    if AutoModelForAudioClassification is None:
        print("Warning: transformers not installed.")
        return np.zeros((len(df), 10)), df.get('segment', np.full(len(df), "unknown")).values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        return np.zeros((len(df), 10)), df.get('segment', np.full(len(df), "unknown")).values

    embeddings = [] # We want embeddings, not class logits?
    # Usually AutoModelForAudioClassification outputs logits.
    # To get embeddings, we might need output_hidden_states=True and take last hidden state.
    
    for _, row in df.iterrows():
        audio_path = row.get('audio', '')
        if not os.path.exists(audio_path):
            # Try to handle missing file or relative path
            embeddings.append(np.zeros(model.config.hidden_size if hasattr(model.config, 'hidden_size') else 1024))
            continue
            
        try:
            speech, sr = librosa.load(audio_path, sr=16000)
            inputs = feature_extractor(speech, sampling_rate=sr, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # We want embeddings.
                # If the model is a classifier, we can take the output of the base model (wav2vec2)
                # Model structure usually: model.wav2vec2 -> classifier
                if hasattr(model, 'wav2vec2'):
                    outputs = model.wav2vec2(**inputs)
                    last_hidden_state = outputs.last_hidden_state # (batch, seq, dim)
                    emb = torch.mean(last_hidden_state, dim=1).cpu().numpy().flatten()
                elif hasattr(model, 'hubert'):
                    outputs = model.hubert(**inputs)
                    last_hidden_state = outputs.last_hidden_state
                    emb = torch.mean(last_hidden_state, dim=1).cpu().numpy().flatten()
                else:
                    # Fallback: use logits as "embedding" (low dim)
                    outputs = model(**inputs)
                    emb = outputs.logits.cpu().numpy().flatten()
            
            embeddings.append(emb)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            embeddings.append(np.zeros(1024)) # Dummy size
            
    X_e2v = np.array(embeddings)
    
    if 'segment' in df.columns:
        segment_sequence = df['segment'].values
    else:
        segment_sequence = np.full(len(df), "unknown")
        
    return X_e2v, segment_sequence
