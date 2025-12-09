"""
layered_analysis.embedding_vector.tned_vector

TNED (Temporal Neural Entrainment Distance) 手法を用いて、
発話単位のエントレインメント埋め込み行列を生成するモジュール。

埋め込みの定義はローカルの `tned_local_diff.py` と同一とする。
対話から発話単位でエントレインメント埋め込みを算出し、
1 発話あたり 1 ベクトルとして行列を構成する。

出力は次の 2 つとする:
    X_tned: np.ndarray, shape (T, D_tned)
        各行が 1 発話に対応し、各列が tned_local_diff.py で定義された
        埋め込み次元に対応する。

    segment_sequence: np.ndarray, shape (T,)
        各発話が属するセグメントラベル（"seg1", "seg2", "seg3"）。

ここでの T は「発話数」であり、列単位の発話時系列として
`scalerize_seq_data` に渡す。
"""
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools

# --- Copy of EmbeddingNet from networks.py ---
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.fc = nn.Sequential(nn.Linear(228, 256),
                                nn.ReLU(),
                                nn.Linear(256, 128),
                                nn.ReLU(),
                                nn.Linear(128, 30)
                                )

    def forward(self, x):
        output = self.fc(x.float())
        return output

    def get_embedding(self, x):
        return self.forward(x)

# --- tned_feature helper (from test.py logic) ---
def get_tned_features_from_df(df):
    nested_feature_names = ["F0final_sma",
                         "pcm_loudness_sma",
                         "F0final_sma_de",
                         "pcm_loudness_sma_de",
                         [f"pcm_fftMag_mfcc_sma[{i}]" for i in range(15)],
                         [f"lspFreq_sma[{i}]" for i in range(8)],
                         [f"logMelFreqBand_sma[{i}]" for i in range(8)],
                         "jitterLocal_sma",
                         "jitterDDP_sma",
                         "shimmerLocal_sma",
                         ]
    feature_names = list(itertools.chain.from_iterable(elem if isinstance(elem, list) else [elem] for elem in nested_feature_names))
    columns = list(itertools.chain.from_iterable([[name+"_mean", name+"_median", name+"_std", name+"_percentile_1", name+"_percntile_99", name+"_range"] for name in feature_names]))
    
    # Check if columns exist
    missing = [c for c in columns if c not in df.columns]
    if missing:
        # Fill missing with 0
        for c in missing:
            df[c] = 0.0
            
    features = df[columns].to_numpy()
    return features


def extract_tned_vector(df):
    """
    Extract TNED embeddings (and their diffs).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Model path resolution
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "../tned/model/triplet_64d_50ep_candor_test_3000.pth")
    # Also check if it's inside the repo structure
    if not os.path.exists(model_path):
        # Fallback to a hardcoded assumption or relative path from previous context
        model_path = "../../tned/model/triplet_64d_50ep_candor_test_3000.pth"

    embedding_net = EmbeddingNet()
    model = embedding_net
    
    try:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("embedding_net.", "")
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
        else:
            print(f"Warning: TNED model not found at {model_path}. Using random weights.")
    except Exception as e:
        print(f"Error loading TNED model: {e}. Using random weights.")

    model.to(device)
    model.eval()

    feature = get_tned_features_from_df(df)
    
    embeddings = []
    
    for i in range(len(feature)):
        x_data = feature[i]
        x_data = Variable(torch.from_numpy(x_data)).double().to(device)
        
        if len(x_data.shape) == 1:
            x_data = x_data.unsqueeze(0)
            
        # Ensure float type for model (model expects float, x_data might be double)
        x_data = x_data.float()
        
        with torch.no_grad():
            z = model.get_embedding(x_data)
        embeddings.append(z.detach().cpu().numpy().flatten())
        
    embeddings = np.array(embeddings)
    
    # Calculate Diff (TNED feature)
    X_tned_list = []
    for i in range(len(embeddings)):
        if i == 0:
            X_tned_list.append(np.zeros_like(embeddings[i]))
        else:
            diff = np.abs(embeddings[i] - embeddings[i-1])
            X_tned_list.append(diff)
            
    X_tned = np.array(X_tned_list)

    if 'segment' in df.columns:
        segment_sequence = df['segment'].values
    else:
        segment_sequence = np.full(len(df), "unknown")

    return X_tned, segment_sequence
