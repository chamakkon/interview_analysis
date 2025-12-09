import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from networks import *
import os
def tned_feature(df):
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

    features = df[columns].to_numpy()
    print(features)
    return features

def local_tned_diff(df):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = "../../tned/model/triplet_64d_50ep_candor_test_3000.pth"
    embedding_net = EmbeddingNet()

    model = embedding_net
    state_dict = torch.load(model_name, map_location=device)

    # Strip the "embedding_net." prefix from keys in state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("embedding_net.", "")
        new_state_dict[new_key] = v

    # Load the updated state dict into the model
    model = embedding_net  # Assuming embedding_net is your model's architecture
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    feature = tned_feature(df)
    # 各時刻間の埋め込みベクトル差（ベクトルそのもの）を保存
    local_diff_scores = []
    for i in range(len(feature) - 1):
        if df["speaker"][i] == df["speaker"][i+1]:
            continue
        elif df["speaker"][i] == "subject":
            continue
        elif df["segment"][i] == "seg4" or df["segment"][i] == "seg5":
            continue
        x_data = feature[i]
        y_data = feature[i + 1]
        x_data = Variable(torch.from_numpy(x_data)).double().to(device)
        y_data = Variable(torch.from_numpy(y_data)).double().to(device)
        zx = model.get_embedding(x_data)
        zy = model.get_embedding(y_data)
        # 差分ベクトルを要素ごとの絶対値にする
        local_diff = torch.abs(zx - zy)
        local_diff_scores.append(local_diff.detach().cpu().numpy().flatten())
    # shape: (T-1, D) の 2次元配列にして返す
    print(local_diff_scores)
    return np.stack(local_diff_scores, axis=0)

if __name__ == "__main__":
    # 特徴量を読み込み
    session = "031910"
    df = pd.read_csv(f"audio_feature/{session}_features.csv")

    # 局所的な差分（ベクトル）の時系列を計算: shape (T-1, D)
    local_diff_vectors = local_tned_diff(df)  # numpy.ndarray

    # 各次元ごとの推移をプロット
    num_steps, dim = local_diff_vectors.shape
    
    os.makedirs("local_diff_fig", exist_ok=True)
    os.makedirs(f"local_diff_fig/{session}", exist_ok=True)
    for d in range(dim):
        plt.figure(figsize=(12, 6))
        plt.plot(range(num_steps), local_diff_vectors[:, d], alpha=0.6)
        plt.xlabel("Step (frame index)")
        plt.ylabel("Local diff (per-dimension value)")
        plt.title("Per-dimension changes of local_diff vectors")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"local_diff_fig/{session}/{d}.png")





