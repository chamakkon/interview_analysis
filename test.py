from networks import *
import torch
import pandas as pd
from torch.autograd import Variable
import json
import numpy as np
import itertools





def lp_distance(x1, x2, p=1):
    dist = sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5
    return dist

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


def tned_score(df):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = "../tned/model/triplet_64d_50ep_candor_test_3000.pth"
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
    tned_scores = [0]
    for i in range(len(feature)-1):
        x_data = feature[i]
        y_data = feature[i+1]
        x_data = Variable(torch.from_numpy(x_data)).double().to(device)
        y_data = Variable(torch.from_numpy(y_data)).double().to(device)

        zx = model.get_embedding(x_data)
        zy = model.get_embedding(y_data)

        loss_real = lp_distance(zx, zy).data
        tned_scores.append(float(loss_real))
    df["tned"] = np.nan
    for i in range(len(df)-1):
        if i == 0:
            continue
        if df["speaker"][i] != df["speaker"][i-1]:
            df["tned"][i] = tned_scores[i]
    
    return df