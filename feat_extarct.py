import pandas as pd
import json
from pydub import AudioSegment
from typing import List
import os
import subprocess
import itertools

def ipu_features(utt_df, output_path):
    CONFIG_openSMILE = "../../interview_analysis_demo/emobase2010_haoqi_revised.confのコピー"
    #df = pd.read_csv(csv_path)
    df = utt_df
    os.makedirs(output_path, exist_ok=True)
    result_df = pd.DataFrame()
    for i, row in df.iterrows():
        cmd_feat = 'SMILExtract -nologfile -C %s -I %s -O %s' %(CONFIG_openSMILE, row['audio'], f"{output_path}/{i}.csv")
        subprocess.call(cmd_feat, shell  = True)
    means, stds = get_session_level_norm(df["audio"].tolist(), output_path)
    for i, path in enumerate(df["audio"].tolist()):
        extracted = pd.read_csv(f"{output_path}/{i}.csv")
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
        feature_values = itertools.chain.from_iterable(calculate_func([extracted[feature] for feature in feature_names], means, stds) )
        columns = itertools.chain.from_iterable([[name+"_mean", name+"_median", name+"_std", name+"_percentile_1", name+"_percntile_99", name+"_range"] for name in feature_names])
        new_df = pd.DataFrame(data=[feature_values], columns=columns)
        new_df["audio"] = path
        result_df = pd.concat([result_df, new_df], ignore_index=True)
    return result_df


def calculate_func(features, means, stds):
    if type(features) != list:
        features = [features]
    functional_features = []
    for i, feature in enumerate(features):
        if i <2:
            feature/means[i]
        else:
            (feature-mean)/stds[i]
        mean = feature.mean()
        median = feature.median()
        std = feature.std()
        percentile_1 = feature.quantile(0.01)
        percentile_99 = feature.quantile(0.99)
        range_value = percentile_99 - percentile_1
        functional_features.append([mean, median, std, percentile_1, percentile_99, range_value])
    return functional_features

def get_session_level_norm(paths, output_path):
    num_data = len(paths)
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
    df = pd.DataFrame()
    for i in range(num_data):
        f = pd.read_csv(os.path.join(*[output_path, f"{i}.csv"]))
        new_df = f[feature_names]
        df = pd.concat([df, new_df])
    means = [df[feature_name].mean() for feature_name in feature_names]
    stds = [df[feature_name].std() for feature_name in feature_names]
    return means, stds
    
def add_speaker_info(df, data_path,i):
    f = open(data_path, "r")
    data_info = json.load(f)
    speaker = data_info[str(i)]["spks"]
    print(speaker)
    df["spks"] = speaker
    return df


def extract(data_path, csv_path, output_path, i=None):
    df = ipu_features(csv_path, output_path)
    print(df.head())
    #df = add_speaker_info(df, data_path,i)
    os.makedirs("features", exist_ok=True)
    if i:
        path = f"features/extracted_{i}.csv"
        df.to_csv(path)
        return path
    path = "features/extracted.csv"
    df.to_csv(path)
    return path