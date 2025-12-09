"""
layered_analysis.raw_features.audio_features

音声波形と発話区間情報から、発話単位の音声特徴量行列を抽出するモジュール。

このモジュールで抽出する特徴量とライブラリ構成は、
ローカルの `feat_extract.py` で使用しているものと同一とする。
`feat_extract.py` 内で定義された FEATURE_NAMES と同じ順序・定義で
発話ごとの特徴量を計算する。

出力は次の 2 つとする:
    X_audio: np.ndarray, shape (T, D)
        各行が 1 発話に対応し、各列が FEATURE_NAMES で定義された
        発話単位の音声特徴量（発話区間内で統計量を取った F0, log F0,
        強度、jitter、shimmer、MFCC 系特徴など）に対応する。

    segment_sequence: np.ndarray, shape (T,)
        各発話が属するセグメントラベル（例: "seg1", "seg2", "seg3"）。

ここでの T は「発話数」であり、後段では
    scalerize_seq_data(X_audio[:, j], segment_sequence)
の形で列単位の時系列として扱う。
"""
import sys
import os
import numpy as np
import pandas as pd
import subprocess
import itertools

def extract_audio_features(df, output_dir="tmp/temp_audio_features"):
    """
    Extract audio features using feat_extract.py logic (re-implemented).
    """
    # Configuration path from original file
    # Adjust relative path assuming execution from project root or similar.
    # The original file uses "../../interview_analysis_demo/..."
    # We should try to find it relative to current file location?
    # Or assume the user runs this script from a location where that path is valid.
    # We will use the absolute path relative to this file to be safe if possible, 
    # but the config file is outside this repo structure shown in workspace?
    # Workspace: /mnt/home/shusuke-k/interview/interview_analysis
    # Config:    ../../interview_analysis_demo/...
    # So it's in a sibling directory of the parent of workspace.
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # project_root = .../interview_analysis
    # We want .../interview_analysis_demo/... which is likely at .../interview_analysis/../../interview_analysis_demo
    # Wait, ../../ form project root?
    # The original file is in project_root. So it points to ../../interview_analysis_demo
    # Let's resolve that path.
    
    config_path = os.path.join(project_root, "../../interview_analysis_demo/emobase2010_haoqi_revised.confのコピー")
    if not os.path.exists(config_path):
        # Fallback to the exact string in original file if resolution fails
        config_path = "../../interview_analysis_demo/emobase2010_haoqi_revised.confのコピー"

    os.makedirs(output_dir, exist_ok=True)
    
    # Feature extraction loop
    # feat_extract.py iterates df and calls SMILExtract
    
    for i, row in df.iterrows():
        audio_path = row.get('audio', '')
        if not audio_path or not os.path.exists(audio_path):
            continue
            
        out_csv_path = os.path.join(output_dir, f"{i}.csv")
        # Run SMILExtract
        # Assuming SMILExtract is in PATH
        cmd_feat = f'SMILExtract -nologfile -C "{config_path}" -I "{audio_path}" -O "{out_csv_path}"'
        subprocess.call(cmd_feat, shell=True)
        
    # Calculate statistics
    # Logic copied from feat_extract.py
    
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
    
    # Columns for the result
    columns = list(itertools.chain.from_iterable([[name+"_mean", name+"_median", name+"_std", name+"_percentile_1", name+"_percntile_99", name+"_range"] for name in feature_names]))
    
    X_list = []
    
    for i, row in df.iterrows():
        csv_path = os.path.join(output_dir, f"{i}.csv")
        if not os.path.exists(csv_path):
            # Fill zeros
            X_list.append(np.zeros(len(columns)))
            continue
            
        try:
            # Try comma first (default for pd.read_csv), then semicolon
            extracted = pd.read_csv(csv_path)
            if extracted.shape[1] < 2: # Heuristic to check if parsing failed
                 extracted = pd.read_csv(csv_path, sep=';')

            # DEBUG: Print first file info
            if i == 0:
                print(f"DEBUG: Processing {csv_path}")
                print(f"DEBUG: Columns found: {extracted.columns.tolist()[:5]} ...")
                print(f"DEBUG: First feature '{feature_names[0]}' in columns: {feature_names[0] in extracted.columns}")
                if feature_names[0] in extracted.columns:
                    print(f"DEBUG: First feature mean: {extracted[feature_names[0]].mean()}")

            functional_features = []
            for feature_name in feature_names:
                if feature_name not in extracted.columns:
                    # Should not happen if config is correct
                    functional_features.extend([0]*6)
                    continue
                    
                feature = extracted[feature_name]
                
                # Logic from calculate_func
                mean = feature.mean()
                median = feature.median()
                std = feature.std()
                percentile_1 = feature.quantile(0.01)
                percentile_99 = feature.quantile(0.99)
                range_value = percentile_99 - percentile_1
                
                functional_features.extend([mean, median, std, percentile_1, percentile_99, range_value])
            
            X_list.append(functional_features)
            
        except Exception as e:
            print(f"Error reading features for {i}: {e}")
            X_list.append(np.zeros(len(columns)))

    X_audio = np.array(X_list)
    
    if 'segment' in df.columns:
        segment_sequence = df['segment'].values
    else:
        segment_sequence = np.full(len(df), "unknown")
        
    return X_audio, segment_sequence
