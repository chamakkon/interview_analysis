# セッション031714だけに対して、音声・言語系のみの分析を行うパイプライン。
import sys
import os
import pandas as pd
import numpy as np

# パス設定: layered_analysis パッケージをインポートできるようにする
# このファイルは layered_analysis/test_pipeline.py なので、
# layered_analysis ディレクトリ自体をパッケージとして扱うため、
# その親ディレクトリ (interview_analysis) を sys.path に追加するのが適切ですが、
# 内部モジュール間の import (from raw_features ...) を機能させるため、
# このファイルのあるディレクトリ (layered_analysis) を基点にします。
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 各モジュールのインポート
try:
    from raw_features.audio_features import extract_audio_features
    from raw_features.audio_ent_features import extract_audio_ent_features
    from raw_features.word_count_features import extract_word_count_features
    from embedding_vector.hubert import extract_hubert_features
    from embedding_vector.tned_vector import extract_tned_vector
    from embedding_vector.uclid_vector import extract_uclid_vector
    from interpretable.emotion2vector import extract_emotion2vec_features
    from interpretable.roberta_emo import extract_text_emo_features

    from func.scalerize_seq_data import scalerize_seq_data
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running this script from the project root or with appropriate PYTHONPATH.")
    # Fallback for running directly inside layered_analysis folder
    try:
        from .raw_features.audio_features import extract_audio_features
    except:
        pass


def run_test_pipeline(session_id="031714"):
    print(f"Running pipeline for session: {session_id}")
    
    # 1. データ読み込み
    # プロジェクトルートの audio_feature フォルダにあると仮定
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, f"audio_feature/{session_id}_features.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        # Try finding it in current dir relative path just in case
        data_path = f"audio_feature/{session_id}_features.csv"
        if not os.path.exists(data_path):
             return

    df = pd.read_csv(data_path)
    print(f"Loaded data: {len(df)} turns.")

    # 2. 特徴量抽出関数リスト
    # (関数, 名前) のタプル
    # 音声・言語系のみ
    extractors = [
        (extract_audio_features, "audio_feat"),
        (extract_audio_ent_features, "audio_ent"),
        (extract_word_count_features, "word_count"),
        (extract_hubert_features, "hubert"),
        (extract_tned_vector, "tned"),
        (extract_uclid_vector, "uclid"),
        (extract_emotion2vec_features, "emo2vec"),
        (extract_text_emo_features, "text_emo"),
    ]

    all_scalar_features = {}

    for extract_func, name in extractors:
        print(f"Extracting {name}...")
        try:
            # 特徴量抽出: (T, D), (T,)
            # 引数に output_dir が必要なものはとりあえずデフォルトか一時ディレクトリを指定
            if name == "audio_feat":
                 X, segments = extract_func(df, output_dir=f"temp_{session_id}_{name}")
            else:
                 X, segments = extract_func(df)
            
            print(f"  Shape: {X.shape}")
            
            # 3. スカラー化 (列ごとに適用)
            # 次元 D ごとに 4つのスカラー値を計算
            dim = X.shape[1]
            for d in range(dim):
                seq_data = X[:, d]
                
                # スカラー化
                mean_val, trend, rmssd, seg_diff = scalerize_seq_data(seq_data, segments)
                
                # 結果を保存 (特徴量名_次元_統計量)
                # 次元名が特定できない場合はインデックスを使用
                prefix = f"{name}_dim{d}"
                all_scalar_features[f"{prefix}_mean"] = mean_val
                all_scalar_features[f"{prefix}_trend"] = trend
                all_scalar_features[f"{prefix}_rmssd"] = rmssd
                all_scalar_features[f"{prefix}_seg3-seg1"] = seg_diff
                
        except Exception as e:
            print(f"  Failed to extract {name}: {e}")
            #import traceback
            #traceback.print_exc()

    # 4. 結果出力
    if not all_scalar_features:
        print("No features extracted.")
        return

    result_df = pd.DataFrame([all_scalar_features])
    result_df.insert(0, "session_id", session_id)
    
    output_csv = f"result_{session_id}_audio_text.csv"
    result_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    
    # 簡易表示 (最初の数カラム)
    print(result_df.iloc[:, :10])

if __name__ == "__main__":
    run_test_pipeline()
