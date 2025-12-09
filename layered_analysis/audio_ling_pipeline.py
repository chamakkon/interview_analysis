import sys
import os
import pandas as pd
import numpy as np
import glob
import re

# Path setting: allows importing layered_analysis package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
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
except ImportError:
    # Fallback if running from root without package structure
    try:
        from .raw_features.audio_features import extract_audio_features
        from .raw_features.audio_ent_features import extract_audio_ent_features
        from .raw_features.word_count_features import extract_word_count_features
        from .embedding_vector.hubert import extract_hubert_features
        from .embedding_vector.tned_vector import extract_tned_vector
        from .embedding_vector.uclid_vector import extract_uclid_vector
        from .interpretable.emotion2vector import extract_emotion2vec_features
        from .interpretable.roberta_emo import extract_text_emo_features
        from .func.scalerize_seq_data import scalerize_seq_data
    except ImportError as e:
        print(f"Import Error: {e}")
        sys.exit(1)

def run_all_sessions_pipeline():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audio_feature_dir = os.path.join(project_root, "audio_feature")
    
    if not os.path.exists(audio_feature_dir):
        print(f"Error: Directory not found: {audio_feature_dir}")
        return

    # Get all session files
    session_files = glob.glob(os.path.join(audio_feature_dir, "*_features.csv"))
    session_ids = []
    for f in session_files:
        basename = os.path.basename(f)
        match = re.match(r"(\d+)_features\.csv", basename)
        if match:
            session_ids.append(match.group(1))
    
    session_ids.sort()
    print(f"Found {len(session_ids)} sessions: {session_ids}")
    
    all_sessions_data = []

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

    for i, session_id in enumerate(session_ids):
        print(f"\n[{i+1}/{len(session_ids)}] Processing session: {session_id}")
        data_path = os.path.join(audio_feature_dir, f"{session_id}_features.csv")
        
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            print(f"  Error loading data for {session_id}: {e}")
            continue

        session_results = {"session_id": session_id}
        
        for extract_func, name in extractors:
            print(f"  Extracting {name}...")
            try:
                # Prepare args
                kwargs = {}
                if name == "audio_feat":
                    # Use a session-specific temp dir
                    # Using tmp/ inside project root
                    kwargs["output_dir"] = os.path.join(project_root, "tmp", f"temp_{session_id}_{name}")
                
                # Extract
                X, segments = extract_func(df, **kwargs)
                
                # Check for empty result
                if X is None or len(X) == 0:
                     print(f"    Warning: Empty features for {name}")
                     continue

                # Scalerize
                dim = X.shape[1]
                for d in range(dim):
                    seq_data = X[:, d]
                    mean_val, trend, rmssd, seg_diff = scalerize_seq_data(seq_data, segments)
                    
                    prefix = f"{name}_dim{d}"
                    session_results[f"{prefix}_mean"] = mean_val
                    session_results[f"{prefix}_trend"] = trend
                    session_results[f"{prefix}_rmssd"] = rmssd
                    session_results[f"{prefix}_seg3-seg1"] = seg_diff
                    
            except Exception as e:
                print(f"  Failed to extract {name} for {session_id}: {e}")
                import traceback
                traceback.print_exc()
        
        all_sessions_data.append(session_results)

    if not all_sessions_data:
        print("No data collected.")
        return

    # Create final DataFrame
    final_df = pd.DataFrame(all_sessions_data)
    
    # Reorder columns to have session_id first
    cols = ["session_id"] + [c for c in final_df.columns if c != "session_id"]
    final_df = final_df[cols]
    
    output_csv = os.path.join(project_root, "result_all_sessions_audio_text.csv")
    final_df.to_csv(output_csv, index=False)
    print(f"\nAll sessions processed. Results saved to {output_csv}")

if __name__ == "__main__":
    run_all_sessions_pipeline()

