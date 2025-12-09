import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

def analyze_pearson_corr():
    # Load data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pipeline_result_path = os.path.join(project_root, "result_all_sessions_audio_text.csv")
    questionnaire_path = os.path.join(project_root, "questionaire_score.csv")
    
    if not os.path.exists(pipeline_result_path):
        print(f"Error: Pipeline result not found at {pipeline_result_path}")
        return
    if not os.path.exists(questionnaire_path):
        print(f"Error: Questionnaire score not found at {questionnaire_path}")
        return

    df_res = pd.read_csv(pipeline_result_path)
    df_q = pd.read_csv(questionnaire_path)
    
    # Preprocess session_id to match
    # Ensure session_id columns are strings and padded
    # df_q usually has 'sessionid' as int (e.g. 31714) or str
    if 'sessionid' in df_q.columns:
        df_q['session_id'] = df_q['sessionid'].astype(str).str.zfill(6)
    elif 'session_id' in df_q.columns:
        df_q['session_id'] = df_q['session_id'].astype(str).str.zfill(6)
    else:
        print("Error: No session id column found in questionnaire data.")
        return

    df_res['session_id'] = df_res['session_id'].astype(str).str.zfill(6)
    
    # Merge
    merged_df = pd.merge(df_res, df_q, on='session_id', how='inner')
    
    if merged_df.empty:
        print("No matching sessions found between pipeline results and questionnaire scores.")
        return
        
    print(f"Matched {len(merged_df)} sessions.")
    
    # Identification of feature columns and score columns
    # Score cols: original columns from df_q excluding session identifiers
    q_cols = [c for c in df_q.columns if c not in ['sessionid', 'session_id']]
    # Feature cols: original columns from df_res excluding session identifiers
    res_cols = [c for c in df_res.columns if c != 'session_id']
    
    print(f"Analyzing {len(res_cols)} features against {len(q_cols)} scores.")
    print("Significant Pearson Correlations (p < 0.05):")
    
    count = 0
    for feat in res_cols:
        for score in q_cols:
            # Check for NaN and alignment
            valid_mask = merged_df[feat].notna() & merged_df[score].notna()
            if valid_mask.sum() < 2:
                continue
                
            x = merged_df.loc[valid_mask, feat]
            y = merged_df.loc[valid_mask, score]
            
            # Check if constant
            if x.nunique() <= 1 or y.nunique() <= 1:
                continue

            # Pearson correlation
            try:
                r, p = stats.pearsonr(x, y)
                if p < 0.05:
                    print(f"Pearson: {feat} vs {score} | r={r:.4f}, p={p:.6f}")
                    count += 1
            except Exception as e:
                pass
                
    if count == 0:
        print("No significant correlations found.")

if __name__ == "__main__":
    analyze_pearson_corr()

