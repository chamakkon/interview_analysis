import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

def analyze_linear_reg():
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
    
    # Preprocess session_id
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
        print("No matching sessions found.")
        return
        
    print(f"Matched {len(merged_df)} sessions.")
    
    q_cols = [c for c in df_q.columns if c not in ['sessionid', 'session_id']]
    res_cols = [c for c in df_res.columns if c != 'session_id']
    
    print(f"Analyzing {len(res_cols)} features against {len(q_cols)} scores.")
    print("Significant Linear Regressions (slope p < 0.05):")
    
    count = 0
    for feat in res_cols:
        for score in q_cols:
            valid_mask = merged_df[feat].notna() & merged_df[score].notna()
            if valid_mask.sum() < 2:
                continue
                
            x = merged_df.loc[valid_mask, feat]
            y = merged_df.loc[valid_mask, score]
            
            if x.nunique() <= 1 or y.nunique() <= 1:
                continue

            try:
                # Linregress: y = slope * x + intercept
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                if p_value < 0.05:
                    print(f"LinearReg: {feat} -> {score} | slope={slope:.4f}, p={p_value:.6f}, R^2={r_value**2:.4f}")
                    count += 1
            except Exception:
                pass

    if count == 0:
        print("No significant regressions found.")

if __name__ == "__main__":
    analyze_linear_reg()

