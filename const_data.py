import pandas as pd
from split_audio import split_audio
from feat_extarct import ipu_features

def construct(session):
    audio_file = f"interview_audio/{session}.wav"
    transcript_file = f"transcript/csv/{session}.csv"
    df = split_audio(audio_file, transcript_file, session)
    result_df = ipu_features(df, f"features/{session}")
    df = pd.merge(df, result_df, on="audio")
    df.to_csv(f"{session}_features.csv", index=False)


