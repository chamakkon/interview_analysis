import pandas as pd
from pydub import AudioSegment
import os

def split_audio(audio_file, transcript_file, session):

    audio = AudioSegment.from_file(audio_file, format="wav")
    transcript_df = pd.read_csv(transcript_file)
    print(len(transcript_df))
    current_utt = {}
    audio_id = 0
    utt_dict = []

    for i, row in transcript_df.iterrows():
        if i == 0:
            current_utt = {"audio":str(audio_id).zfill(4), "text":row["text"], "start":row["start"], "end":row["end"], "speaker":row["speaker"], "segment":row["segment"]}
            continue
        if row["speaker"] != current_utt["speaker"]:
            utt_dict.append(current_utt)
            audio_id += 1
            current_utt = {"audio":str(audio_id).zfill(4), "text":row["text"], "start":row["start"], "end":row["end"], "speaker":row["speaker"], "segment":row["segment"]}
        elif row["start"] > current_utt["end"]+0.5:
            utt_dict.append(current_utt)
            audio_id += 1
            current_utt = {"audio":str(audio_id).zfill(4), "text":row["text"], "start":row["start"], "end":row["end"], "speaker":row["speaker"], "segment":row["segment"]}
        elif row["segment"] != current_utt["segment"]:
            utt_dict.append(current_utt)
            audio_id += 1
            current_utt = {"audio":str(audio_id).zfill(4), "text":row["text"], "start":row["start"], "end":row["end"], "speaker":row["speaker"], "segment":row["segment"]}
        else:
            current_utt["end"] = row["end"]
            current_utt["text"] += row["text"]
    utt_dict.append(current_utt)
    df = pd.DataFrame(utt_dict)
    os.makedirs(f"features/{session}", exist_ok=True)

    for i, row in df.iterrows():
        segment = audio[int(row["start"]*1000):int(row["end"]*1000)]
        segment.export(f"features/{session}/{row['audio']}.wav", format="wav")
        df["audio"][i] = f"features/{session}/{row['audio']}.wav"
    os.makedirs("ipus", exist_ok=True)
    df.to_csv(f"ipus/{session}_ipu.csv")
    return df
        
