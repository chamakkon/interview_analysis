import whisper
import os
import pandas as pd
from pydub import AudioSegment

model = whisper.load_model("openai/whisper-large-v3")
audio_samples = os.listdir("interview_audio")
os.makedirs("transcript", exist_ok=True)
os.makedirs("gained_audio", exist_ok=True)
for auido in audio_samples:
    segment_list = []
    audio_path = f"interview_audio/{auido}"
    audio_segment = AudioSegment.from_file(audio_path, format="wav")
    gained = audio_segment * 2
    gained_path = f"gained_audio/{auido}"
    gained.export(gained_path, format="wav")
    result = model.transcribe(gained_path, verbose=True, language="ja", task="transcribe")
    for segment in result["segments"]:
        segment_list.append([segment["start"], segment["end"], segment["text"]])
    df = pd.DataFrame(segment_list, columns=["start", "end", "text"])
    df.to_csv(f"transcript/{auido[:-4]}.csv", index=False)

