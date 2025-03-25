import whisper
import os
import pandas as pd
from pydub import AudioSegment
from pyannote.audio import Pipeline
import torch

# define the diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")
pipeline.to(torch.device("cuda"))
#　define speech recognition model
model = whisper.load_model("large-v3")
audio_samples = os.listdir("interview_audio")
#　make nessesary directories
os.makedirs("transcript", exist_ok=True)
os.makedirs("gained_audio", exist_ok=True)
os.makedirs("segment_audio", exist_ok=True)



for auido in audio_samples:
    segment_list = []
    os.makedirs(f"segment_audio/{audio[:-4]}", exist_ok=True)
    audio_path = f"interview_audio/{auido}"
    diarization = pipeline(audio_path)
    full_audio = AudioSegment.from_file(audio_path, format="wav")
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        #diarization and audio split
        segment_audio = full_audio[turn.start * 1000: turn.end * 1000]
        segment_audio.export(f"segment_audio/{auido[:-4]}/{i}.wav", format="wav")
        #speech recognition
        result = model.transcribe(f"segment_audio/{auido[:-4]}/{i}.wav", language="ja", task="transcribe")
        # add data
        segment_list.append({"start": turn.start, "end": turn.end, "speaker": speaker, "text": result["text"]})
    df = pd.DataFrame(segment_list, columns=["start", "end", "speaker", "text"])
    df.to_csv(f"transcript/{auido[:-4]}.csv", index=False)
        

