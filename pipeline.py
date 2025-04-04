
import pandas as pd
from tned_score import neural_scores
from lexical_ent import lexical_entrainment_score
from linear_ent import linear_ent_score
from const_data import construct
import os
import json
print("Start")
sessions = os.listdir("transcript/csv")
for session_csv in sessions:
    session = session_csv[:-4]
    segment_list = ["seg1", "seg2", "seg3", "seg4", "seg5"]
    collection_part = ["seg1", "seg2", "seg3"]
    scores = {"session":session}
    construct(session)
    df = neural_scores(session)
    print(f"========== calculating whole {session} session ==============")
    scores["tned"] = df.dropna(subset=["tned"])["tned"].mean()
    scores["tned_subject"] = df[df["speaker"]=="subject"].dropna(subset=["tned"])["tned"].mean()
    scores["wmd"] = df.dropna(subset=["wmd"])["wmd"].mean()
    scores["wmd_subject"] = df[df["speaker"]=="subject"].dropna(subset=["wmd"])["wmd"].mean()
    scores["lexical"] = lexical_entrainment_score(df)
    scores["linear"] = linear_ent_score(df)
    print(f"========== calculating {sessioin} data collection part ==============")
    scores["tned_collection"] = df[df["segment"].isin(collection_part)].dropna(subset=["tned"])["tned"].mean()
    scores["wmd_collection"] = df[df["segment"].isin(collection_part)].dropna(subset=["wmd"])["wmd"].mean()
    scores["tned_collection_subject"] = df[(df["segment"].isin(collection_part)) & (df["speaker"]=="subject")].dropna(subset=["tned"])["tned"].mean()
    scores["wmd_collection_subject"] = df[(df["segment"].isin(collection_part)) & (df["speaker"]=="subject")].dropna(subset=["wmd"])["wmd"].mean()
    scores["lexical_collection"] = lexical_entrainment_score(df[df["segment"].isin(collection_part)])
    scores["linear_collection"] = linear_ent_score(df[df["segment"].isin(collection_part)])
    for segment in segment_list:
        print(f"========== calculating {session} {segment}==============")
        sec_scores ={}
        sec_scores[f"tned_{segment}"] = df[df["segment"]==segment].dropna(subset=["tned"])["tned"].mean()
        sec_scores[f"wmd_{segment}"] = df[df["segment"]==segment].dropna(subset=["wmd"])["wmd"].mean()
        sec_scores[f"tned_subject_{segment}"] = df[(df["segment"]==segment) & (df["speaker"]=="subject")].dropna(subset=["tned"])["tned"].mean()
        sec_scores[f"wnd_subject_{segment}"] = df[(df["segment"]==segment) & (df["speaker"]=="subject")].dropna(subset=["wmd"])["wmd"].mean()
        sec_scores[f"linear_{segment}"] = linear_ent_score(df, segment=segment)
        sec_scores[f"lexical_{segment}"] = lexical_entrainment_score(df, segment=segment)
        scores[segment]= sec_scores
    scores["tned_3-1"] = scores["sec1"]["tned_sec1"]- scores["sec3"]["tned_sec3"]
    scores["wmd_3-1"] = scores["sec1"]["wmd_sec1"]-scores["sec3"]["tned_sec3"]
    open(f"{session}_result.json", "w").write(json.dumps(scores, indent=4))

