import pandas as pd
import MeCab
from collections import Counter, defaultdict
import numpy as np


def get_vocab(df, n=10, segment=None):
    transcript = df
    mecab = MeCab.Tagger("-Owakati")
    vocabs = []
    if segment:
        transcript = transcript[transcript["segment"]==segment]
    utts = transcript["text"]
    utts = utts.map(lambda x:mecab.parse(x))
    for utt in utts.tolist():
        vocabs += utt.strip().split(" ")
    return Counter(vocabs).most_common(n), transcript

def count_keyword(vocab, transcript):
    mecab = MeCab.Tagger("-Owakati")
    speakers = list(set(transcript["speaker"].tolist()))
    counter_a = 0
    counter_b = 0
    result = {}
    for keyword in vocab:
        for i, row in transcript.iterrows():
            utt = mecab.parse(row["text"]).split(" ")
            word_count = utt.count(keyword[0])
            if word_count == 0:
                continue
            elif row["speaker"] == speakers[0]:
                counter_a += word_count
            else:
                counter_b += word_count
        result[keyword] = {speakers[0]:counter_a, speakers[1]:counter_b}
    return result, speakers

def get_ent(word_use, speakers):
    ent_scores = []
    total = 0
    for ratio in word_use.values():
        ent = -np.abs(ratio[speakers[0]]-ratio[speakers[1]])
        total += ratio[speakers[0]]
        total += ratio[speakers[1]]
        ent_scores.append(ent)
    lexical_ent = sum(ent_scores)/total
    return lexical_ent

def lexical_entrainment_score(transcript_df, segment=None):
    return 0.0
    vocab, transcript = get_vocab(transcript_df, segment=segment)
    result, speakers = count_keyword(vocab, transcript)
    return get_ent(result, speakers)
