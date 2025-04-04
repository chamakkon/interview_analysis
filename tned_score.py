from test import tned_score
from wmd_ent import uclid
from lexical_ent import lexical_entrainment_score
import pandas as pd

def neural_scores(session):
    df = pd.read_csv(f"{session}_features.csv")
    df = tned_score(df)
    df = uclid(df)
    df.to_csv()
    return df
