from entrainment_metrics import InterPausalUnit
from entrainment_metrics.continuous import TimeSeries, calculate_metric
import pandas as pd
import numpy as np
import itertools

def linear_ent_score(df, section=False):
    if section:
        df = df[df["section"]==section]
    df = df.dropna(how="any")
    indy_ipu = []
    subject_ipu = []
    scores = {}

    metrics = ["synchrony", "convergence", "proximity"]
    nested_feature_names = ["F0final_sma",
                            "pcm_loudness_sma",
                            "F0final_sma_de",
                            "pcm_loudness_sma_de",
                            [f"pcm_fftMag_mfcc_sma[{i}]" for i in range(15)],
                            [f"lspFreq_sma[{i}]" for i in range(8)],
                            [f"logMelFreqBand_sma[{i}]" for i in range(8)],
                            "jitterLocal_sma",
                            "jitterDDP_sma",
                            "shimmerLocal_sma",
                            ]
    feature_names = list(itertools.chain.from_iterable(elem if isinstance(elem, list) else [elem] for elem in nested_feature_names))
    columns = list(itertools.chain.from_iterable([[name+"_mean", name+"_median", name+"_std", name+"_percentile_1", name+"_percntile_99", name+"_range"] for name in feature_names]))
    for i, row in df.iterrows():
        ipu = InterPausalUnit(
        start=row["start"],
        end=row["end"],
        features_values=row[columns].to_dict()
        )
        if row["speaker"]=="indy":
            indy_ipu.append(ipu)
        else:
            subject_ipu.append(ipu)
    for feature in columns:
        applying_indy_ipus = [ipu for ipu in indy_ipu if type(ipu.feature_value(feature)) is float]
        applying_subject_ipus = [ipu for ipu in subject_ipu if type(ipu.feature_value(feature)) is float]
        try:
            indy_ts= TimeSeries(
                    interpausal_units=applying_indy_ipus,
                    feature=feature,
                    method='knn',
                    k=3,
                    )
            subject_ts = TimeSeries(
                    interpausal_units=applying_subject_ipus,
                    feature=feature,
                    method='knn',
                    k=3,
                    )
        except ValueError:
            for metric in metrics:
                scores[f"{feature}_{metric}"] = np.nan
            continue
        for metric in metrics:
            scores[f"{feature}_{metric}"] = calculate_metric(
                                metric,
                                indy_ts,
                                subject_ts,
                            )
    return scores
                        

