import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

demo_df = pd.read_csv("questionaire/demographic.csv", index_col="sessionid")
nars_df = pd.read_csv("questionaire/nars-ras.csv", index_col="sessionid")
tipi_df = pd.read_csv("questionaire/tipij.csv", index_col="sessionid")
sas_df = pd.read_csv("questionaire/sas.csv", index_col="sessionid")
post_df = pd.read_csv("questionaire/post.csv", index_col="sessionid")
df = pd.merge(demo_df, nars_df, how="left", left_index=True, right_index=True)
df = pd.merge(df, tipi_df, how="left", left_index=True, right_index=True)
df = pd.merge(df, sas_df, how="left", left_index=True, right_index=True)
df = pd.merge(df, post_df, how="left", left_index=True, right_index=True)
df.to_csv("questionaire_result_all.csv")

df = pd.read_csv("questionaire_result_all.csv", index_col="sessionid")
score_df = pd.DataFrame()
score_df["naturalness"] = df["naturalness"]
score_df["humanlike"] = df["humanlike"]
score_df["will"] = df["will"]
score_df["continuous"] = df["continuous"]
score_df["intimacy"] = df["intimacy"]
score_df["frank_reject"] = df["frank_unconfort"]+df["frank_fear"]+df["frank_rude"]
score_df["frank_avoid"] = df["frank_avoid"]
score_df["approach_reject"] = df["approach_unconfort"]+df["approach_fear"]+df["approach_rude"]
score_df["approach_avoid"] = df["appraoch_avoid"]
score_df["partner_reject"] = df["partner_unconfort"]+df["partner_fear"]+df["partner_rude"]
score_df["partner_avoid"] = df["partner_avoid"]
score_df["ext"] = df["ext1"]+8-df["ext2"]
score_df["open"] = df["open1"]+8-df["open2"]
score_df["agree"] = df["agr1"]+8-df["agr2"]
score_df["neu"] = df["neu1"]+8-df["neu2"]
score_df["int"] = df["ind1"]+8-df["ind2"]
score_df["nars"] = df[[f"n{i}" for i in range(1, 14)]].sum(axis=1)
score_df["ras"] = df[[f"r{i}" for i in range(1,11)]].sum(axis=1)
score_df["sas"] = df[[str(i) for i in range(1,20)]].sum(axis=1)
score_df.to_csv("questionaire_score.csv")
score_df_corr = score_df.corr()[["ext", "open", "agree", "neu", "int", "nars", "ras", "sas"]]
score_df_corr = score_df_corr.loc[["naturalness", "humanlike", "will", "continuous", "intimacy", "frank_reject", "frank_avoid", "approach_reject", "approach_avoid", "partner_reject", "partner_avoid"]]
#sns.heatmap(score_df_corr, cmap="coolwarm", annot=True)
#plt.show()
