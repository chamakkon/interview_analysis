import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("questionaire_score.csv", index_col="sessionid")
X = df[["ext", "open", "agree", "neu", "int", "nars", "ras", "sas"]]
y = df["partner_reject"]
y = [1 if x > 4 else 0 for x in y.tolist()]
print(y.count(1), y.count(0))


# データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ロジスティック回帰モデル学習
model = LogisticRegression(multi_class='ovr', solver='lbfgs')
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # クラス1の確率

# 精度評価出力
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")
print(y_test)
importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)
importance.sort_values(ascending=False).plot.bar()
plt.show()