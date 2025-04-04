import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("questionaire_score.csv", index_col="sessionid")
X = df[["ext", "open", "agree", "neu", "int", "nars", "ras", "sas"]]
y = df["approach_reject"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

importance = pd.Series(model.coef_, index=X.columns)
importance.sort_values(ascending=False).plot.bar()


import statsmodels.api as sm

X_const = sm.add_constant(X)  # 切片を追加
model = sm.OLS(y, X_const).fit()
print(model.summary())
plt.show()