import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "age": rng.normal(35, 10, n).round(1),
    "income": rng.lognormal(mean=10, sigma=0.5, size=n),
    "city": rng.choice(["SF", "NYC", "LA", "SEA", None], size=n, p=[.25,.25,.2,.2,.1]),
    "device": rng.choice(["ios", "android", "web"], size=n),
    "signup_days": rng.integers(0, 365, size=n),
    "clicked": rng.choice([0,1], size=n, p=[.6,.4])
})
df.loc[rng.choice(df.index, 60, replace=False), "income"] = np.nan
df.loc[rng.choice(df.index, 40, replace=False), "age"] = np.nan

y = df["clicked"].astype(int)
X = df.drop(columns=["clicked"])

num_col = X.select_dtypes(include=np.number).columns
X[num_col] = X[num_col].apply(lambda x: x.fillna(x.median()))
cat_col = X.select_dtypes(include="object").columns
X[cat_col] = X[cat_col].apply(lambda x: x.fillna("UNKNOWN"))

X_encoded = pd.get_dummies(X, columns=cat_col, prefix=cat_col, drop_first=True)

X_tr, X_te, y_tr, y_te = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)

clf = LogisticRegression(max_iter=1000)

clf.fit(X_tr, y_tr)
print(classification_report(y_te, clf.predict(X_te)))