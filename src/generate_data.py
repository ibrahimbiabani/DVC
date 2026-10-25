from pathlib import Path
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500, n_features=5, n_informative=3,
    n_redundant=0, random_state=42
)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
df["target"] = y
Path("data").mkdir(exist_ok=True)
df.to_csv("data/train.csv", index=False)
print("Dataset generated:", df.shape)
