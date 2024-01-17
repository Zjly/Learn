import pandas as pd

df = pd.read_feather("train.feather")

df["label"] = df["api_sequence"].apply(lambda x: x[-1])
df["api_sequence"] = df["api_sequence"].apply(lambda x: x[:-1])
df.drop("question", axis=1, inplace=True)

df.to_feather("dataset.feather")
df.head(10000).to_feather("dataset_10000.feather")
