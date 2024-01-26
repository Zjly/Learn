import pandas as pd

df = pd.read_feather("test.feather")

df["label"] = df["api_sequence"].apply(lambda x: x[-1])
df["api_sequence"] = df["api_sequence"].apply(lambda x: x[:-1])
df.drop("question", axis=1, inplace=True)

df.head(1000).to_feather("dataset_test_1000.feather")
# df.head(100000).to_feather("dataset_100000.feather")
