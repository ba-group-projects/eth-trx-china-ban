import pandas as pd

dfm = pd.read_csv("data/data_example.csv")
print(dfm[dfm["from_address"]==dfm["to_address"]])
print("1")