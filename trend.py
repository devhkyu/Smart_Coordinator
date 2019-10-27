import pandas as pd

df = pd.read_csv('pred_camscon.csv', index_col=[0])
a = df['class_ids'][0:1]
print(df)