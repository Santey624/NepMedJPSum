import pandas as pd

input_file="/Volumes/research112/pre-trainingdata/Japanese/CC-100-jp/cc100ja/cc100_ja_0.csv"
df = pd.read_csv(input_file)
view= df.head()
print(view)