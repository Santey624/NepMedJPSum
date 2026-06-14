import pandas as pd

input_file = "/Volumes/research112/pre-trainingdata/Japanese/CC-100-jp/cc100ja/cc100_ja_0.csv"
output_file = "/Volumes/research112/pre-trainingdata/Japanese/CC-100-jp/cc100ja/cc100_ja_0.txt"

df = pd.read_csv(input_file)

with open(output_file, 'w', encoding='utf-8') as f:
    for text in df['text']:
        f.write(str(text) + '\n')

print(f"Saved {len(df)} lines to {output_file}")