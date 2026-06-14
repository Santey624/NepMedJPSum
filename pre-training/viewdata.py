import pandas as pd

csv_path = "/Volumes/research112/medicaldataset/Pubmed_Central/text99.csv"

df = pd.read_csv(csv_path)

print(df[["body"]].head())       # Show first 5 rows

# Select only the 'front' and 'body' columns
new_df = df[["body"]]

# Save to a new CSV file
out_path = "/Volumes/research112/medicaldataset/Pubmed_Central_cleaned/text99.csv"
new_df.to_csv(out_path, index=False)