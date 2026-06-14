import pandas as pd

input_dir = "/Volumes/research112/pre-trainingdata/Nepali/Nepalidata/Nepalidata2splitted/Nepalidata2_part_001.csv"

df = pd.read_csv(input_dir, on_bad_lines='skip', engine='python')

# Show one article completely
print("=" * 80)
sample = df.sample(1).iloc[0]

print(f"Category: {sample['catagory']} ({sample['clean_categories']})")
print(f"Date: {sample['date']}")
print(f"Heading: {sample['heading']}")
print(f"\nFull Text:\n{sample['text']}")
print(f"\nLink: {sample['link']}")
print("=" * 80)