import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import os
from datetime import datetime

# Input directory containing many CSVs
input_dir = "/Volumes/research112/pre-trainingdata/Nepali/Nepalidata/Nepalidata2.csv"

# Timestamped output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"/Volumes/research112/pre-trainingdata/Nepali/Nepalidata/Nepalidata2splitted/text_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

# 1) Read ALL CSV files in the directory
ddf = dd.read_csv(
    os.path.join(input_dir, "*.csv"),
    encoding="latin1",          # <--- important change
    assume_missing=True,
    dtype={
        "file_name": "object",
        "article_id": "object",
        "front": "object",
        "body": "object",
    },
    # If you still get parsing errors (NOT decode errors), you can also add:
    # on_bad_lines="skip",      # pandas>=1.3
)

print("Columns Dask sees:", list(ddf.columns))
# 2) Keep only 'front' and 'body'
ddf_small = ddf[["front", "body"]]

# 3) Repartition to get ~500MB partitions
ddf_small_500 = ddf_small.repartition(partition_size="500MB")

# 4) Write out: one CSV per partition
out_pattern = os.path.join(output_folder, f"part_{timestamp}_*.csv")

with ProgressBar():
    ddf_small_500.to_csv(out_pattern, index=False)
