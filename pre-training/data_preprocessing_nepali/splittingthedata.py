import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import os
from datetime import datetime

# Input file
inputfile = "/Volumes/research112/medicaldataset/Pubmed_Central"

# Timestamped output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"/Volumes/research112/medicaldataset/Pubmed_CentralDownsizedone/text_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

# Read CSV with Dask
ddf = dd.read_csv(
    inputfile,
    blocksize="500MB",
    dtype={"abstract": "object"}  # keep this to avoid the dtype error
)

# ⚠️ Keep only the columns you want
ddf_small = ddf[["front", "body"]]

# Save with a progress bar
with ProgressBar():
    ddf_small.to_csv(f"{output_folder}/part_{timestamp}_*.csv", index=False)
