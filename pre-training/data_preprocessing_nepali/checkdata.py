import pandas as pd
import time

inputfile = "/Volumes/research112/medicaldataset/Pubmed_Central/text1.csv"
chunksize = 10**6  # 1 million rows per chunk

# Start measuring total time
start_total = time.time()

for i, chunk in enumerate(pd.read_csv(inputfile, chunksize=chunksize, encoding='utf-8')):
    start_chunk = time.time()  # start time for this chunk

    # Filter rows where 'abstract' is not empty
    abstracts_only = chunk[chunk['abstract'].notna()]
    print(abstracts_only.head())  # preview first 5 rows

    end_chunk = time.time()  # end time for this chunk
    print(f"Chunk {i} processed in {end_chunk - start_chunk:.2f} seconds")

    break  # only preview first chunk

end_total = time.time()
print(f"Total processing time: {end_total - start_total:.2f} seconds")