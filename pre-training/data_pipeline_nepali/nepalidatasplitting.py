import pandas as pd
import os
import time
import csv
from datetime import datetime

# CRITICAL: Increase CSV field size limit
csv.field_size_limit(10485760)  # 10 MB limit (or use csv.field_size_limit(2147483647) for max)

input_path = "/Volumes/research112/pre-trainingdata/Nepali/Nepalidata/Nepalidata2.csv"
output_dir = "/Volumes/research112/pre-trainingdata/Nepali/Nepalidata/text_only_chunks"
chunk_size = 500 * 1024 * 1024  # 500 MB in bytes

os.makedirs(output_dir, exist_ok=True)

start_time = time.time()
start_dt = datetime.now()
print(f"Extraction started at: {start_dt:%Y-%m-%d %H:%M:%S}")

chunk_iter = pd.read_csv(
    input_path, 
    chunksize=10000, 
    on_bad_lines='skip', 
    engine='python'
)

file_index = 1
current_size = 0
out_path = os.path.join(output_dir, f"nepali_text_part_{file_index:03}.txt")
f_out = open(out_path, "w", encoding="utf-8")

total_articles = 0

try:
    for chunk_df in chunk_iter:
        for idx, row in chunk_df.iterrows():
            text = str(row['text']) if pd.notna(row['text']) else ""
            text = text.strip()
            
            if not text or text == 'nan':
                continue
            
            # Just text with double newline between articles
            text_to_write = text + "\n\n"
            text_bytes = text_to_write.encode("utf-8")
            
            if current_size + len(text_bytes) > chunk_size and current_size > 0:
                f_out.close()
                print(f"Chunk {file_index}: {current_size / (1024*1024):.2f} MB, {total_articles} articles")
                
                file_index += 1
                current_size = 0
                total_articles = 0
                out_path = os.path.join(output_dir, f"nepali_text_part_{file_index:03}.txt")
                f_out = open(out_path, "w", encoding="utf-8")
            
            f_out.write(text_to_write)
            current_size += len(text_bytes)
            total_articles += 1

except Exception as e:
    print(f"Error occurred: {e}")
    print(f"Stopped at article count: {total_articles}")

finally:
    f_out.close()
    print(f"Last chunk {file_index}: {current_size / (1024*1024):.2f} MB, {total_articles} articles")

end_time = time.time()
elapsed = end_time - start_time
print(f"\nTotal chunks: {file_index}")
print(f"Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")