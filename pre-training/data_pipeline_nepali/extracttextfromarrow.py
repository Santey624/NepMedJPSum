import os
from datasets import Dataset
import glob

# Configuration
input_dir = "/Volumes/research112/pre-trainingdata/Nepali/Nepalidata/train"
output_dir = "/Volumes/research112/pre-trainingdata/Nepali/Nepalidata/extracted_articles_train"
chunk_size = 500 * 1024 * 1024  # 500 MB

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Find all arrow files
arrow_files = sorted(glob.glob(f"{input_dir}/data-*.arrow"))
print(f"Found {len(arrow_files)} arrow files\n")

# Initialize
file_index = 1
current_size = 0
out_path = os.path.join(output_dir, f"articles_part_{file_index:03}.txt")
f_out = open(out_path, "w", encoding="utf-8")

total_articles = 0
articles_in_current_file = 0
skipped = 0

print("Extracting articles...\n")

for arrow_file in arrow_files:
    print(f"Processing: {os.path.basename(arrow_file)}")
    
    try:
        # Read arrow file
        dataset = Dataset.from_file(arrow_file)
        df = dataset.to_pandas()
        
        # Extract articles
        for idx, row in df.iterrows():
            try:
                article = str(row['Article']).strip()
                
                # Skip if empty or too short
                if not article or article == 'nan' or len(article) < 50:
                    skipped += 1
                    continue
                
                # Add article with double newline separator
                text_to_write = article + "\n\n"
                text_bytes = text_to_write.encode("utf-8")
                
                # Check if need to create new file
                if current_size + len(text_bytes) > chunk_size and current_size > 0:
                    f_out.close()
                    print(f"  ✓ Created: articles_part_{file_index:03}.txt "
                          f"({current_size / (1024*1024):.2f} MB, "
                          f"{articles_in_current_file:,} articles)")
                    
                    # Start new file
                    file_index += 1
                    current_size = 0
                    articles_in_current_file = 0
                    out_path = os.path.join(output_dir, f"articles_part_{file_index:03}.txt")
                    f_out = open(out_path, "w", encoding="utf-8")
                
                # Write to file
                f_out.write(text_to_write)
                current_size += len(text_bytes)
                articles_in_current_file += 1
                total_articles += 1
                
            except Exception as e:
                skipped += 1
                continue
        
        print(f"  Processed: {len(df):,} rows")
        
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")
        continue

# Close last file
f_out.close()
print(f"  ✓ Created: articles_part_{file_index:03}.txt "
      f"({current_size / (1024*1024):.2f} MB, "
      f"{articles_in_current_file:,} articles)")

print(f"\n{'='*70}")
print("EXTRACTION COMPLETE!")
print(f"{'='*70}")
print(f"Output directory: {output_dir}")
print(f"Total files created: {file_index}")
print(f"Total articles extracted: {total_articles:,}")
print(f"Skipped (invalid/short): {skipped:,}")
print(f"{'='*70}")
