import pandas as pd
import json
from pathlib import Path

input_file = Path(r"C:\Users\gaire\Desktop\medicaldataset\pubmed_baseline\part_20251113_133518_67.csv")
output_file = Path(r"C:\Users\gaire\Desktop\pubmedjsonl\part67_new.jsonl")  # Changed name

output_file.parent.mkdir(parents=True, exist_ok=True)

print(f"Processing: {input_file.name} → {output_file.name}")

try:
    try:
        df = pd.read_csv(input_file, 
                        header=None, 
                        encoding='utf-8',
                        on_bad_lines='warn',
                        engine='python')
    except UnicodeDecodeError:
        print(f"  ⚠ UTF-8 failed, trying latin1 encoding...")
        df = pd.read_csv(input_file, 
                        header=None, 
                        encoding='latin1',
                        on_bad_lines='warn',
                        engine='python')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            text = ' '.join([str(v) for v in row.values if pd.notna(v)])
            json.dump({'text': text}, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✓ Wrote {len(df)} lines")
    print(f"✅ Success!")
    
except Exception as e:
    print(f"✗ ERROR: {e}")