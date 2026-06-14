#!/usr/bin/env python3
import sys
import pandas as pd
from nepalitextcleaner import NepaliTextCleaner   # note: filename nepalitextcleaner.py

def main():
    # Expect: python cleaning.py <input.txt> <output.txt>
    if len(sys.argv) < 3:
        print("Usage: python cleaning.py <input.txt> <output.txt>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Reading TXT from: {input_path}")

    # Read input TXT: one line = one row of text
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    # Put into a DataFrame with column name 'text'
    df = pd.DataFrame({"text": lines})

    # Use your existing cleaner
    cleaner = NepaliTextCleaner(
        keep_english=True,
        keep_numbers=True,
    )

    df_cleaned = cleaner.clean_dataframe(
        df,
        text_column="text",
        remove_duplicates=True,
        remove_incomplete=True,
        inplace=False,
    )

    # Write cleaned lines back to TXT
    print(f"Writing cleaned TXT to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in df_cleaned["text"]:
            f.write(line + "\n")

    print("Done.")

if __name__ == "__main__":
    main()
