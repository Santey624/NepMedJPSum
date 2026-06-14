import os
import glob

# ====== CONFIGURE THESE PATHS ======
input_dir = "/volumes/research112/pre-trainingdata/Japanese/CC-100-jp/cc100ja"
output_dir = os.path.join(input_dir, "split_500mb")  # new folder for all split files
# ==================================

MAX_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB

os.makedirs(output_dir, exist_ok=True)

def split_csv_file(input_file: str, output_dir: str, max_size_bytes: int) -> None:
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    print(f"Processing {input_file} ...")

    with open(input_file, "r", encoding="utf-8") as src:
        header = src.readline()
        if not header:
            print(f"  Skipping {input_file} (empty file).")
            return

        header_bytes = header.encode("utf-8")

        part_index = 0
        current_size = 0
        out = None

        def open_new_part():
            nonlocal part_index, current_size, out
            # close previous file if open
            if out is not None:
                out.close()
            part_filename = f"{base_name}_part{part_index}.csv"
            part_path = os.path.join(output_dir, part_filename)
            out = open(part_path, "wb")
            out.write(header_bytes)
            current_size = len(header_bytes)
            print(f"  -> Writing {part_filename}")
            part_index += 1

        # Open first part file
        open_new_part()

        for line in src:
            line_bytes = line.encode("utf-8")

            # If adding this line would exceed the 500 MB limit, start a new part
            if current_size + len(line_bytes) > max_size_bytes:
                open_new_part()

            out.write(line_bytes)
            current_size += len(line_bytes)

        if out is not None:
            out.close()


# Find all CSV files in the input directory
csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

for input_file in csv_files:
    split_csv_file(input_file, output_dir, MAX_SIZE_BYTES)

print("Done! All split files are in:", output_dir)
