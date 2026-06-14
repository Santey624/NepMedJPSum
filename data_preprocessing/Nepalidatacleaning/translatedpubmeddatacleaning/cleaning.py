import fasttext
import json
import time

model = fasttext.load_model("lid.176.bin")


def get_lang_info(text: str):
    text = text.replace("\n", " ").strip()
    predictions = model.predict(text, k=3)
    labels = predictions[0]
    scores = predictions[1]
    label_score = dict(zip(labels, scores))
    ne_score = label_score.get("__label__ne", 0.0)
    hi_score = label_score.get("__label__hi", 0.0)

    is_nepali = True
    if labels[0] == "__label__hi" and hi_score > ne_score * 2:
        is_nepali = False

    return is_nepali, labels[0], round(ne_score, 4), round(hi_score, 4)


def count_lines(path):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def main():
    input_path = "/workspace/Nepmedjp/datapreporcessing/Nepalidatacleaning/translatedpubmeddatacleaning/Part10output_01_deduped.jsonl"
    output_path = "/workspace/Nepmedjp/datapreporcessing/Nepalidatacleaning/translatedpubmeddatacleaning/Part10output_01_filtered.jsonl"
    removed_path = "/workspace/Nepmedjp/datapreporcessing/Nepalidatacleaning/translatedpubmeddatacleaning/Part10output_01_removed.txt"

    print("Counting total lines...")
    total = count_lines(input_path)
    print(f"Total records: {total}\n")

    processed = 0
    kept = 0
    removed = 0
    start_time = time.time()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout, \
         open(removed_path, "w", encoding="utf-8") as fremoved:

        fremoved.write(f"{'#':<8} {'TOP_LANG':<16} {'NE_SCORE':<10} {'HI_SCORE':<10} TEXT\n")
        fremoved.write("=" * 120 + "\n")

        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "")
            is_nep, top_lang, ne_score, hi_score = get_lang_info(text)

            if is_nep:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            else:
                removed += 1
                fremoved.write(f"{removed:<8} {top_lang:<16} {ne_score:<10} {hi_score:<10} {text[:200]}\n")

            processed += 1

            if processed % 1000 == 0 or processed == total:
                elapsed = time.time() - start_time
                speed = processed / elapsed
                remaining = (total - processed) / speed if speed > 0 else 0
                pct = (processed / total) * 100
                print(
                    f"[{pct:5.1f}%] {processed}/{total} | "
                    f"Kept: {kept} | "
                    f"Removed: {removed} | "
                    f"Speed: {speed:.0f} rec/s | "
                    f"Elapsed: {format_time(elapsed)} | "
                    f"ETA: {format_time(remaining)}"
                )

    elapsed = time.time() - start_time
    print(f"\nDone! Processed {processed} records in {format_time(elapsed)}")
    print(f"Kept: {kept} | Removed: {removed}")
    print(f"Removed items saved to: {removed_path}")


if __name__ == "__main__":
    main()