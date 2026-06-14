import fasttext, json
model = fasttext.load_model("lid.176.bin")

with open("part0nepalitranslated_deduped.jsonl") as fin, open("removed_samples.jsonl", "w") as fout:
    count = 0
    for line in fin:
        rec = json.loads(line)
        text = rec["text"].replace("\n", " ").strip()
        pred = model.predict(text, k=3)
        if pred[0][0] != "__label__ne":
            fout.write(line)
            count += 1
            if count >= 20:
                break