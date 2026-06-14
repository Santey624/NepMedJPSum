#!/usr/bin/env python3
import os, tarfile, json
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = "/path/to/pmc_tar_gz"
OUTPUT_DIR = "/path/to/pmc_text"  # text files here
META_FILE = os.path.join(OUTPUT_DIR, "metadata.jsonl")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_article_xml(xml_bytes):
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return None
    # very simple extraction — adjust to PMC XML namespace if needed
    ns = {"x": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    def find_text(path):
        e = root.find(path, ns)
        return (e.text or "").strip() if e is not None else ""
    title = find_text(".//article-title")
    abstract = " ".join([p.text.strip() for p in root.findall(".//abstract//p", ns) if p.text])
    body_paragraphs = [p.text.strip() for p in root.findall(".//body//p", ns) if p.text]
    body = "\n\n".join(body_paragraphs)
    # collect metadata
    journal = find_text(".//journal-title")
    year = find_text(".//pub-date/year")
    pmcid = find_text(".//article-id[@pub-id-type='pmcid']")
    authors = []
    for contrib in root.findall(".//contrib[@contrib-type='author']", ns):
        name = "".join([t.text or "" for t in contrib.findall(".//surname", ns) + contrib.findall(".//given-names", ns)])
        if name: authors.append(name)
    return {"pmcid": pmcid or None, "title": title, "abstract": abstract,
            "body": body, "journal": journal, "year": year, "authors": authors}

def process_tar(tar_path):
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile(): continue
            name = os.path.basename(member.name)
            if not name.lower().endswith(".xml"): continue
            out_txt = os.path.join(OUTPUT_DIR, name.replace(".xml", ".txt"))
            # resumable: skip if exists
            if os.path.exists(out_txt): 
                continue
            try:
                f = tf.extractfile(member)
                xml_bytes = f.read()
                data = parse_article_xml(xml_bytes)
                if data is None:
                    with open(os.path.join(OUTPUT_DIR, "errors.log"), "a") as el:
                        el.write(f"parse_error\t{tar_path}\t{member.name}\n")
                    continue
                # write text
                with open(out_txt, "w", encoding="utf-8") as w:
                    w.write(data["title"] + "\n\n" + data["abstract"] + "\n\n" + data["body"])
                # write metadata
                meta = {
                    "pmcid": data["pmcid"],
                    "filename": os.path.basename(out_txt),
                    "title": data["title"],
                    "journal": data["journal"],
                    "year": data["year"],
                    "authors": data["authors"],
                    "source_tar": os.path.basename(tar_path)
                }
                with open(META_FILE, "a", encoding="utf-8") as m:
                    m.write(json.dumps(meta, ensure_ascii=False) + "\n")
            except Exception as e:
                with open(os.path.join(OUTPUT_DIR, "errors.log"), "a") as el:
                    el.write(f"error\t{tar_path}\t{member.name}\t{str(e)}\n")

if __name__ == "__main__":
    tar_files = sorted([os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith(".tar.gz")])
    for tar in tqdm(tar_files):
        process_tar(tar)
