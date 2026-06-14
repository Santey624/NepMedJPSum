import os
import tarfile
import xml.etree.ElementTree as ET
import csv
import time

TAR_FILE = "/Users/santoshgairesharma/Documents/NepMedJP/pmc_noncomm_xml/oa_noncomm_xml.PMC012xxxxxx.baseline.2025-06-26.tar.gz"
OUTPUT_CSV = "/Volumes/research112/text136.csv"

def get_text(elem):
    """Extract text with paragraph breaks preserved"""
    if elem is None:
        return ''
    
    # Get text from paragraphs separately to preserve structure
    paragraphs = []
    for p in elem.iter():
        if p.tag.endswith('p'):
            text = ''.join(p.itertext()).strip()
            if text:
                paragraphs.append(text)
    
    # If no paragraphs found, fall back to all text
    if not paragraphs:
        return ''.join(elem.itertext()).strip()
    
    return '\n\n'.join(paragraphs)

def extract_article_data(article, file_name):
    # Try several ways to get article-id
    article_id = None
    for elem in article.iter():
        if elem.tag.endswith("article-id") and elem.text:
            article_id = elem.text
            break

    front = body = ""
    
    # Find front and body elements directly
    for elem in article.iter():
        tag = elem.tag.lower()
        if tag.endswith("front") and not front:
            front = get_text(elem)
        elif tag.endswith("body") and not body:
            body = get_text(elem)
            # Stop after finding body to avoid re-processing
            if front:
                break

    return {
        'file_name': file_name, 
        'article_id': article_id, 
        'front': front, 
        'body': body
    }

def process_tar_file(tar_path, writer):
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.xml'):
                f = tar.extractfile(member)
                if f:
                    try:
                        tree = ET.parse(f)
                        root = tree.getroot()
                        for article in root.iter():
                            if article.tag.lower().endswith("article"):
                                writer.writerow(extract_article_data(article, os.path.basename(tar_path)))
                                break  # Process only first article element
                    except Exception as e:
                        print(f"Error processing {member.name}: {e}")

if __name__ == "__main__":
    start_time = time.time()

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ['file_name', 'article_id', 'front', 'body']

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        print(f"Processing {TAR_FILE} ...")
        process_tar_file(TAR_FILE, writer)

    end_time = time.time()
    print(f"\n✅ Extraction complete! CSV saved at: {OUTPUT_CSV}")
    print(f"⏱ Time taken: {(end_time - start_time):.2f} seconds")