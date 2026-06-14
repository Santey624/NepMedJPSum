import os
import tarfile
import xml.etree.ElementTree as ET
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

TAR_DIR = "/Users/santoshgairesharma/Documents/NepMedJP/pmc_noncomm_xml"
OUTPUT_CSV = "/Volumes/research112/pmc_noncomm_full_articles.csv"

def extract_article_data(article, file_name):
    def get_text(elem):
        return ''.join(elem.itertext()) if elem is not None else ''
    article_id_elem = article.find('.//article-id')
    article_id = article_id_elem.text if article_id_elem is not None else None
    front = get_text(article.find('front'))
    body = get_text(article.find('body'))
    back = get_text(article.find('back'))
    return {'file_name': file_name, 'article_id': article_id, 'front': front, 'body': body, 'back': back}

def process_tar_file(tar_path):
    """Return a list of all articles in this tar.gz"""
    articles = []
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.xml'):
                f = tar.extractfile(member)
                if f:
                    tree = ET.parse(f)
                    root = tree.getroot()
                    for article in root.findall('article'):
                        articles.append(extract_article_data(article, os.path.basename(tar_path)))
    return articles

if __name__ == "__main__":
    start_time = time.time()

    # Prepare CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ['file_name', 'article_id', 'front', 'body', 'back']
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        tar_files = [os.path.join(TAR_DIR, tf) for tf in os.listdir(TAR_DIR) if tf.endswith('.tar.gz')]

        # Use multiprocessing safely
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_tar_file, tf): tf for tf in tar_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tar.gz files"):
                articles = future.result()
                writer.writerows(articles)

    end_time = time.time()
    print(f"\n✅ Extraction complete! CSV saved at: {OUTPUT_CSV}")
    print(f"⏱ Total processing time: { (end_time - start_time)/60:.2f} minutes")
