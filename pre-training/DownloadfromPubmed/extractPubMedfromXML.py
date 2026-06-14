import xml.etree.ElementTree as ET
import csv
import os
import gzip
import multiprocessing
from typing import List, Dict, Optional
from tqdm import tqdm  # progress bar

# --- CONFIGURATION ---
NUM_CORES = min(multiprocessing.cpu_count(), 6)
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_folder = os.path.join(script_dir, 'pubmed_baseline')
csv_file = '/Volumes/research112/pubmed_articles.csv'
processed_log_file = os.path.join(script_dir, 'processed_files.txt')
# --- END CONFIGURATION ---

ArticleData = Dict[str, Optional[str]]

def extract_data_from_file(filename: str, xml_folder: str) -> List[ArticleData]:
    xml_path = os.path.join(xml_folder, filename)
    extracted_articles: List[ArticleData] = []

    try:
        with gzip.open(xml_path, 'rt', encoding='utf-8') as xml_file:
            context = ET.iterparse(xml_file, events=("end",))
            for event, article in context:
                if article.tag == 'PubmedArticle':
                    pmid_element = article.find('.//MedlineCitation/PMID')
                    title_element = article.find('.//Article/ArticleTitle')
                    abstract_elements = article.findall('.//Article/Abstract/AbstractText')

                    pmid = pmid_element.text if pmid_element is not None else None
                    title = title_element.text if title_element is not None else None
                    abstract = ' '.join([elem.text for elem in abstract_elements if elem.text]) if abstract_elements else None

                    if pmid and title:
                        extracted_articles.append({'pmid': pmid, 'title': title, 'abstract': abstract})

                    article.clear()
        return extracted_articles
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []

def worker(filename: str, xml_folder: str, queue: multiprocessing.Queue):
    data = extract_data_from_file(filename, xml_folder)
    queue.put((filename, data))

def main():
    # Load list of already processed files (for resume)
    processed_files = set()
    if os.path.exists(processed_log_file):
        with open(processed_log_file, 'r') as f:
            processed_files = set(line.strip() for line in f)

    # Collect files to process
    files_to_process = [f for f in os.listdir(xml_folder) if f.endswith('.xml.gz') and f not in processed_files]
    total_files = len(files_to_process)
    if total_files == 0:
        print("No new files to process.")
        return

    manager = multiprocessing.Manager()
    queue = manager.Queue()

    # Open CSV in append mode
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['pmid', 'title', 'abstract'])
        if os.stat(csv_file).st_size == 0:
            writer.writeheader()

        pool = multiprocessing.Pool(NUM_CORES)
        for filename in files_to_process:
            pool.apply_async(worker, args=(filename, xml_folder, queue))

        pool.close()

        total_articles = 0
        completed_files = set()

        # Progress bar using tqdm
        with tqdm(total=total_files, desc="Processing files") as pbar:
            while len(completed_files) < total_files:
                try:
                    filename, articles = queue.get(timeout=10)
                    if articles:
                        writer.writerows(articles)
                        total_articles += len(articles)
                    completed_files.add(filename)

                    # Update processed log
                    with open(processed_log_file, 'a') as log_f:
                        log_f.write(filename + '\n')

                    pbar.update(1)
                except multiprocessing.queues.Empty:
                    # Check if pool is alive
                    if not any(p.is_alive() for p in pool._pool):
                        break

        pool.join()

    print(f"\nExtraction complete! Total articles extracted: {total_articles}")
    print(f"CSV file saved at {csv_file}")

if __name__ == '__main__':
    main()
