import os
import ftplib
import hashlib
from urllib import request

# === Configuration ===
GCS_BUCKET_PATH = "gs://pretraining-datasets-nepmedjp"
LOCAL_DATASET_FOLDER = "/Users/santoshgairesharma/Documents/Pre-training Dataset collection/Nepali"

# Create a subfolder inside your Nepali dataset folder
PUBMED_FOLDER = os.path.join(LOCAL_DATASET_FOLDER, "pubmed_baseline")
os.makedirs(PUBMED_FOLDER, exist_ok=True)

FTP_SERVER = "ftp.ncbi.nlm.nih.gov"
FTP_PATH = "/pubmed/baseline/"
RETRIES = 3


def verify_md5(file_path, expected_md5):
    """Check MD5 hash of downloaded file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_md5


def download_file(filename):
    """Download a single file with retries."""
    url = f"https://{FTP_SERVER}{FTP_PATH}{filename}"
    local_path = os.path.join(PUBMED_FOLDER, filename)

    for attempt in range(RETRIES):
        try:
            print(f"⬇️  Downloading: {filename} (attempt {attempt + 1})")
            request.urlretrieve(url, local_path)
            print(f"✅ Saved: {local_path}")
            return True
        except Exception as e:
            print(f"⚠️  Error downloading {filename}: {e}")
    print(f"❌ Failed after {RETRIES} attempts: {filename}")
    return False


# === Connect to FTP and list files ===
print("📡 Connecting to PubMed FTP server...")
with ftplib.FTP(FTP_SERVER) as ftp:
    ftp.login()
    ftp.cwd(FTP_PATH)
    files = ftp.nlst()

# === Separate data and md5 files ===
xml_files = [f for f in files if f.endswith(".xml.gz")]
md5_files = [f for f in files if f.endswith(".md5")]

print(f"Found {len(xml_files)} data files and {len(md5_files)} md5 files.")

# === Download MD5 files first ===
for f in md5_files:
    download_file(f)

# === Build MD5 dictionary ===
md5_map = {}
for f in md5_files:
    path = os.path.join(PUBMED_FOLDER, f)
    if not os.path.exists(path):
        continue
    with open(path) as mf:
        parts = mf.read().strip().split()
        if len(parts) == 2:
            md5_map[parts[1]] = parts[0]

# === Download XML.GZ files ===
for f in xml_files:
    local_path = os.path.join(PUBMED_FOLDER, f)
    if os.path.exists(local_path):
        print(f"⏩ Skipping (already exists): {f}")
        continue
    if download_file(f):
        expected_md5 = md5_map.get(f)
        if expected_md5:
            print(f"🔍 Verifying {f} ...")
            if verify_md5(local_path, expected_md5):
                print(f"✅ MD5 OK: {f}")
            else:
                print(f"❌ MD5 mismatch: {f}")

print("🎉 Download complete!")
print(f"📁 All files saved in: {PUBMED_FOLDER}")
