import os
import pandas as pd
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from threading import Lock

# === CONFIG ===
train_csv = "train.csv"
test_csv = "test.csv"
image_dir = "image_cache"
MAX_WORKERS = 20   # ⚡ tuned for Legion 7i Pro NVMe SSD

# === LOAD DATA ===
df_train = pd.read_csv(train_csv, usecols=["sample_id", "image_link"])
df_test = pd.read_csv(test_csv, usecols=["sample_id", "image_link"])
df = pd.concat([df_train, df_test], ignore_index=True)

def extract_filename(url: str) -> str:
    return os.path.basename(urlparse(url).path)

df["filename"] = df["image_link"].apply(extract_filename)
mapping = dict(zip(df["filename"], df["sample_id"]))

all_files = os.listdir(image_dir)
print(f"📦 Found {len(all_files)} files in '{image_dir}'")

# === COUNTERS ===
stats = {"renamed": 0, "already_named": 0, "missing": 0}
lock = Lock()

def rename_file(fname):
    """Thread-safe rename logic"""
    base_name, ext = os.path.splitext(fname)
    old_path = os.path.join(image_dir, fname)

    if fname.startswith("."):
        return
    if base_name.isdigit():
        with lock:
            stats["already_named"] += 1
        return

    sample_id = mapping.get(fname)
    if sample_id is None:
        for k in mapping.keys():
            if os.path.splitext(k)[0] == base_name:
                sample_id = mapping[k]
                break
    if sample_id is None:
        with lock:
            stats["missing"] += 1
        return

    new_path = os.path.join(image_dir, f"{sample_id}{ext.lower()}")
    if os.path.exists(new_path):
        with lock:
            stats["already_named"] += 1
        return

    try:
        os.rename(old_path, new_path)
        with lock:
            stats["renamed"] += 1
    except Exception as e:
        print(f"⚠️ Error renaming {fname} → {sample_id}{ext}: {e}")

# === PARALLEL EXECUTION ===
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    list(tqdm(executor.map(rename_file, all_files),
              total=len(all_files),
              desc=f"⚙️ Renaming using {MAX_WORKERS} threads"))

# === SUMMARY ===
print("\n📊 SUMMARY REPORT")
print(f"✅ Renamed successfully : {stats['renamed']}")
print(f"🌀 Already correct name : {stats['already_named']}")
print(f"⚠️ Missing in CSV map   : {stats['missing']}")
print(f"📁 Total files checked  : {len(all_files)}")
print("🏁 Parallel renaming complete!")
