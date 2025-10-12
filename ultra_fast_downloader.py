"""
Ultra-Fast Parallel Image Downloader (Pure Python Version)
----------------------------------------------------------
✅ Works without aria2c (uses aiohttp)
✅ Fully async + parallel shards
✅ Live progress per shard
✅ Retry logic, resume-safe
✅ Manifest CSV at the end

Usage:
    python ultra_fast_downloader.py \
        --train dataset/train.csv \
        --test dataset/test.csv \
        --out_dir dataset/image_cache \
        --shards 8 \
        --connections 32
"""

import os
import argparse
import asyncio
import aiohttp
import async_timeout
import pandas as pd
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from concurrent.futures import ThreadPoolExecutor
import math
import time
import hashlib

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def collect_urls(train_csv, test_csv):
    train = pd.read_csv(train_csv, on_bad_lines="skip", engine="python")
    test  = pd.read_csv(test_csv,  on_bad_lines="skip", engine="python")
    urls = pd.concat([train["image_link"], test["image_link"]]).dropna().unique().tolist()
    print(f"🔹 Collected {len(urls):,} unique URLs.")
    return urls

def write_shards(urls, n_shards):
    """Split URLs into shard lists (in memory)."""
    step = math.ceil(len(urls) / n_shards)
    return [urls[i*step : (i+1)*step] for i in range(n_shards)]

def safe_filename(url):
    """Hash URL to a safe filename with .jpg fallback."""
    h = hashlib.md5(url.encode()).hexdigest()
    return f"{h}.jpg"

# ---------------------------------------------------------------------
# Async Download Logic
# ---------------------------------------------------------------------
async def fetch_image(session, sem, url, out_dir, retries=3, timeout=15):
    """Download a single image with retries."""
    filename = safe_filename(url)
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        return True

    async with sem:
        for attempt in range(retries):
            try:
                async with async_timeout.timeout(timeout):
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            with open(out_path, "wb") as f:
                                f.write(data)
                            return True
                        else:
                            await asyncio.sleep(1)
            except Exception:
                await asyncio.sleep(1)
        return False

async def download_shard(shard_id, urls, out_dir, connections=32):
    """Download one shard asynchronously."""
    os.makedirs(out_dir, exist_ok=True)
    sem = asyncio.Semaphore(connections)
    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit_per_host=connections, ssl=False)
    success = 0

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            fetch_image(session, sem, url, out_dir)
            for url in urls
        ]
        results = []
        for r in tqdm_asyncio.as_completed(tasks, desc=f"📦 Shard {shard_id}", total=len(tasks)):
            ok = await r
            if ok:
                success += 1
            results.append(ok)
    return success, len(urls)

# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
async def main_async(train, test, out_dir, shards, connections):
    t0 = time.time()
    urls = collect_urls(train, test)
    shard_lists = write_shards(urls, shards)
    os.makedirs(out_dir, exist_ok=True)

    print(f"🚀 Launching {shards} shards × {connections} connections each...\n")

    results = await asyncio.gather(
        *[download_shard(i+1, shard, out_dir, connections) for i, shard in enumerate(shard_lists)]
    )

    total_ok = sum(ok for ok, _ in results)
    total_urls = sum(total for _, total in results)
    elapsed = time.time() - t0

    files = [p for p in Path(out_dir).glob("*") if p.is_file()]
    manifest = pd.DataFrame({"local_path": [str(p) for p in files], "filename": [p.name for p in files]})
    manifest.to_csv(os.path.join(out_dir, "download_manifest.csv"), index=False)

    print(f"\n✅ Download complete: {total_ok}/{total_urls} images.")
    print(f"📄 Manifest saved: {out_dir}/download_manifest.csv")
    print(f"⏱️  Time: {elapsed/60:.2f} min  |  Speed: {total_ok/elapsed:.1f} imgs/sec")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra-Fast Parallel Image Downloader (Python)")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="dataset/image_cache")
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--connections", type=int, default=32)
    args = parser.parse_args()

    asyncio.run(main_async(args.train, args.test, args.out_dir, args.shards, args.connections))
