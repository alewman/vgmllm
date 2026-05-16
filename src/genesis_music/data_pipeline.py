"""Download and prepare Genesis VGM training data from vgmrips.net.

Usage:
    # Step 1: Scrape pack listing to discover download URLs
    python -m genesis_music.data_pipeline scrape

    # Step 1b: OR seed URL list from known page structure (no network needed)
    python -m genesis_music.data_pipeline seed-urls

    # Step 2: Download all packs (resumable, with retries)
    python -m genesis_music.data_pipeline download
    python -m genesis_music.data_pipeline download --max-packs 5   # test with 5

    # Step 3: Extract VGM/VGZ files from zips
    python -m genesis_music.data_pipeline extract

    # Step 4: Validate and build dataset stats
    python -m genesis_music.data_pipeline validate

    # Or run all steps (uses scrape, not seed-urls):
    python -m genesis_music.data_pipeline all

Data is stored under data/ in the project root:
    data/
        pack_urls.json      <- discovered download URLs
        zips/               <- downloaded zip files
        vgm/                <- extracted VGM/VGZ files (flat, deduplicated)
        dataset_stats.json  <- validation results and statistics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

from .vgm_parser import load_vgm, summarize_vgm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ZIPS_DIR = DATA_DIR / "zips"
VGM_DIR = DATA_DIR / "vgm"
URLS_FILE = DATA_DIR / "pack_urls.json"
STATS_FILE = DATA_DIR / "dataset_stats.json"

BASE_URL = "https://vgmrips.net"

# Pages to scrape (Mega Drive + Genesis system pages share the same files)
SYSTEM_PAGES = [
    "/packs/system/sega/mega-drive",
    "/packs/system/sega/genesis",
]

# Additional system/chip pages for stretch-goal datasets
EXTRA_SYSTEM_PAGES = {
    "sms-fm": [
        "/packs/chip/ym2413",           # YM2413 (OPLL) — Japanese SMS FM + MSX
    ],
    "arcade": [
        "/packs/system/other/arcade-machine",  # 262 packs — mixed chips
        "/packs/system/sega/arcade",           # 11 packs — Sega arcade
        "/packs/system/sega/system-16b",       # 25 packs — System 16B (YM2151+)
        "/packs/system/sega/system-16a",       # 9 packs — System 16A
        "/packs/system/sega/system-18",        # 10 packs — System 18 (YM3438)
        "/packs/system/sega/system-32",        # 16 packs — System 32
        "/packs/system/sega/system-1",         # 20 packs — System 1
        "/packs/system/sega/system-2",         # 8 packs — System 2
    ],
}

# Be respectful: delay between requests
SCRAPE_DELAY = 1.5   # seconds between page fetches
DOWNLOAD_DELAY = 2.0  # seconds between file downloads

HEADERS = {
    "User-Agent": "vgmllm/0.1 (research dataset collection)",
}


def _make_session() -> requests.Session:
    """Create a requests session with automatic retry and backoff."""
    session = requests.Session()
    session.headers.update(HEADERS)
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,  # 2, 4, 8, 16, 32 second waits
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _make_session()


# ---------------------------------------------------------------------------
# Step 1: Scrape pack listing pages to find download URLs
# ---------------------------------------------------------------------------

def _extract_download_urls(html: str) -> list[str]:
    """Extract .zip download URLs from a vgmrips pack listing page."""
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".zip") and "/files/" in href:
            # Normalize to absolute URL
            full_url = urljoin(BASE_URL, href)
            urls.append(full_url)
    return urls


def _get_max_page(html: str) -> int:
    """Parse pagination to find the last page number."""
    soup = BeautifulSoup(html, "html.parser")
    max_page = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        match = re.search(r"[?&]p=(\d+)", href)
        if match:
            page = int(match.group(1))
            max_page = max(max_page, page)
    return max_page


def scrape_pack_urls() -> list[str]:
    """Scrape all Genesis/Mega Drive pack download URLs from vgmrips."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_urls: set[str] = set()

    for system_path in SYSTEM_PAGES:
        url = f"{BASE_URL}{system_path}"
        log.info(f"Fetching first page: {url}")

        try:
            resp = SESSION.get(url, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"Failed to fetch {url}: {e}")
            log.error("If vgmrips.net is unreachable, try 'seed-urls' instead.")
            continue

        page_urls = _extract_download_urls(resp.text)
        all_urls.update(page_urls)
        max_page = _get_max_page(resp.text)

        log.info(f"  Found {len(page_urls)} URLs, {max_page + 1} total pages")

        for page_num in range(1, max_page + 1):
            time.sleep(SCRAPE_DELAY)
            page_url = f"{url}?p={page_num}"
            log.info(f"  Fetching page {page_num + 1}/{max_page + 1}: {page_url}")

            try:
                resp = SESSION.get(page_url, timeout=60)
                resp.raise_for_status()
                page_urls = _extract_download_urls(resp.text)
                all_urls.update(page_urls)
                log.info(f"    +{len(page_urls)} URLs (total unique: {len(all_urls)})")
            except requests.RequestException as e:
                log.warning(f"    Failed: {e}")

    url_list = sorted(all_urls)
    URLS_FILE.write_text(json.dumps(url_list, indent=2))
    log.info(f"Saved {len(url_list)} unique download URLs to {URLS_FILE}")
    return url_list


def seed_pack_urls(extra_pages: list[str] | None = None) -> list[str]:
    """Generate the pack URL list by scraping page-by-page.

    This is the same as scrape_pack_urls but with more aggressive
    resumption: if a partial URL file exists, it extends it.
    If scraping fails entirely, it provides instructions for
    the manual alternative.

    Args:
        extra_pages: Additional system/chip page paths to scrape
            beyond the default SYSTEM_PAGES.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load any existing URLs to extend
    existing: set[str] = set()
    if URLS_FILE.exists():
        existing = set(json.loads(URLS_FILE.read_text()))
        log.info(f"Loaded {len(existing)} existing URLs from {URLS_FILE}")

    all_urls = set(existing)
    success = False

    pages_to_scrape = list(SYSTEM_PAGES)
    if extra_pages:
        pages_to_scrape.extend(extra_pages)

    for system_path in pages_to_scrape:
        url = f"{BASE_URL}{system_path}"
        log.info(f"Fetching: {url}")

        try:
            resp = SESSION.get(url, timeout=60)
            resp.raise_for_status()
            success = True
        except requests.RequestException as e:
            log.warning(f"  Cannot reach {url}: {e}")
            continue

        page_urls = _extract_download_urls(resp.text)
        all_urls.update(page_urls)
        max_page = _get_max_page(resp.text)
        log.info(f"  Page 1: {len(page_urls)} URLs, {max_page + 1} total pages")

        for page_num in range(1, max_page + 1):
            time.sleep(SCRAPE_DELAY)
            page_url = f"{url}?p={page_num}"

            try:
                resp = SESSION.get(page_url, timeout=60)
                resp.raise_for_status()
                page_urls = _extract_download_urls(resp.text)
                all_urls.update(page_urls)
                if (page_num + 1) % 5 == 0:
                    log.info(f"  Page {page_num + 1}/{max_page + 1}: "
                             f"total unique={len(all_urls)}")
            except requests.RequestException as e:
                log.warning(f"  Page {page_num + 1} failed: {e}")

    if not success and not existing:
        log.error(
            "Could not reach vgmrips.net and no existing URLs found.\n"
            "Options:\n"
            "  1. Check your internet connection and try again\n"
            "  2. Manually download packs from https://vgmrips.net/packs/system/sega/mega-drive\n"
            "     and place the .zip files in: {}\n".format(ZIPS_DIR)
        )
        return []

    url_list = sorted(all_urls)
    URLS_FILE.write_text(json.dumps(url_list, indent=2))
    new_count = len(all_urls) - len(existing)
    log.info(f"Saved {len(url_list)} URLs to {URLS_FILE} ({new_count} new)")
    return url_list


# ---------------------------------------------------------------------------
# Step 2: Download zip files
# ---------------------------------------------------------------------------

def download_packs(max_packs: int | None = None) -> None:
    """Download VGM pack zips from the URL list. Resumable — skips existing files."""
    if not URLS_FILE.exists():
        log.error(f"URL list not found at {URLS_FILE}. Run 'scrape' or 'seed-urls' first.")
        return

    urls = json.loads(URLS_FILE.read_text())
    if max_packs is not None:
        urls = urls[:max_packs]

    ZIPS_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0
    failed_urls: list[str] = []

    for i, url in enumerate(urls):
        filename = unquote(url.split("/")[-1])
        dest = ZIPS_DIR / filename

        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue

        if (i + 1) % 20 == 1 or max_packs is not None:
            log.info(f"[{i + 1}/{len(urls)}] Downloading: {filename}")

        try:
            time.sleep(DOWNLOAD_DELAY)
            resp = SESSION.get(url, timeout=120, stream=True)
            resp.raise_for_status()

            # Write to temp file first, then rename (atomic-ish)
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    f.write(chunk)
            tmp.rename(dest)
            downloaded += 1

        except requests.RequestException as e:
            log.warning(f"  Failed [{i + 1}]: {filename} — {e}")
            failed += 1
            failed_urls.append(url)
            # Clean up partial temp
            tmp = dest.with_suffix(".tmp")
            if tmp.exists():
                tmp.unlink()
        except Exception as e:
            log.error(f"  Unexpected error [{i + 1}]: {filename} — {e}")
            failed += 1
            failed_urls.append(url)

        # Progress every 50 downloads
        if downloaded > 0 and downloaded % 50 == 0:
            log.info(f"  Progress: {downloaded} downloaded, {skipped} skipped, {failed} failed")

    log.info(
        f"Download complete: {downloaded} new, {skipped} already existed, {failed} failed"
    )
    if failed_urls:
        failed_file = DATA_DIR / "failed_downloads.json"
        failed_file.write_text(json.dumps(failed_urls, indent=2))
        log.info(f"Failed URLs saved to {failed_file} — retry with 'download' command")


# ---------------------------------------------------------------------------
# Step 3: Extract VGM/VGZ files from zip archives
# ---------------------------------------------------------------------------

def extract_vgm_files() -> None:
    """Extract all VGM/VGZ files from downloaded zips into a flat directory."""
    if not ZIPS_DIR.exists():
        log.error(f"Zips directory not found at {ZIPS_DIR}. Run 'download' first.")
        return

    VGM_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(ZIPS_DIR.glob("*.zip"))
    log.info(f"Processing {len(zip_files)} zip archives...")

    total_extracted = 0
    total_skipped = 0
    seen_hashes: set[str] = set()

    for zip_path in zip_files:
        pack_name = zip_path.stem
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for info in zf.infolist():
                    name_lower = info.filename.lower()
                    if not (name_lower.endswith(".vgm") or name_lower.endswith(".vgz")):
                        continue

                    # Read the file content
                    data = zf.read(info.filename)

                    # Deduplicate by content hash
                    content_hash = hashlib.sha256(data).hexdigest()[:16]
                    if content_hash in seen_hashes:
                        total_skipped += 1
                        continue
                    seen_hashes.add(content_hash)

                    # Build output filename: pack_name__original_name
                    orig_name = Path(info.filename).name
                    safe_pack = re.sub(r"[^\w\-.]", "_", pack_name)
                    safe_name = re.sub(r"[^\w\-.]", "_", orig_name)
                    out_name = f"{safe_pack}__{safe_name}"
                    out_path = VGM_DIR / out_name

                    out_path.write_bytes(data)
                    total_extracted += 1

        except zipfile.BadZipFile:
            log.warning(f"  Bad zip file: {zip_path.name}")
        except Exception as e:
            log.warning(f"  Error processing {zip_path.name}: {e}")

    log.info(
        f"Extraction complete: {total_extracted} VGM files extracted, "
        f"{total_skipped} duplicates skipped"
    )


# ---------------------------------------------------------------------------
# Step 4: Validate extracted files with our parser and build stats
# ---------------------------------------------------------------------------

def validate_dataset() -> dict:
    """Parse all extracted VGM files and collect statistics."""
    if not VGM_DIR.exists():
        log.error(f"VGM directory not found at {VGM_DIR}. Run 'extract' first.")
        return {}

    vgm_files = sorted(
        list(VGM_DIR.glob("*.vgm")) + list(VGM_DIR.glob("*.vgz"))
    )
    log.info(f"Validating {len(vgm_files)} VGM files...")

    stats = {
        "total_files": len(vgm_files),
        "valid_files": 0,
        "invalid_files": 0,
        "has_ym2612": 0,
        "has_sn76489": 0,
        "total_duration_seconds": 0.0,
        "total_events": 0,
        "total_ym2612_writes": 0,
        "files_by_duration_bucket": {
            "0-30s": 0,
            "30-60s": 0,
            "60-120s": 0,
            "120-180s": 0,
            "180-300s": 0,
            "300s+": 0,
        },
        "errors": [],
    }

    for i, vgm_path in enumerate(vgm_files):
        if (i + 1) % 500 == 0:
            log.info(f"  Validated {i + 1}/{len(vgm_files)}...")

        try:
            vgm = load_vgm(vgm_path)
            summary = summarize_vgm(vgm)

            stats["valid_files"] += 1
            if summary["has_ym2612"]:
                stats["has_ym2612"] += 1
            if summary["has_sn76489"]:
                stats["has_sn76489"] += 1

            dur = summary["duration_seconds"]
            stats["total_duration_seconds"] += dur
            stats["total_events"] += summary["total_events"]
            stats["total_ym2612_writes"] += (
                summary["ym2612_port0_writes"] + summary["ym2612_port1_writes"]
            )

            if dur < 30:
                stats["files_by_duration_bucket"]["0-30s"] += 1
            elif dur < 60:
                stats["files_by_duration_bucket"]["30-60s"] += 1
            elif dur < 120:
                stats["files_by_duration_bucket"]["60-120s"] += 1
            elif dur < 180:
                stats["files_by_duration_bucket"]["120-180s"] += 1
            elif dur < 300:
                stats["files_by_duration_bucket"]["180-300s"] += 1
            else:
                stats["files_by_duration_bucket"]["300s+"] += 1

        except Exception as e:
            stats["invalid_files"] += 1
            stats["errors"].append({
                "file": vgm_path.name,
                "error": str(e)[:200],
            })

    # Truncate errors list for readability
    if len(stats["errors"]) > 50:
        stats["errors"] = stats["errors"][:50]
        stats["errors_truncated"] = True

    stats["total_duration_hours"] = round(stats["total_duration_seconds"] / 3600, 2)
    stats["avg_duration_seconds"] = round(
        stats["total_duration_seconds"] / max(stats["valid_files"], 1), 1
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, indent=2))
    log.info(f"Validation complete. Stats saved to {STATS_FILE}")

    # Print summary
    log.info("=" * 60)
    log.info(f"  Total files:        {stats['total_files']}")
    log.info(f"  Valid:              {stats['valid_files']}")
    log.info(f"  Invalid:            {stats['invalid_files']}")
    log.info(f"  With YM2612:        {stats['has_ym2612']}")
    log.info(f"  With SN76489:       {stats['has_sn76489']}")
    log.info(f"  Total duration:     {stats['total_duration_hours']} hours")
    log.info(f"  Avg duration:       {stats['avg_duration_seconds']} seconds")
    log.info(f"  Total events:       {stats['total_events']:,}")
    log.info(f"  YM2612 writes:      {stats['total_ym2612_writes']:,}")
    log.info(f"  Duration buckets:   {stats['files_by_duration_bucket']}")
    log.info("=" * 60)

    return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare Genesis VGM training data",
    )
    parser.add_argument(
        "command",
        choices=["scrape", "seed-urls", "download", "extract", "validate", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--max-packs",
        type=int,
        default=None,
        help="Maximum number of packs to download (for testing)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--systems",
        nargs="*",
        choices=list(EXTRA_SYSTEM_PAGES.keys()),
        default=[],
        help="Additional system groups to scrape: sms-fm, arcade",
    )

    args = parser.parse_args()

    if args.data_dir:
        global DATA_DIR, ZIPS_DIR, VGM_DIR, URLS_FILE, STATS_FILE
        DATA_DIR = Path(args.data_dir)
        ZIPS_DIR = DATA_DIR / "zips"
        VGM_DIR = DATA_DIR / "vgm"
        URLS_FILE = DATA_DIR / "pack_urls.json"
        STATS_FILE = DATA_DIR / "dataset_stats.json"

    if args.command == "seed-urls":
        extra = []
        for sys_name in args.systems:
            extra.extend(EXTRA_SYSTEM_PAGES[sys_name])
        seed_pack_urls(extra_pages=extra if extra else None)
    elif args.command in ("scrape", "all"):
        scrape_pack_urls()

    if args.command in ("download", "all"):
        download_packs(max_packs=args.max_packs)

    if args.command in ("extract", "all"):
        extract_vgm_files()

    if args.command in ("validate", "all"):
        validate_dataset()


if __name__ == "__main__":
    main()
