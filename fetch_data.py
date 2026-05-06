"""
fetch_data.py – Download NOAA Storm Events detail CSVs into DataForProject/

Usage:
    python fetch_data.py                  # download all years (1950–present)
    python fetch_data.py --start 2000     # start from a specific year
    python fetch_data.py --start 2000 --end 2020
    python fetch_data.py --skip-existing  # skip years already downloaded (default: True)
"""

import argparse
import gzip
import os
import re
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles"
DATA_DIR = Path(__file__).parent / "DataForProject"
CURRENT_YEAR = datetime.now().year


def fetch_index() -> str:
    """Return the HTML directory listing from the NOAA FTP index page."""
    with urllib.request.urlopen(BASE_URL + "/", timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def find_file_for_year(index_html: str, year: int) -> str | None:
    """Return the filename for the given year, or None if not found."""
    pattern = rf"StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz"
    matches = re.findall(pattern, index_html)
    if not matches:
        return None
    # Pick the most recently released version (highest c-date)
    return sorted(matches)[-1]


def download_year(filename: str, dest_dir: Path, *, verbose: bool = True) -> Path:
    """Download a .csv.gz file, decompress it, and return the final .csv path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    gz_path = dest_dir / filename
    csv_path = dest_dir / filename.replace(".gz", "")

    url = f"{BASE_URL}/{filename}"
    if verbose:
        print(f"  Downloading {filename} ...", end=" ", flush=True)

    with urllib.request.urlopen(url, timeout=60) as resp, open(gz_path, "wb") as out:
        shutil.copyfileobj(resp, out)

    with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    gz_path.unlink()  # remove the compressed file

    if verbose:
        size_mb = csv_path.stat().st_size / 1_048_576
        print(f"done ({size_mb:.1f} MB)")

    return csv_path


def already_downloaded(year: int, dest_dir: Path) -> bool:
    """Return True if any CSV for this year exists in dest_dir."""
    return bool(list(dest_dir.glob(f"StormEvents_details-ftp_v1.0_d{year}_*.csv")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NOAA Storm Events CSVs")
    parser.add_argument("--start", type=int, default=1950, help="First year to download (default: 1950)")
    parser.add_argument("--end", type=int, default=CURRENT_YEAR, help=f"Last year to download (default: {CURRENT_YEAR})")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-download years that already exist locally")
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    print(f"Fetching NOAA directory index from {BASE_URL} ...")
    try:
        index = fetch_index()
    except Exception as exc:
        print(f"ERROR: Could not reach NOAA server – {exc}", file=sys.stderr)
        sys.exit(1)

    years = range(args.start, args.end + 1)
    missing, skipped, failed = [], [], []

    for year in years:
        if args.skip_existing and already_downloaded(year, DATA_DIR):
            skipped.append(year)
            continue

        filename = find_file_for_year(index, year)
        if filename is None:
            missing.append(year)
            continue

        try:
            download_year(filename, DATA_DIR)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append(year)

    print("\nSummary")
    print(f"  Downloaded : {len(years) - len(skipped) - len(missing) - len(failed)}")
    print(f"  Skipped (already exist): {len(skipped)}")
    print(f"  Not on server : {len(missing)}" + (f"  {missing}" if missing else ""))
    print(f"  Errors        : {len(failed)}" + (f"  {failed}" if failed else ""))


if __name__ == "__main__":
    main()
