"""Download the raw UFCStats scrape at a pinned commit and verify checksums.

The data comes from github.com/Greco1899/scrape_ufc_stats (GPL-3.0), a
third-party scrape of ufcstats.com. Pinning the commit means every result in
this repo can be reproduced against exactly the same rows, even though the
upstream repo updates daily.

This repository does not include or redistribute any UFC data. Running this
script downloads a third-party dataset to your machine. UFC's Terms of Use
restrict scraping and building databases from UFC website content, and it is
unclear whether those terms cover ufcstats.com. Read them and decide for
yourself before running this. You are responsible for your own use.

Usage:
    python scripts/fetch_data.py --i-have-read-the-terms   # download + verify
    python scripts/fetch_data.py --pin      # (maintainer) record checksums for the commit below
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

COMMIT = "1ccacc5cd4f642bd2deb8b278405a0205791bfa3"  # 2026-10-04, last event 2026-10-03
FILES = ["ufc_event_details.csv", "ufc_fight_results.csv", "ufc_fight_stats.csv"]
BASE = f"https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/{COMMIT}/"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if "--i-have-read-the-terms" not in sys.argv and "--pin" not in sys.argv:
        sys.exit(__doc__)
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dest = RAW / f
        if not dest.exists():
            print(f"downloading {f}")
            urllib.request.urlretrieve(BASE + f, dest)

    if "--pin" in sys.argv:
        manifest = {
            "source": "https://github.com/Greco1899/scrape_ufc_stats",
            "license": "GPL-3.0 (upstream data scraped from ufcstats.com)",
            "commit": COMMIT,
            "files": {f: sha256(RAW / f) for f in FILES},
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"wrote {MANIFEST}")
        return

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["commit"] == COMMIT, "MANIFEST commit does not match COMMIT"
    for f, expected in manifest["files"].items():
        got = sha256(RAW / f)
        if got != expected:
            sys.exit(f"CHECKSUM MISMATCH for {f}: delete data/raw and re-run")
    print("raw data verified against pinned commit", COMMIT[:10])


if __name__ == "__main__":
    main()
