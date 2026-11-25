"""Stream Smithsonian Open Access S3 metadata shards for one unit, keep only lines that mention
roof-tile vocabulary (pre-filter for counting), and write per-shard counts.

Usage: python -I scripts/si_dump_filter.py <unit> [--save-full]
Outputs:
  raw/smithsonian/dump/<unit>_matched.jsonl      lines whose text matches KEYWORDS
  raw/smithsonian/dump/<unit>_shard_counts.csv   shard,url,records,matched
"""
from __future__ import annotations

import csv
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, Http, setup_console  # noqa: E402

BASE = "https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan"
KEYWORDS = re.compile(r"roof[\s-]*tile|tile[\s-]*end|end[\s-]*tile|eaves?[\s-]*tile|antefix|antefixes|onigawara|"
                      r"wadang|瓦当|瓦當|軒丸瓦|軒平瓦|鬼瓦|막새|ridge[\s-]*tile|roof[\s-]*end|ridge[\s-]*end", re.I)


def main():
    setup_console()
    unit = sys.argv[1]
    save_full = "--save-full" in sys.argv
    out_dir = RAW / "smithsonian" / "dump"
    out_dir.mkdir(parents=True, exist_ok=True)
    http = Http(rate_per_sec=20, timeout=300)
    r = http.get(f"{BASE}/{unit}/index.txt")
    idx = r.text.split() if r.status_code == 200 else []
    if not idx or not idx[0].startswith("https://"):
        # some units (e.g. nmaa) have no index.txt; shards are 00..ff
        idx = [f"{BASE}/{unit}/{i:02x}.txt" for i in range(256)]
    results = {}

    def work(url):
        try:
            txt = http.get(url).content.decode("utf-8")
        except Exception as e:  # noqa: BLE001
            return url, -1, [], str(e)
        if save_full:
            (out_dir / unit).mkdir(exist_ok=True)
            (out_dir / unit / url.rsplit("/", 1)[1]).write_text(txt, encoding="utf-8")
        lines = [ln for ln in txt.split("\n") if ln.strip()]
        hits = [ln for ln in lines if KEYWORDS.search(ln)]
        return url, len(lines), hits, ""

    with ThreadPoolExecutor(8) as ex:
        for url, n, hits, err in ex.map(work, idx):
            results[url] = (n, hits, err)
    with open(out_dir / f"{unit}_matched.jsonl", "w", encoding="utf-8") as f:
        for url in idx:
            for ln in results[url][1]:
                f.write(ln + "\n")
    with open(out_dir / f"{unit}_shard_counts.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shard", "url", "records", "matched", "error"])
        for url in idx:
            n, hits, err = results[url]
            w.writerow([url.rsplit("/", 1)[1], url, n, len(hits), err])
    tot = sum(max(0, results[u][0]) for u in idx)
    m = sum(len(results[u][1]) for u in idx)
    errs = sum(1 for u in idx if results[u][2])
    print(f"{unit}: shards={len(idx)} records={tot} matched={m} errors={errs}")


if __name__ == "__main__":
    main()
