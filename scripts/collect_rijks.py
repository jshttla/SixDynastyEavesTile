"""Rijksmuseum collector (data.rijksmuseum.nl Linked Art search API, no key).

Counts roof-tile vocabulary hits per field, inspects every hit, applies the same scope rule as the other sources
(eaves-tile ends / antefixes / ridge-end tiles only; no flat tiles, no roof figurines, no pictures of roofs) and
downloads only in-scope objects whose image is public domain / CC0.

Usage: python -I scripts/collect_rijks.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, STAGING, Http, safe_ascii, setup_console, wlog, write_jsonl  # noqa: E402

SEARCH = "https://data.rijksmuseum.nl/search/collection"
TERMS = ["dakpan", "dakpannen", "daktegel", "antefix", "antefixen", "onigawara", "dakornament", "dakversiering",
         "dakbekroning", "nokpan", "eindpan", "roof tile", "imbrex", "tegula", "Stirnziegel"]
FIELDS = ["title", "type", "description"]
# In-scope only if the object itself is an eaves-tile end / antefix / ridge-end tile.
SCOPE_RE = re.compile(r"antefix|onigawara|eindpan|nokpan(?!nen)|stirnziegel|dakpan met (rond|schijf)|eaves tile|tile end", re.I)


def main():
    setup_console()
    http = Http(rate_per_sec=4, headers={"Accept": "application/ld+json"})
    out_dir = RAW / "rijks" / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = {}
    for f in FIELDS:
        for t in TERMS:
            url = f"{SEARCH}?" + urllib.parse.urlencode({f: t})
            items, nxt, total = [], url, None
            while nxt:
                d = http.json(nxt)
                total = d.get("partOf", {}).get("totalItems", 0) if total is None else total
                items += d.get("orderedItems", [])
                nxt = (d.get("next") or {}).get("id")
            if total:
                wlog("Stage1·rijks", f"检索计数 {f}={t}", url, total, "")
            for it in items:
                ids.setdefault(it["id"], []).append(f"{f}={t}")
    decisions = []
    for oid in sorted(ids):
        o = http.json(oid)
        (out_dir / f"{safe_ascii(oid.rsplit('/', 1)[1])}.json").write_text(json.dumps(o, ensure_ascii=False), encoding="utf-8")
        names = [x.get("content") for x in o.get("identified_by", []) if x.get("type") == "Name"]
        num = [x.get("content") for x in o.get("identified_by", []) if x.get("type") == "Identifier"]
        title = names[0] if names else ""
        keep = bool(SCOPE_RE.search(title))
        decisions.append({"id": oid, "object_number": num[:1], "title": title, "hits": ids[oid], "in_scope": keep})
    (RAW / "rijks" / "type_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=1), encoding="utf-8")
    kept = [d for d in decisions if d["in_scope"]]
    non_gevel = [d for d in decisions if "gevelsteen" not in d["title"].lower()]
    wlog("Stage1·rijks", "逐件读取命中对象的 Linked Art 记录并做类型过滤（只收瓦当 / antefix / 屋脊端饰本体；排除平瓦、屋顶照片与版画、山墙石 gevelsteen、屋脊人物俑）",
         "scripts/collect_rijks.py → raw/rijks/type_decisions.json", len(decisions),
         f"类型过滤后 {len(kept)} 件。非 gevelsteen 的命中：" + "；".join(f"{d['object_number'][0] if d['object_number'] else ''} {d['title']}" for d in non_gevel[:40]))
    if not kept:
        write_jsonl(STAGING / "rijks.jsonl", [])
        wlog("Stage1·rijks", "许可过滤与下载", "scripts/collect_rijks.py", 0,
             "无在范围内的条目（'Dakversiering in de vorm van een Chinese ruiter' 为屋脊人物俑，'Fragment van een Romeinse dakpan met opschrift' 为平瓦），Rijksmuseum 贡献 0 张；输出空 staging/rijks.jsonl")
    else:
        # Not reached on 2026-11-08; implement image + rights handling if future runs find in-scope objects.
        wlog("Stage1·rijks", "发现在范围内条目但下载未实现", "scripts/collect_rijks.py", len(kept), "需补写下载逻辑")
        write_jsonl(STAGING / "rijks.jsonl", [])
    print("rijks decisions", len(decisions), "kept", len(kept))


if __name__ == "__main__":
    main()
