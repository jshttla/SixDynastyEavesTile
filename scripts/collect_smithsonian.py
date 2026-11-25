"""Smithsonian Open Access collector.

No api.data.gov key was provided, so counting uses
  (a) the public S3 metadata dumps (scripts/si_dump_filter.py <unit>: per-shard record + keyword-match counts), and
  (b) a handful of api.si.edu searches with the public DEMO_KEY (10 requests/hour), saved under raw/smithsonian/search/.
Only media with online_media.media[].usage.access == "CC0" are eligible (方案 §3). Images come from IDS.

Usage: python -I scripts/collect_smithsonian.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RAW, ROOT, STAGING, Http, download_image, image_size, license_evidence, safe_ascii,  # noqa: E402
                    setup_console, stage_row, today, wlog, write_jsonl)

UNITS = ["nmaa", "fsg", "nmnhanthro", "chndm", "nmah"]
DUMP = RAW / "smithsonian" / "dump"
SEARCH = RAW / "smithsonian" / "search"
TYPE_RE = re.compile(r"roof[\s-]*tile|rooftile|tile[\s-]*end|end[\s-]*tile|eaves?[\s-]*tile|antefix|onigawara|wadang|"
                     r"瓦当|瓦當|軒丸瓦|軒平瓦|鬼瓦|막새|ridge[\s-]*end", re.I)
EXCL_RE = re.compile(r"cement|mold|mould|stamped with japanese text|trade catalog", re.I)
POLICY = "https://www.si.edu/openaccess"


def title_of(r: dict) -> str:
    return r.get("title") or r.get("content", {}).get("descriptiveNonRepeating", {}).get("title", {}).get("content", "")


def cc0_media(r: dict) -> list[dict]:
    media = r.get("content", {}).get("descriptiveNonRepeating", {}).get("online_media", {}).get("media", []) or []
    return [m for m in media if (m.get("usage") or {}).get("access") == "CC0" and m.get("type", "Images") in ("Images", "Image")]


def ft(r: dict, key: str) -> str | None:
    vals = [f"{x.get('label')}: {x.get('content')}" for x in r.get("content", {}).get("freetext", {}).get(key, []) or []]
    return " | ".join(vals) if vals else None


def main():
    setup_console()
    recs: dict[str, dict] = {}
    # ---- (a) dumps
    for u in UNITS:
        cf = DUMP / f"{u}_shard_counts.csv"
        mf = DUMP / f"{u}_matched.jsonl"
        if not cf.exists() or not mf.exists():
            wlog("Stage1·smithsonian", f"S3 转储单位 {u}：缺少过滤结果，跳过", f"python -I scripts/si_dump_filter.py {u}", "", "未计数")
            continue
        with open(cf, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        n_rec = sum(max(0, int(r["records"])) for r in rows)
        n_match = sum(int(r["matched"]) for r in rows)
        n_err = sum(1 for r in rows if r["error"])
        matched = [json.loads(ln) for ln in open(mf, encoding="utf-8") if ln.strip()]
        for r in matched:
            recs.setdefault(r["id"], r)
        wlog("Stage1·smithsonian", f"S3 整库元数据转储计数：单位 {u}，{len(rows)} 个分片逐行扫描关键词（roof tile / tile end / antefix / 瓦当 / 軒丸瓦 等）",
             f"https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/{u}/<00..ff>.txt（逐分片计数见 raw/smithsonian/dump/{u}_shard_counts.csv）",
             f"记录 {n_rec}，关键词命中 {n_match}", f"分片错误 {n_err}")
    # ---- (b) DEMO_KEY searches
    for f in sorted(SEARCH.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows = d.get("response", {}).get("rows", []) or []
        for r in rows:
            recs.setdefault(r["id"], r)
        wlog("Stage1·smithsonian", f"api.si.edu 检索（DEMO_KEY，每小时 10 次）：{f.name}",
             f"https://api.si.edu/openaccess/api/v1.0/search?q=<见文件名>&rows=1000&api_key=DEMO_KEY → raw/smithsonian/search/{f.name}",
             f"rowCount {d.get('response', {}).get('rowCount')}，取回 {len(rows)}", "")
    # ---- type filter
    typed = {k: r for k, r in recs.items() if TYPE_RE.search(title_of(r)) and not EXCL_RE.search(title_of(r))
             and r.get("unitCode", "").upper() not in ("SIL", "SILNMAHTL", "SLA_SRO", "SILAF")}
    wlog("Stage1·smithsonian", "合并转储与检索结果后做类型过滤（标题含 roof tile / antefix / tile end 等；排除图书馆书目、水泥瓦、瓦模、近代印字砖）",
         "scripts/collect_smithsonian.py", len(recs),
         f"类型过滤后 {len(typed)} 件：{Counter(r.get('unitCode') for r in typed.values()).most_common()}；"
         f"标题：{Counter(title_of(r) for r in typed.values()).most_common()}")
    # ---- licence filter
    lic = {k: r for k, r in typed.items() if cc0_media(r)}
    wlog("Stage1·smithsonian", "许可过滤：只收 online_media.media[].usage.access == 'CC0' 的媒体",
         "scripts/collect_smithsonian.py", len(typed),
         f"许可过滤后 {len(lic)} 件。无 CC0 媒体的类型命中条目（标题 × 件数）："
         f"{Counter(title_of(r) for k, r in typed.items() if k not in lic).most_common()}")
    # ---- download (if any)
    http = Http(rate_per_sec=2, timeout=180)
    out = []
    for k, r in sorted(lic.items()):
        dnr = r["content"]["descriptiveNonRepeating"]
        sid = dnr.get("record_ID") or k
        safe = safe_ascii(sid, 100)
        (RAW / "smithsonian" / "meta").mkdir(parents=True, exist_ok=True)
        (RAW / "smithsonian" / "meta" / f"{safe}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        for i, m in enumerate(cc0_media(r)):
            ids = m.get("idsId")
            url = f"https://ids.si.edu/ids/deliveryService?id={ids}"
            note = None
            try:
                p, sha, nb = download_image(http, url, RAW / "smithsonian" / "img" / f"{safe}_{i}")
            except Exception:  # noqa: BLE001
                url = url + "&max=4000"
                note = "原图下载失败，改用 IDS max=4000"
                try:
                    p, sha, nb = download_image(http, url, RAW / "smithsonian" / "img" / f"{safe}_{i}")
                except Exception as e:  # noqa: BLE001
                    print("fail", sid, e)
                    continue
            w, h = image_size(p)
            out.append(stage_row(
                source="smithsonian", source_id=sid, image_key=f"{sid}#{i}", image_index=i,
                source_url=dnr.get("record_link") or dnr.get("guid"), image_url=url, record_file=f"raw/smithsonian/meta/{safe}.json",
                title_raw=title_of(r), object_name_raw=ft(r, "objectType"), description_raw=ft(r, "notes"),
                culture_raw=ft(r, "culture"), date_raw=ft(r, "date"), region_raw=ft(r, "place"),
                medium_raw=ft(r, "physicalDescription"), credit_line_raw=ft(r, "creditLine"),
                holding_institution=dnr.get("data_source") or r.get("unitCode"), accession_number=ft(r, "identifier"),
                license="CC0-1.0", rights_raw=f"usage.access = CC0 ({(m.get('usage') or {}).get('text', '')})",
                license_evidence=license_evidence(POLICY, today(), "Smithsonian Open Access: usage.access == 'CC0'", "CC0"),
                raw_file=p.relative_to(ROOT).as_posix(), sha256=sha, bytes=nb, width=w, height=h,
                date_accessed=today(), notes=note))
    write_jsonl(STAGING / "smithsonian.jsonl", out)
    wlog("Stage1·smithsonian", "下载", "scripts/collect_smithsonian.py", len(lic), f"下载成功 {len(out)} 张；输出 staging/smithsonian.jsonl")
    print("smithsonian rows", len(out))


if __name__ == "__main__":
    main()
