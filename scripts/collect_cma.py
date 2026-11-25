"""Stage 1 collector: Cleveland Museum of Art Open Access API (source key `cma`).

    .venv/Scripts/python -I scripts/collect_cma.py [--count-only]

- Only share_license_status == "CC0".
- Image: the 'print' JPEG rendition (largest JPEG offered; the 'full' rendition is an uncompressed TIFF of
  ~70 MB per image). Bytes are saved exactly as delivered (no re-encoding). alternate_images = extra views.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EVIDENCE, RAW, ROOT, STAGING, Http, download_image, image_size, license_evidence,  # noqa: E402
                    read_jsonl, setup_console, stage_row, today, wlog, write_jsonl)

SRC = "cma"
API = "https://openaccess-api.clevelandart.org/api/artworks/"
QUERIES = ["antefix", "roof tile", "tile end", "eaves tile", "eave", "roof-tile end", "end tile", "roof ornament",
           "ridge tile", "onigawara", "wadang", "roof end", "gorgoneion"]
POLICY_URL = "https://www.clevelandart.org/open-access"
POLICY_QUOTE_FALLBACK = "share_license_status: CC0 (Open Access API field); CMA Open Access images are released under Creative Commons Zero (CC0)"

INCLUDE = re.compile(r"antefix|roof[- ]?tile|tile[- ]end|end[- ]tile|eaves?[- ]?tile|ridge[- ]?tile|ridge[- ]end|"
                     r"onigawara|roof ornament|roof[- ]end|wadang|chiwei|roof[- ]finial|\beaves?\b", re.I)
EXCLUDE = re.compile(r"floor tile|wall tile|tile panel|kiln|stove tile|revetment|inkstone|\bmodel\b|tegula|imbrex|"
                     r"pantile|flat tile|\bbrick\b", re.I)
KEEP = re.compile(r"antefix|tile[- ]end|end[- ]tile|eaves?[- ]?tile|onigawara", re.I)
REPRO = re.compile(r"\b(copy|replica|reproduction|facsimile|forgery|plaster cast|electrotype)\b", re.I)


def save_policy(h: Http) -> tuple[str, str]:
    dest = EVIDENCE / f"{SRC}_policy_2026-11-08.html"
    txt = None
    if dest.exists():
        txt = dest.read_text(encoding="utf-8", errors="replace")
    else:
        try:
            r = h.get(POLICY_URL, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                txt = r.text
                EVIDENCE.mkdir(exist_ok=True)
                dest.write_text(txt, encoding="utf-8")
        except Exception:
            pass
    if txt:
        import html as _h
        txt2 = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", txt)
        flat = re.sub(r"\s+", " ", _h.unescape(re.sub(r"<[^>]+>", " ", txt2)))
        flat = flat.replace("(opens in a new tab)", "").replace(chr(160), " ")
        flat = re.sub(r"\s+", " ", flat)
        m = re.search(r"[^.]{0,220}(CC0|Creative Commons Zero)[^.]{0,220}\.", flat)
        if m:
            return "2026-11-08", m.group(0).strip()
    return "2026-11-08 (policy page not retrievable; API field used)", POLICY_QUOTE_FALLBACK


def type_ok(o: dict) -> tuple[bool, str]:
    text = " | ".join([o.get("title") or "", o.get("type") or ""])
    if not INCLUDE.search(text):
        return False, "no_type_keyword"
    if EXCLUDE.search(text) and not KEEP.search(text):
        return False, "excluded_type"
    lo = o.get("creation_date_earliest")
    if isinstance(lo, int) and lo >= 1900:
        return False, "modern"
    return True, ""


def jpg_renditions(o: dict) -> list[str]:
    out = []
    im = o.get("images") or {}
    for rend in ("print", "web"):
        if (im.get(rend) or {}).get("url"):
            out.append(im[rend]["url"])
            break
    for alt in o.get("alternate_images") or []:
        for rend in ("print", "web"):
            if (alt.get(rend) or {}).get("url"):
                out.append(alt[rend]["url"])
                break
    return out


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    h = Http(rate_per_sec=3)
    meta_dir, img_dir = RAW / SRC / "meta", RAW / SRC / "img"
    for d in (meta_dir, img_dir):
        d.mkdir(parents=True, exist_ok=True)

    recs: dict[int, dict] = {}
    per = {}
    for q in QUERIES:
        skip, got = 0, 0
        total = None
        while True:
            r = h.json(API, params={"q": q, "has_image": 1, "limit": 100, "skip": skip})
            total = r["info"]["total"]
            for o in r["data"]:
                recs[o["id"]] = o
            got += len(r["data"])
            skip += 100
            if not r["data"] or skip >= total:
                break
        per[q] = total
    wlog("Stage1", "CMA 计数：每个检索词 has_image=1（全馆，CMA API 无部门参数时按全馆检索）",
         f"{API}?q=<q>&has_image=1&limit=100&skip=<n>", json.dumps(per, ensure_ascii=False),
         f"检索词并集 {len(recs)} 件")
    if a.count_only:
        return 0

    typed, reasons = [], {}
    for o in recs.values():
        ok, why = type_ok(o)
        if ok:
            typed.append(o)
        else:
            reasons[why] = reasons.get(why, 0) + 1
    wlog("Stage1", "CMA 类型过滤（title/type 含 antefix/roof tile/tile end/eave/ridge tile 等；排除墙砖/地砖/砚/模型等；"
         "creation_date_earliest ≥ 1900 排除）", "", len(recs),
         f"过滤后 {len(typed)}：{[o['title'] for o in typed]}；排除原因 {json.dumps(reasons)}")
    lic = [o for o in typed if o.get("share_license_status") == "CC0" and jpg_renditions(o)]
    wlog("Stage1", "CMA 许可过滤：只收 share_license_status == \"CC0\" 且有图", "", len(typed),
         f"通过 {len(lic)}；不通过 {len(typed) - len(lic)}（{[(o['title'], o.get('share_license_status')) for o in typed if o not in lic]}）")

    ev_date, quote = save_policy(h)
    prev = {r["image_key"]: r for r in read_jsonl(STAGING / f"{SRC}.jsonl")}
    rows, fail, n_imgs = [], 0, 0
    for o in lic:
        sid = str(o["id"])
        (meta_dir / f"{sid}.json").write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
        urls = jpg_renditions(o)
        n_imgs += len(urls)
        title_txt = o.get("title") or ""
        for n, u in enumerate(urls):
            key = f"{sid}#{n}"
            if key in prev and (ROOT / prev[key]["raw_file"]).exists() and prev[key]["image_url"] == u:
                p, sha, nb = ROOT / prev[key]["raw_file"], prev[key]["sha256"], prev[key]["bytes"]
            else:
                try:
                    p, sha, nb = download_image(h, u, img_dir / f"{sid}_{n}")
                except Exception as e:
                    fail += 1
                    print("  fail", sid, n, e)
                    continue
            w, hh = image_size(p)
            culture = "; ".join(o.get("culture") or []) or None
            rows.append(stage_row(
                source=SRC, source_id=sid, image_key=key, image_index=n,
                source_url=o.get("url") or f"https://clevelandart.org/art/{o.get('accession_number')}",
                image_url=u, iiif_manifest=None,
                record_file=f"raw/{SRC}/meta/{sid}.json",
                title_raw=o.get("title") or None, object_name_raw=o.get("type") or None,
                description_raw=o.get("description") or None, classification_raw=o.get("type") or None,
                culture_raw=culture, period_raw=None, date_raw=o.get("creation_date") or None,
                date_from=o.get("creation_date_earliest"), date_to=o.get("creation_date_latest"),
                region_raw=(f"find_spot: {o['find_spot']}" if o.get("find_spot") else None),
                medium_raw=o.get("technique") or None, dimensions_raw=o.get("measurements") or None,
                credit_line_raw=o.get("creditline") or None, subjects_raw=None,
                holding_institution="The Cleveland Museum of Art", holding_institution_qid="Q657415",
                accession_number=o.get("accession_number") or None,
                license="CC0-1.0", rights_raw=f"share_license_status: {o.get('share_license_status')}",
                license_evidence=license_evidence(POLICY_URL, ev_date, quote,
                                                  f"share_license_status: CC0 ({API}{sid})"),
                rights_holder=None,
                attribution_text=f"The Cleveland Museum of Art, {o.get('creditline') or ''} {o.get('accession_number') or ''}".strip(),
                modality_hint="rubbing" if re.search(r"rubbing", title_txt, re.I) else "photo",
                is_reproduction=bool(REPRO.search(title_txt)),
                raw_file=str(p.relative_to(ROOT)).replace("\\", "/"), sha256=sha, bytes=nb, width=w, height=hh,
                date_accessed=today(), mirror_of=None,
                notes=f"CMA department: {o.get('department')}; tombstone: {o.get('tombstone')}; image rendition: "
                      f"{'print' if '_print' in u else 'web'} JPEG (full TIFF not taken)",
            ))
    write_jsonl(STAGING / f"{SRC}.jsonl", rows)
    wlog("Stage1", "CMA 下载（print JPEG 原样保存）并写 staging/cma.jsonl", "GET openaccess-cdn.clevelandart.org/<acc>/<acc>_print.jpg",
         n_imgs, f"成功 {len(rows)} 张 / {len({r['source_id'] for r in rows})} 件；失败 {fail}")
    print(f"cma: {len(rows)} images, fail {fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
