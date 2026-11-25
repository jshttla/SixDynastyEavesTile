"""Stage 1 collector: Art Institute of Chicago API (source key `aic`).

    .venv/Scripts/python -I scripts/collect_aic.py [--count-only]

- Search: POST /api/v1/artworks/search with an Elasticsearch phrase query (the plain `q=` search returns the whole
  collection as `total`, so phrase queries are used to get honest counts).
- Only is_public_domain == true. Images via IIIF: full/max and full/full are refused (HTTP 403); the largest size
  AIC serves for public-domain works is full/1686,/0/default.jpg, falling back to full/843,.
- The IIIF host is behind a Cloudflare managed challenge for some requests. We do NOT try to bypass it: polite
  retries only; images that stay blocked are counted as failures.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EVIDENCE, RAW, ROOT, STAGING, Http, image_size, license_evidence, read_jsonl,  # noqa: E402
                    setup_console, sha256_bytes, sniff_ext, stage_row, today, wlog, write_jsonl)

SRC = "aic"
SEARCH = "https://api.artic.edu/api/v1/artworks/search"
ART = "https://api.artic.edu/api/v1/artworks/{}"
IIIF = "https://www.artic.edu/iiif/2/{}/full/{},/0/default.jpg"
PHRASES = ["antefix", "roof tile", "tile end", "eaves tile", "eave tile", "end tile", "roof-end", "roof end",
           "roof ornament", "ridge tile", "onigawara", "wadang", "roof edge ornament", "gorgoneion"]
FIELDS = ["id", "title", "artwork_type_title", "classification_titles", "department_title", "date_display",
          "date_start", "date_end", "place_of_origin", "is_public_domain", "image_id", "alt_image_ids",
          "medium_display", "dimensions", "credit_line", "main_reference_number", "culture", "style_title",
          "description", "short_description", "inscriptions", "provenance_text", "term_titles", "subject_titles"]
POLICY_URL = "https://www.artic.edu/open-access/open-access-images"
QUOTE_FALLBACK = ("AIC Open Access: images of public-domain artworks are available under Creative Commons Zero (CC0) "
                  "(API field is_public_domain)")

INCLUDE = re.compile(r"antefix|roof[- ]?tile|tile[- ]end|end[- ]tile|eaves?[- ]?tile|ridge[- ]?tile|ridge[- ]end|"
                     r"onigawara|roof ornament|roof[- ]end|roof edge ornament|wadang|chiwei", re.I)
# comparison group: architectural ceramics with a frontal face / mask
FACE_ARCH = re.compile(r"\b(mask|face|head)\b", re.I)
EXCLUDE = re.compile(r"floor tile|wall tile|tile panel|kiln|inkstone|\bmodel\b|tegula|imbrex|pantile|flat tile|"
                     r"study for|drawing|print", re.I)
CERAMIC = re.compile(r"earthenware|terracotta|terra cotta|stoneware|ceramic|clay|glaze", re.I)


def search_phrase(h: Http, phrase: str) -> tuple[int, list[dict]]:
    out, page, total = [], 1, 0
    while True:
        body = {"query": {"multi_match": {"query": phrase, "type": "phrase",
                                          "fields": ["title", "artwork_type_title", "classification_titles",
                                                     "term_titles"]}},
                "limit": 100, "page": page, "fields": FIELDS}
        r = h.s.post(SEARCH, json=body, timeout=60)
        r.raise_for_status()
        d = r.json()
        total = d["pagination"]["total"]
        out += d["data"]
        if page >= d["pagination"]["total_pages"] or not d["data"]:
            break
        page += 1
        time.sleep(0.5)
    return total, out


def search_arch_fragments(h: Http) -> tuple[int, list[dict]]:
    out, page, total = [], 1, 0
    while True:
        body = {"query": {"bool": {"must": [{"match_phrase": {"artwork_type_title": "Architectural fragment"}}]}},
                "limit": 100, "page": page, "fields": FIELDS}
        r = h.s.post(SEARCH, json=body, timeout=60)
        r.raise_for_status()
        d = r.json()
        total = d["pagination"]["total"]
        out += d["data"]
        if page >= d["pagination"]["total_pages"] or not d["data"]:
            break
        page += 1
        time.sleep(0.5)
    return total, out


def type_ok(o: dict) -> tuple[bool, str]:
    t = o.get("title") or ""
    typ = o.get("artwork_type_title") or ""
    if typ in ("Print", "Drawing and Watercolor", "Photograph", "Painting", "Architectural Drawing", "Book"):
        return False, "not_object_" + typ
    if EXCLUDE.search(t):
        return False, "excluded_type"
    if INCLUDE.search(t):
        pass
    elif typ == "Architectural fragment" and FACE_ARCH.search(t) and CERAMIC.search(o.get("medium_display") or ""):
        pass  # comparison group
    else:
        return False, "no_type_keyword"
    s = o.get("date_start")
    if isinstance(s, int) and s >= 1900:
        return False, "modern"
    return True, ""


def fetch_iiif(h: Http, image_id: str, dest_stem: Path) -> tuple[Path, str, int, str]:
    last = None
    for width in (1686, 843):
        url = IIIF.format(image_id, width)
        for attempt in range(3):
            try:
                r = h.get(url, headers={"AIC-User-Agent": "wadang-dataset (open research dataset)"},
                          allow_redirects=True)
            except Exception as e:
                last = e
                time.sleep(5)
                continue
            b = r.content
            ext = sniff_ext(b) if r.status_code == 200 else None
            if ext:
                p = dest_stem.with_suffix(ext)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b)
                return p, sha256_bytes(b), len(b), url
            last = RuntimeError(f"HTTP {r.status_code} {r.headers.get('content-type')} {url}")
            time.sleep(10 * (attempt + 1))
    raise last  # type: ignore[misc]


def save_policy(h: Http) -> tuple[str, str]:
    dest = EVIDENCE / f"{SRC}_policy_2026-11-08.html"
    txt = dest.read_text(encoding="utf-8", errors="replace") if dest.exists() else None
    if txt is None:
        try:
            r = h.get(POLICY_URL, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and "Just a moment" not in r.text:
                txt = r.text
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
    return "2026-11-08 (policy page blocked by Cloudflare; API field used)", QUOTE_FALLBACK


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    h = Http(rate_per_sec=1)
    meta_dir, img_dir = RAW / SRC / "meta", RAW / SRC / "img"
    for d in (meta_dir, img_dir):
        d.mkdir(parents=True, exist_ok=True)

    recs: dict[int, dict] = {}
    per = {}
    for ph in PHRASES:
        total, data = search_phrase(h, ph)
        per[ph] = total
        for o in data:
            recs[o["id"]] = o
    tot_af, af = search_arch_fragments(h)
    per["artwork_type_title=Architectural fragment"] = tot_af
    for o in af:
        recs[o["id"]] = o
    wlog("Stage1", "AIC 计数：Elasticsearch 短语检索（title/artwork_type/classification/term）+ 全部 Architectural fragment",
         f"POST {SEARCH} {{query:{{multi_match:{{query:<phrase>,type:phrase}}}}}}", json.dumps(per, ensure_ascii=False),
         f"并集 {len(recs)} 件")
    if a.count_only:
        return 0
    typed, reasons = [], {}
    for o in recs.values():
        ok, why = type_ok(o)
        if ok:
            typed.append(o)
        else:
            reasons[why] = reasons.get(why, 0) + 1
    wlog("Stage1", "AIC 类型过滤（标题含屋瓦类关键词；或 Architectural fragment 且标题含 mask/face/head 且陶质 = 比较组；"
         "排除版画/素描/照片/地砖/墙砖；date_start ≥ 1900 排除）", "", len(recs),
         f"过滤后 {len(typed)}：{[(o['id'], o['title'], o['is_public_domain']) for o in typed]}；排除 {json.dumps(reasons, ensure_ascii=False)}")
    lic = [o for o in typed if o.get("is_public_domain") is True and o.get("image_id")]
    wlog("Stage1", "AIC 许可过滤：只收 is_public_domain == true 且有 image_id", "", len(typed),
         f"通过 {len(lic)}；不通过 {len(typed) - len(lic)}")

    ev_date, quote = save_policy(h)
    prev = {r["image_key"]: r for r in read_jsonl(STAGING / f"{SRC}.jsonl")}
    rows, fail, n_imgs, fail_list = [], 0, 0, []
    for o in lic:
        sid = str(o["id"])
        try:
            full = h.json(ART.format(sid))["data"]
        except Exception:
            full = o
        (meta_dir / f"{sid}.json").write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
        img_ids = [o["image_id"]] + list(o.get("alt_image_ids") or [])
        n_imgs += len(img_ids)
        for n, iid in enumerate(img_ids):
            key = f"{sid}#{n}"
            if key in prev and (ROOT / prev[key]["raw_file"]).exists():
                pr = prev[key]
                p, sha, nb, url = ROOT / pr["raw_file"], pr["sha256"], pr["bytes"], pr["image_url"]
            else:
                try:
                    p, sha, nb, url = fetch_iiif(h, iid, img_dir / f"{sid}_{n}")
                except Exception as e:
                    fail += 1
                    fail_list.append(f"{sid}#{n}: {e}")
                    continue
            w, hh = image_size(p)
            rows.append(stage_row(
                source=SRC, source_id=sid, image_key=key, image_index=n,
                source_url=f"https://www.artic.edu/artworks/{sid}", image_url=url,
                iiif_manifest=f"https://api.artic.edu/api/v1/artworks/{sid}/manifest.json",
                record_file=f"raw/{SRC}/meta/{sid}.json",
                title_raw=o.get("title"), object_name_raw=o.get("artwork_type_title"),
                description_raw=re.sub(r"<[^>]+>", "", full.get("description") or "") or None,
                classification_raw="; ".join(o.get("classification_titles") or []) or None,
                culture_raw="; ".join(o.get("culture") or []) if isinstance(o.get("culture"), list) else o.get("culture"),
                period_raw=o.get("style_title"), date_raw=o.get("date_display"),
                date_from=o.get("date_start"), date_to=o.get("date_end"),
                region_raw=(f"place_of_origin: {o['place_of_origin']}" if o.get("place_of_origin") else None),
                medium_raw=o.get("medium_display"), dimensions_raw=o.get("dimensions"),
                credit_line_raw=o.get("credit_line"),
                subjects_raw="; ".join((o.get("subject_titles") or []) + (o.get("term_titles") or [])) or None,
                holding_institution="Art Institute of Chicago", holding_institution_qid="Q239303",
                accession_number=o.get("main_reference_number"),
                license="CC0-1.0", rights_raw="is_public_domain: true",
                license_evidence=license_evidence(POLICY_URL, ev_date, quote, f"is_public_domain: true ({ART.format(sid)})"),
                rights_holder=None, attribution_text=f"Art Institute of Chicago, {o.get('credit_line') or ''}".strip(", "),
                modality_hint="photo", is_reproduction=False,
                raw_file=str(p.relative_to(ROOT)).replace("\\", "/"), sha256=sha, bytes=nb, width=w, height=hh,
                date_accessed=today(), mirror_of=None,
                notes=f"AIC department: {o.get('department_title')}; IIIF size requested: {url.split('/full/')[1].split('/')[0]}",
            ))
    write_jsonl(STAGING / f"{SRC}.jsonl", rows)
    wlog("Stage1", "AIC 下载（IIIF full/1686, 不行再 843,；不绕过 Cloudflare 验证）并写 staging/aic.jsonl",
         IIIF.format("<image_id>", "1686"), n_imgs,
         f"成功 {len(rows)} 张 / {len({r['source_id'] for r in rows})} 件；失败 {fail} {fail_list[:10]}")
    print(f"aic: {len(rows)} images, fail {fail}", fail_list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
