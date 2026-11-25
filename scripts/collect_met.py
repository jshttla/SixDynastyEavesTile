"""Stage 1 collector: The Metropolitan Museum of Art Open Access API.

Steps (方案 Stage 1): count -> type filter -> licence filter -> download.
    .venv/Scripts/python -I scripts/collect_met.py            # full run (idempotent)
    .venv/Scripts/python -I scripts/collect_met.py --count-only

Notes
- /public/collection/v1/search was retired 2026-10-01 (HTTP 410); v1.1/search is paginated (limit<=500, offset).
- Only isPublicDomain == true AND primaryImage non-empty. additionalImages = extra views of the same object.
- Museum text is copied verbatim into *_raw fields; no labels are assigned here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RAW, ROOT, STAGING, local_settings, Http, download_image, image_size, license_evidence, read_jsonl,  # noqa: E402
                    review, safe_ascii, setup_console, sha256_file, stage_row, today, wlog, EVIDENCE)

SRC = "met"
SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1.1/search"
OBJ = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"
QUERIES = ["antefix", "roof tile", "tile end", "eaves tile", "wadang", "roof-tile end", "end tile",
           "roof ornament", "onigawara"]
DEPTS = {6: "Asian Art", 13: "Greek and Roman Art", 3: "Ancient West Asian Art", 17: "Medieval Art",
         7: "The Cloisters", 14: "Islamic Art", 12: "European Sculpture and Decorative Arts", 10: "Egyptian Art",
         5: "Arts of Africa, Oceania, and the Americas"}

POLICY_URL = "https://www.metmuseum.org/policies/image-resources"
POLICY_QUOTE = ("On February 7, 2017, The Metropolitan Museum of Art implemented a new policy known as Open Access, "
                "which makes images of artworks it believes to be in the public domain widely and freely available "
                "for unrestricted use, and at no cost, in accordance with the Creative Commons Zero (CC0) designation")
SNAPSHOT_FALLBACK = Path(local_settings().get("met_snapshot_html", RAW / "met" / "image_resources_snapshot.html"))  # optional saved copy

INCLUDE = re.compile(r"antefix|roof[- ]?tile|tile[- ]end|end[- ]tile|eaves?[- ]?tile|ridge[- ]?tile|ridge[- ]end|"
                     r"onigawara|roof ornament|roof[- ]end|wadang|瓦|chiwei|shibi|roof[- ]finial", re.I)
# roof ornament matches must be ceramic
CERAMIC = re.compile(r"earthenware|terracotta|terra cotta|stoneware|ceramic|clay|porcelain|faience|glaze", re.I)
EXCLUDE = re.compile(r"floor tile|wall tile|tile panel|tile mosaic|kiln|stove tile|revetment|inkstone|ink stone|"
                     r"\bmodel\b|tegula|imbrex|pantile|flat tile|\bbrick\b|tile-?work|dado|cuerda seca", re.I)
KEEP_EVEN_IF_EXCLUDED = re.compile(r"antefix|tile[- ]end|end[- ]tile|eaves?[- ]?tile|onigawara", re.I)
REPRO = re.compile(r"\b(copy|replica|reproduction|facsimile|forgery|modern cast|plaster cast|electrotype)\b", re.I)


def get_json(h: Http, url: str, params: dict | None = None):
    """Met's CDN (Imperva) sometimes answers 403/429 to bursts: back off and retry."""
    for i in range(6):
        r = h.get(url, params=params)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        time.sleep(15 * (i + 1))
    raise RuntimeError(f"giving up {url} {params}")


def save_policy(h: Http) -> str:
    """Save the policy page as evidence; returns the evidence date string."""
    dest = EVIDENCE / f"{SRC}_policy_2026-11-08.html"
    if dest.exists():
        head = dest.read_text(encoding="utf-8", errors="replace")[:20]
        return "2026-11-06 snapshot (2026-11-08 live fetch HTTP 429)" if head.startswith("<!-- snapshot") else "2026-11-08"
    EVIDENCE.mkdir(exist_ok=True)
    try:
        r = h.get(POLICY_URL, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and "Creative Commons Zero" in r.text:
            dest.write_text(r.text, encoding="utf-8")
            return "2026-11-08"
    except Exception:
        pass
    # live fetch blocked (Imperva 429) -> keep the 2026-11-06 snapshot from 03_下载素材 and say so
    if SNAPSHOT_FALLBACK.exists():
        txt = SNAPSHOT_FALLBACK.read_text(encoding="utf-8", errors="replace")
        dest.write_text("<!-- snapshot of " + POLICY_URL + " taken 2026-11-06 (03_下载素材); live fetch on 2026-11-08 "
                        "returned HTTP 429 from Imperva -->\n" + txt, encoding="utf-8")
        return "2026-11-06 snapshot (2026-11-08 live fetch HTTP 429)"
    return "2026-11-08 (page not retrievable)"


def region_raw(o: dict) -> str | None:
    parts = []
    for k in ("geographyType", "city", "state", "county", "country", "region", "subregion", "locale", "locus",
              "excavation", "river"):
        v = (o.get(k) or "").strip()
        if v:
            parts.append(f"{k}: {v}")
    return "; ".join(parts) or None


GENERIC_NAMES = {"", "tile", "tiles", "fragment", "architectural fragment", "architectural element", "ornament",
                 "relief", "plaque", "sculpture", "head", "mask", "terracotta"}
NOT_OBJECT_CLASSES = re.compile(r"prints|drawings|photographs|paintings|books|textiles", re.I)


def type_ok(o: dict) -> tuple[bool, str]:
    text = " | ".join([o.get("objectName") or "", o.get("title") or "", o.get("classification") or ""])
    if not INCLUDE.search(text):
        return False, "no_type_keyword"
    if NOT_OBJECT_CLASSES.search(o.get("classification") or ""):
        return False, "not_an_object (print/drawing/photo)"
    name = (o.get("objectName") or "").strip()
    if not INCLUDE.search(name) and name.lower() not in GENERIC_NAMES:
        # e.g. "Censer in the Form of Sparrow on a Roof Tile": the object is a censer, not a tile
        return False, "object_name_not_tile"
    if EXCLUDE.search(text) and not KEEP_EVEN_IF_EXCLUDED.search(text):
        return False, "excluded_type"
    m = INCLUDE.search(text)
    if m and m.group(0).lower() in ("roof ornament",) and not CERAMIC.search(o.get("medium") or ""):
        return False, "roof_ornament_not_ceramic"
    b = o.get("objectBeginDate")
    if isinstance(b, int) and b >= 1900:
        return False, "modern"
    return True, ""


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    h = Http(rate_per_sec=3)
    cache = RAW / SRC / "cache"
    meta_dir = RAW / SRC / "meta"
    img_dir = RAW / SRC / "img"
    for d in (cache, meta_dir, img_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. count ------------------------------------------------------------
    ids: dict[int, set] = {}
    for q in QUERIES:
        per = {}
        for dep in DEPTS:
            off, got = 0, []
            total = None
            while True:
                r = get_json(h, SEARCH, {"q": q, "hasImages": "true", "departmentId": dep, "limit": 500, "offset": off})
                total = r.get("total", 0) if r else 0
                batch = (r or {}).get("objectIDs") or []
                got += batch
                off += 500
                if not batch or off >= total:
                    break
            per[DEPTS[dep]] = total
            for i in got:
                ids.setdefault(i, set()).add(q)
        wlog("Stage1", f"Met 计数：q={q!r}，hasImages=true，按 9 个部门分别检索",
             f"{SEARCH}?q={q}&hasImages=true&departmentId=<id>&limit=500&offset=<n>",
             json.dumps(per, ensure_ascii=False), "原始返回数（v1.1 模糊匹配，未过滤）")
    wlog("Stage1", "Met：9 个检索词 × 9 个部门的 objectID 去重并集", "", len(ids), "进入逐件读取与类型过滤")
    if a.count_only:
        return 0

    # 2. fetch + type filter ---------------------------------------------------
    recs: dict[int, dict] = {}
    fail = 0
    for n, oid in enumerate(sorted(ids), 1):
        cp = cache / f"{oid}.json"
        if cp.exists():
            o = json.loads(cp.read_text(encoding="utf-8"))
        else:
            try:
                o = get_json(h, OBJ.format(oid))
            except Exception:
                o = None
            if not o:
                fail += 1
                continue
            cp.write_text(json.dumps(o, ensure_ascii=False), encoding="utf-8")
        recs[oid] = o
        if n % 200 == 0:
            print(f"  fetched {n}/{len(ids)}", flush=True)
    reasons: dict[str, int] = {}
    typed = []
    for oid, o in recs.items():
        ok, why = type_ok(o)
        if ok:
            typed.append(o)
        else:
            reasons[why] = reasons.get(why, 0) + 1
    dep_ct: dict[str, int] = {}
    for o in typed:
        dep_ct[o.get("department")] = dep_ct.get(o.get("department"), 0) + 1
    wlog("Stage1", "Met 逐件读取 + 类型过滤（objectName/title/classification 含 antefix/roof tile/tile end/eaves tile/"
         "ridge tile/onigawara/瓦 等；排除地砖/墙砖/revetment/砚/模型/tegula/imbrex/窑具；roof ornament 须陶质；"
         "objectBeginDate ≥ 1900 排除）", OBJ.format("<objectID>"),
         f"读取 {len(recs)}（失败 {fail}）",
         f"类型过滤后 {len(typed)}；按部门 {json.dumps(dep_ct, ensure_ascii=False)}；排除原因 {json.dumps(reasons)}")

    # 3. licence filter --------------------------------------------------------------
    lic_ok = [o for o in typed if o.get("isPublicDomain") is True and (o.get("primaryImage") or "").strip()]
    not_pd = sum(1 for o in typed if o.get("isPublicDomain") is not True)
    no_img = sum(1 for o in typed if o.get("isPublicDomain") is True and not (o.get("primaryImage") or "").strip())
    n_imgs = sum(1 + len(o.get("additionalImages") or []) for o in lic_ok)
    wlog("Stage1", "Met 许可过滤：只收 isPublicDomain == true 且 primaryImage 非空", "", len(typed),
         f"通过 {len(lic_ok)} 件（共 {n_imgs} 张图，含 additionalImages）；非公有领域 {not_pd}；公有领域但无图 {no_img}")

    # 4. download -------------------------------------------------------------
    ev_date = save_policy(h)
    rows, dl_fail, n_repro = [], 0, 0
    prev = {r["image_key"]: r for r in read_jsonl(STAGING / f"{SRC}.jsonl")}
    for o in lic_ok:
        oid = o["objectID"]
        sid = str(oid)
        (meta_dir / f"{sid}.json").write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
        urls = [o["primaryImage"]] + [u for u in (o.get("additionalImages") or []) if u]
        title_txt = " ".join([o.get("title") or "", o.get("objectName") or ""])
        repro = bool(REPRO.search(title_txt) or re.search(r"\bplaster\b", o.get("medium") or "", re.I))
        n_repro += repro
        for n, u in enumerate(urls):
            key = f"{sid}#{n}"
            stem = img_dir / f"{sid}_{n}"
            p = None
            if key in prev and (ROOT / prev[key]["raw_file"]).exists() and prev[key]["image_url"] == u:
                p = ROOT / prev[key]["raw_file"]
                sha, nb = prev[key]["sha256"], prev[key]["bytes"]
            else:
                try:
                    p, sha, nb = download_image(h, u, stem)
                except Exception as e:
                    dl_fail += 1
                    print("  download failed", sid, n, e)
                    continue
            try:
                w, hgt = image_size(p)
            except Exception:
                dl_fail += 1
                continue
            period = "; ".join(x for x in [o.get("period"), o.get("dynasty"), o.get("reign")] if x) or None
            rows.append(stage_row(
                source=SRC, source_id=sid, image_key=key, image_index=n,
                source_url=o.get("objectURL") or f"https://www.metmuseum.org/art/collection/search/{oid}",
                image_url=u, iiif_manifest=None,
                record_file=str((meta_dir / f"{sid}.json").relative_to(ROOT)).replace("\\", "/"),
                title_raw=o.get("title") or None, object_name_raw=o.get("objectName") or None,
                description_raw=None, classification_raw=o.get("classification") or None,
                culture_raw=o.get("culture") or None, period_raw=period, date_raw=o.get("objectDate") or None,
                date_from=o.get("objectBeginDate"), date_to=o.get("objectEndDate"),
                region_raw=region_raw(o), medium_raw=o.get("medium") or None,
                dimensions_raw=o.get("dimensions") or None, credit_line_raw=o.get("creditLine") or None,
                subjects_raw="; ".join(t.get("term", "") for t in (o.get("tags") or []) if t.get("term")) or None,
                holding_institution="The Metropolitan Museum of Art", holding_institution_qid="Q160236",
                accession_number=o.get("accessionNumber") or None,
                license="CC0-1.0",
                rights_raw=f"isPublicDomain: true (API {OBJ.format(oid)})",
                license_evidence=license_evidence(POLICY_URL, ev_date, POLICY_QUOTE, f"isPublicDomain: true ({OBJ.format(oid)})"),
                rights_holder=None,
                attribution_text=f"The Metropolitan Museum of Art, {o.get('creditLine') or ''}".strip(", "),
                modality_hint="rubbing" if re.search(r"rubbing", title_txt, re.I) else "photo",
                is_reproduction=repro,
                raw_file=str(p.relative_to(ROOT)).replace("\\", "/"), sha256=sha, bytes=nb, width=w, height=hgt,
                date_accessed=today(), mirror_of=None,
                notes=f"Met department: {o.get('department')}; matched queries: {sorted(ids.get(oid, []))}",
            ))
    n_obj = len({r["source_id"] for r in rows})
    out = STAGING / f"{SRC}.jsonl"
    from common import write_jsonl
    write_jsonl(out, rows)
    wlog("Stage1", "Met 下载原图（原样保存，不重压缩）并写 staging/met.jsonl",
         "GET primaryImage / additionalImages (images.metmuseum.org/CRDImages/.../original/...)", n_imgs,
         f"成功 {len(rows)} 张 / {n_obj} 件；失败 {dl_fail}；记录标注为复制品 {n_repro} 件；许可证据日期 {ev_date}")
    print(f"met: {len(rows)} images / {n_obj} objects, failures {dl_fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
