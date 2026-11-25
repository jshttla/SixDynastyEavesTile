"""Stage 1 collector: J. Paul Getty Museum (Linked Art API + SPARQL), source key `getty`.

    .venv/Scripts/python -I scripts/collect_getty.py [--count-only]

- Candidates: SPARQL over rdfs:label of crm:E22_Human-Made_Object (regex), then Python type filter on the
  Primary Title / Object-Work-Type statement.
- Licence is checked PER IMAGE: data.getty.edu/media/image/<uuid> -> subject_to -> Right classified_as
  http://creativecommons.org/publicdomain/zero/1.0/ (CC0) and clearance 'download'. Anything else -> REVIEW_QUEUE.
- Deaccessioned objects are skipped (no current holding institution).
- Museum description texts are NOT copied: the IIIF manifests state "Texts provided here are © J. Paul Getty Trust,
  licensed under CC BY 4.0", which would conflict with the dataset's CC0 metadata. Short factual fields only.
- Image: IIIF full/full/0/default.jpg (JPEG derivative of the TIFF master; bytes saved as delivered).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EVIDENCE, RAW, ROOT, STAGING, Http, download_image, image_size, license_evidence,  # noqa: E402
                    read_jsonl, review, setup_console, stage_row, today, wlog, write_jsonl)

SRC = "getty"
SPARQL = "https://data.getty.edu/museum/collection/sparql"
CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
POLICY_URL = "https://www.getty.edu/about/whatwedo/opencontent.html"
REGEX = ("antefix|roof tile|roof-tile|tile end|tile-end|eaves tile|ridge tile|onigawara|roof ornament|acroter|"
         "akroter|sima|waterspout|water spout|kalypter|eaves|gorgoneion")
INCLUDE = re.compile(r"antefix|roof[- ]?tile|tile[- ]end|end[- ]tile|eaves?[- ]?tile|ridge[- ]?tile|onigawara|"
                     r"roof ornament|acroter|akroter|\bsima\b|water ?spout|kalypter", re.I)
EXCLUDE = re.compile(r"floor tile|wall tile|kiln|inkstone|\bmodel\b|pantile|flat tile|\bbrick\b|photograph|"
                     r"\[|\bleaves\b|drawing|print", re.I)
KEEP = re.compile(r"antefix|tile[- ]end|eaves?[- ]?tile|onigawara", re.I)
TERRACOTTA = re.compile(r"terracotta|terra cotta|earthenware|clay|ceramic", re.I)
REPRO = re.compile(r"\b(copy|replica|reproduction|facsimile|forgery|plaster cast|electrotype|modern)\b", re.I)


def sparql_candidates(h: Http) -> dict[str, str]:
    out, off = {}, 0
    while True:
        q = f"""PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?o ?label WHERE {{ ?o a crm:E22_Human-Made_Object ; rdfs:label ?label .
  FILTER(REGEX(?label, "{REGEX}", "i")) }} ORDER BY ?o LIMIT 1000 OFFSET {off}"""
        r = h.get(SPARQL, params={"query": q}, headers={"Accept": "application/sparql-results+json"})
        r.raise_for_status()
        rows = r.json()["results"]["bindings"]
        for x in rows:
            out.setdefault(x["o"]["value"], x["label"]["value"])
        if len(rows) < 1000:
            break
        off += 1000
    return out


def stmt(o: dict, label: str) -> str | None:
    vals = []
    for x in o.get("referred_to_by", []) or []:
        labs = [c.get("_label") for c in x.get("classified_as", []) or []]
        if label in labs and x.get("content"):
            vals.append(str(x["content"]).strip())
    vals = list(dict.fromkeys(vals))
    return "; ".join(vals) or None


def ident(o: dict, label: str) -> str | None:
    for x in o.get("identified_by", []) or []:
        labs = [c.get("_label") for c in x.get("classified_as", []) or []]
        if label in labs:
            return x.get("content")
    return None


def year(s: str | None) -> int | None:
    if not s:
        return None
    m = re.match(r"^(-?)(\d{1,6})-", s)
    return int(m.group(1) + m.group(2)) if m else None


def image_rights(m: dict) -> tuple[str | None, str]:
    labs, ids = [], []
    for right in m.get("subject_to", []) or []:
        for c in right.get("classified_as", []) or []:
            labs.append(c.get("_label"))
            ids.append(c.get("id"))
    raw = "; ".join(f"{l} <{i}>" for l, i in zip(labs, ids))
    if CC0 in ids or "https://creativecommons.org/publicdomain/zero/1.0/" in ids:
        return "CC0-1.0", raw
    return None, raw


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    h = Http(rate_per_sec=3)
    meta_dir, img_dir, cache = RAW / SRC / "meta", RAW / SRC / "img", RAW / SRC / "cache"
    for d in (meta_dir, img_dir, cache):
        d.mkdir(parents=True, exist_ok=True)

    cands = sparql_candidates(h)
    wlog("Stage1", "Getty 计数：SPARQL 按 rdfs:label 正则检索 E22_Human-Made_Object",
         f"{SPARQL}  REGEX(?label, \"{REGEX}\", \"i\")", len(cands), "原始返回数（含 leaves/eaves 等误匹配，待二次过滤）")
    if a.count_only:
        return 0
    pre = {u: l for u, l in cands.items() if INCLUDE.search(l) and not (EXCLUDE.search(l) and not KEEP.search(l))}
    objs, deacc, reasons = [], 0, {}
    for u, l in pre.items():
        uid = u.rstrip("/").split("/")[-1]
        cp = cache / f"{uid}.json"
        if cp.exists():
            o = json.loads(cp.read_text(encoding="utf-8"))
        else:
            try:
                o = h.json(u)
            except Exception:
                reasons["fetch_fail"] = reasons.get("fetch_fail", 0) + 1
                continue
            cp.write_text(json.dumps(o, ensure_ascii=False), encoding="utf-8")
        if o.get("deaccessioned"):
            deacc += 1
            continue
        title = ident(o, "Primary Title") or ident(o, "Preferred Term") or o.get("_label")
        wtype = stmt(o, "Object/Work Type (Category)") or ""
        medium = stmt(o, "Materials Description") or ""
        text = f"{title} | {wtype}"
        m = INCLUDE.search(text)
        if not m:
            reasons["no_type_keyword"] = reasons.get("no_type_keyword", 0) + 1
            continue
        if re.match(r"acroter|akroter|sima|water ?spout|roof ornament|kalypter", m.group(0), re.I) and not TERRACOTTA.search(medium):
            reasons["roof_ornament_not_terracotta"] = reasons.get("roof_ornament_not_terracotta", 0) + 1
            continue
        if not o.get("shows"):
            reasons["no_image"] = reasons.get("no_image", 0) + 1
            continue
        objs.append((uid, o, title, wtype, medium))
    wlog("Stage1", "Getty 二次过滤：标题/Object-Work-Type 含 antefix/roof tile/tile end/ridge tile/acroterion/sima/waterspout 等"
         "（屋脊与檐口饰件须陶质），排除照片/素描/地砖等；跳过已除藏（deaccessioned）条目；须有图",
         "GET https://data.getty.edu/museum/collection/object/<uuid>", f"标签正则预筛 {len(pre)}",
         f"通过 {len(objs)} 件；deaccessioned {deacc}；其他排除 {json.dumps(reasons, ensure_ascii=False)}")

    # per-image licence -------------------------------------------------------------
    prev = {r["image_key"]: r for r in read_jsonl(STAGING / f"{SRC}.jsonl")}
    rows, n_imgs, n_cc0, n_other, fail = [], 0, 0, 0, 0
    lic_objs = set()
    for uid, o, title, wtype, medium in objs:
        (meta_dir / f"{uid}.json").write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
        page = next((x["id"] for x in o.get("subject_of", []) or [] if "getty.edu/art/collection" in x.get("id", "")),
                    f"https://www.getty.edu/art/collection/object/{uid}")
        manifest = next((x["id"] for x in o.get("subject_of", []) or [] if "/iiif/manifest/3/" in x.get("id", "")), None)
        ts = (o.get("produced_by") or {}).get("timespan") or {}
        date_raw = "; ".join(x.get("content", "") for x in ts.get("identified_by", []) or []) or None
        for n, vi in enumerate(o.get("shows") or []):
            n_imgs += 1
            mid = vi["id"].rstrip("/").split("/")[-1]
            mp = cache / f"media_{mid}.json"
            try:
                if mp.exists():
                    m = json.loads(mp.read_text(encoding="utf-8"))
                else:
                    m = h.json(vi["id"])
                    mp.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
            except Exception:
                fail += 1
                continue
            lic, rights_raw = image_rights(m)
            if lic is None:
                n_other += 1
                review(page, rights_raw or "(no rights statement on media record)", "Getty 图像 rights 不是 CC0",
                       "不收；如作者确认该 rights 属白名单可补收", kind="license")
                continue
            n_cc0 += 1
            ds = (m.get("digitally_shown_by") or [{}])[0]
            svc = next((ap["id"] for ap in ds.get("access_point", []) or []
                        if "IIIF image service" in [c.get("_label") for c in ap.get("classified_as", []) or []]), None)
            if not svc:
                fail += 1
                continue
            url = svc.rstrip("/") + "/full/full/0/default.jpg"
            key = f"{uid}#{n}"
            if key in prev and (ROOT / prev[key]["raw_file"]).exists() and prev[key]["image_url"] == url:
                p, sha, nb = ROOT / prev[key]["raw_file"], prev[key]["sha256"], prev[key]["bytes"]
            else:
                try:
                    p, sha, nb = download_image(h, url, img_dir / f"{uid}_{n}")
                except Exception as e:
                    fail += 1
                    print("  fail", uid, n, e)
                    continue
            w, hh = image_size(p)
            view_note = "; ".join(x.get("value", "") for x in m.get("subject_of", []) or [] if x.get("value"))
            lic_objs.add(uid)
            rows.append(stage_row(
                source=SRC, source_id=uid, image_key=key, image_index=n, source_url=page, image_url=url,
                iiif_manifest=manifest, record_file=f"raw/{SRC}/meta/{uid}.json",
                title_raw=title, object_name_raw=wtype or None, description_raw=None,
                classification_raw="; ".join(c.get("_label", "") for c in o.get("classified_as", []) or []) or None,
                culture_raw=stmt(o, "Culture"), period_raw=None, date_raw=date_raw,
                date_from=year(ts.get("begin_of_the_begin")), date_to=year(ts.get("end_of_the_end")),
                region_raw=(f"Creation Place Description: {stmt(o, 'Creation Place Description')}"
                            if stmt(o, "Creation Place Description") else None),
                medium_raw=medium or None, dimensions_raw=stmt(o, "Dimensions Description"),
                credit_line_raw=stmt(o, "Standard Credit Line") or stmt(o, "Credit Line"), subjects_raw=None,
                holding_institution="J. Paul Getty Museum", holding_institution_qid="Q731126",
                accession_number=ident(o, "Accession Number"),
                license="CC0-1.0", rights_raw=f"media rights: {rights_raw}",
                license_evidence=license_evidence(
                    POLICY_URL, "2026-11-08",
                    "Images provided here are believed to be in the public domain and are available under CC0 through "
                    "Getty's Open Content Program (IIIF manifest requiredStatement)",
                    f"{vi['id']} subject_to: {rights_raw}"),
                rights_holder=None, attribution_text=f"J. Paul Getty Museum, {stmt(o, 'Standard Credit Line') or ''}".strip(", "),
                modality_hint="photo", is_reproduction=bool(REPRO.search(title or "")),
                raw_file=str(p.relative_to(ROOT)).replace("\\", "/"), sha256=sha, bytes=nb, width=w, height=hh,
                date_accessed=today(), mirror_of=None,
                notes=f"Getty image description (verbatim): {view_note or '-'}; museum description text not copied "
                      f"(Getty states texts are CC BY 4.0)",
            ))
    write_jsonl(STAGING / f"{SRC}.jsonl", rows)
    wlog("Stage1", "Getty 逐图许可过滤（media 记录 subject_to = CC0）+ 下载 IIIF full/full JPEG，写 staging/getty.jsonl",
         "GET https://data.getty.edu/media/image/<uuid> ; GET https://media.getty.edu/iiif/image/<id>/full/full/0/default.jpg",
         n_imgs, f"CC0 {n_cc0} 张，非 CC0 {n_other} 张（进 REVIEW_QUEUE）；成功下载 {len(rows)} 张 / {len(lic_objs)} 件；失败 {fail}")
    print(f"getty: {len(rows)} images / {len(lic_objs)} objects; non-CC0 {n_other}; fail {fail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
