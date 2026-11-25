"""Stage 1 collector: Wikimedia Commons.

Rules (方案 §3 Commons row + Stage 1):
- Only LicenseShortName in CC0 / CC BY 3.0 (unported) / CC BY 4.0 / Public domain / KOGL Type 1. BY-SA, NC, ND, GFDL-only: rejected.
- For photographs of 3D objects marked PD-old-*/PD-Art: PD must hold for the PHOTO. If the file also carries a photo
  licence (own work CC...), that licence decides. PD with only object-age templates -> REVIEW_QUEUE, not collected.
- Files mirrored from a museum (Met, Smithsonian, Cleveland, ColBase ...) get mirror_of = museum item URL.
- Counting first (per category, wlog), then type filter (wlog), then licence filter (wlog), then download.

Usage: python -I scripts/collect_commons.py [--count-only] [--no-download]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EVIDENCE, LICENSES, RAW, REVIEW_QUEUE, ROOT, STAGING, Http, download_image, image_size,  # noqa: E402
                    license_evidence, review, safe_ascii, setup_console, sha256_file, stage_row, today, wlog,
                    write_jsonl)

API = "https://commons.wikimedia.org/w/api.php"
UA = "WadangDatasetBuilder/0.1 (https://huggingface.co/; research dataset of roof-tile images) python-requests"
SRC = "commons"
OUT_META = RAW / SRC / "meta"
OUT_IMG = RAW / SRC / "img"

# Seed categories (verified to exist 2026-11-08 via list=search srnamespace=14). need_context=True: branch mixes
# museum objects with in-situ / modern roofs, so a file must show museum/archaeological context to pass type filter.
SEEDS = [
    ("Category:Wadang", False),
    ("Category:Eaves tiles", False),
    ("Category:Eave tiles from China", False),
    ("Category:Roof tiles from China", False),
    ("Category:Roof tiles of the Han Dynasty", False),
    ("Category:Roof tiles of the Qin Dynasty", False),
    ("Category:Roof tiles of the Liao Dynasty", False),
    ("Category:Roof tiles of the Yuan Dynasty", False),
    ("Category:Roof tiles of the Ming Dynasty", True),
    ("Category:Pottery roof tile ends in the Shaanxi History Museum", False),
    ("Category:Pottery roof tiles in the Xi'an Museum", False),
    ("Category:Roof tiles by museum", False),
    ("Category:Roof tiles in the Tokyo National Museum", False),
    ("Category:Roof tiles in Korea", True),
    ("Category:Roof tiles in Gyeongju National Museum", False),
    ("Category:Nokimaru-Gawara", True),
    ("Category:Onigawara", True),
    ("Category:Shibi (roof tile)", True),
    ("Category:Antefixes", True),
    ("Category:Antefixes by civilization", False),
    ("Category:Etruscan antefixes", False),
    ("Category:Ancient Greek antefixes", False),
    ("Category:Ancient Roman antefixes", False),
    ("Category:Antefixes from Italy", False),
    ("Category:Antefixes from Cambodia", False),
]
MAX_DEPTH = 3
# subcategory names that lead away from objects (buildings, roofs, people, jewellery ...)
SKIP_CAT = re.compile(
    r"tiled roofs|roofs in|station|\bhouse\b|\bhall\b|church|kirche|freilicht|straße|strasse|utca|bridge|mosque|palace|"
    r"turret|castle|achelous|compluvia|pendant|necklace|azulejo|socarrat|kawara museum|kawara-kan|photographs created by|"
    r"\bin (19|20)\d\d\b|(january|february|march|april|may|june|july|august|september|october|november|december) \d{4}|"
    r"roof tiles in (austria|spain|portugal|india|the netherlands|france)|concrete|biberschwanz|pantile|marseille|"
    r"screen of|sindü|hohenfelden|yunnan nationalities|solatsu|nagasaka|awaji|sanshu|yasuda|tomebuta",
    re.I)
MODERN_EXCLUDE = re.compile(
    r"tiled roof|replica|reproduction|souvenir|modern|20th[- ]century|21st[- ]century|19th[- ]century|"
    r"kawara museum|復元|復刻|複製|模造|レプリカ|现代|仿制|复制品|令和|平成|昭和|"
    r"가마불|수막새틀|kiln firing",  # non-tile scenes / tools that sit in tile categories
    re.I)
CONTEXT = re.compile(
    r"museum|musée|museo|museu|muzeum|múzeum|antikensammlung|louvre|glyptothek|博物|資料館|资料馆|歴史館|历史馆|"
    r"excavat|出土|archaeolog|考古|etruscan|ancient|greek|roman|hellenistic|antike|antiquit|collection",
    re.I)
MUSEUM_MIRRORS = [
    (re.compile(r"metmuseum\.org", re.I), "met"),
    (re.compile(r"si\.edu|smithsonian|asia\.si\.edu|freer", re.I), "smithsonian"),
    (re.compile(r"clevelandart\.org", re.I), "cma"),
    (re.compile(r"colbase\.nich\.go\.jp|emuseum\.nich\.go\.jp|tnm\.jp", re.I), "colbase"),
    (re.compile(r"artic\.edu", re.I), "aic"),
    (re.compile(r"emuseum\.go\.kr|museum\.go\.kr", re.I), "emuseum"),
    (re.compile(r"getty\.edu/(?:art|museum)/collection", re.I), "getty"),
    (re.compile(r"rijksmuseum\.nl", re.I), "rijks"),
    (re.compile(r"thewalters\.org", re.I), "walters"),
]
POLICY_URL = "https://commons.wikimedia.org/wiki/Commons:Licensing"
POLICY_QUOTE = ("Commons:Licensing — 'Wikimedia Commons only accepts free content' ; per-file licence taken from "
                "the file page licence template (extmetadata LicenseShortName / License)")


def strip_html(s) -> str:
    if s is None:
        return ""
    s = re.sub(r"<br\s*/?>", " ", str(s))
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def api(h: Http, **params):
    params.setdefault("format", "json")
    return h.json(API, params=params)


def cat_info(h, cat):
    d = api(h, action="query", prop="categoryinfo", titles=cat)
    p = next(iter(d["query"]["pages"].values()))
    return p.get("categoryinfo", {}), ("missing" in p)


def members(h, cat, cmtype):
    out, cont = [], {}
    while True:
        d = api(h, action="query", list="categorymembers", cmtitle=cat, cmtype=cmtype, cmlimit=500, **cont)
        out += d["query"]["categorymembers"]
        if "continue" in d:
            cont = d["continue"]
        else:
            return out


def walk(h):
    """Return {category: (depth, need_context, files_count)} after BFS with skip rules."""
    cats: dict[str, tuple[int, bool, int]] = {}
    skipped = []
    queue = [(c, 0, nc) for c, nc in SEEDS]
    while queue:
        cat, depth, nc = queue.pop(0)
        if cat in cats:
            # keep the stricter of need_context flags only if first seen was strict? use the lenient one
            d0, nc0, f0 = cats[cat]
            cats[cat] = (d0, nc0 and nc, f0)
            continue
        info, missing = cat_info(h, cat)
        if missing:
            skipped.append(f"{cat} (missing)")
            continue
        cats[cat] = (depth, nc, int(info.get("files", 0)))
        if depth < MAX_DEPTH and int(info.get("subcats", 0)) > 0:
            for m in members(h, cat, "subcat"):
                t = m["title"]
                if SKIP_CAT.search(t):
                    skipped.append(t)
                    continue
                # children of a need_context branch inherit it; antefix children of country cats too
                queue.append((t, depth + 1, nc))
    return cats, skipped


def file_titles(h, cats):
    by_file = defaultdict(set)
    for cat in cats:
        for m in members(h, cat, "file"):
            by_file[m["title"]].add(cat)
    return by_file


def fetch_info(h, titles):
    out = {}
    titles = list(titles)
    for i in range(0, len(titles), 40):
        chunk = titles[i:i + 40]
        cont = {}
        while True:
            d = api(h, action="query", titles="|".join(chunk), prop="imageinfo|categories|info",
                    iiprop="url|extmetadata|size|sha1|mime", cllimit="max", inprop="url",
                    iiextmetadatalanguage="en", **cont)
            for p in d["query"]["pages"].values():
                t = p["title"]
                o = out.setdefault(t, {"pageid": p.get("pageid"), "title": t, "categories": [], "imageinfo": None,
                                       "fullurl": p.get("fullurl")})
                if p.get("imageinfo"):
                    o["imageinfo"] = p["imageinfo"][0]
                o["categories"] += [c["title"] for c in p.get("categories", [])]
            if "continue" in d:
                cont = d["continue"]
            else:
                break
    return out


LIC_CAT_BAD = re.compile(r"^Category:(CC-BY-SA|GFDL|CC-BY-NC|CC-BY-ND|FAL|Free Art|CC-BY-SA-)", re.I)
LIC_CAT_PD_OBJECT = re.compile(r"^Category:(PD[- ]old|PD-Art|PD[- ]?\d|Public domain due to age|PD-1996|PD-US-expired|"
                               r"PD-China|PD-Japan|PD-Korea|PD Old)", re.I)
# NB: CC-PD-Mark is auto-added by PD-old templates, so it is NOT evidence that the photographer released the photo.
LIC_CAT_PD_PHOTO = re.compile(r"^Category:(PD-self|PD-user|PD-author|Self-published work.*PD|"
                              r"Public domain released by)", re.I)
# red-light institutions of 方案 §3: visitor photos of their objects go to REVIEW_QUEUE instead of being collected
RED_INST = re.compile(r"British Museum|大英博物館|大英博物馆|Victoria and Albert|\bV&A\b|故宫博物院|Palace Museum,? Beijing|"
                      r"National Museum of China|中国国家博物馆|南京博物院|Nanjing Museum", re.I)


def classify_license(info: dict) -> tuple[str | None, str, str]:
    """Return (license_key or None, reason, rights_raw)."""
    em = info["imageinfo"].get("extmetadata", {})
    short = strip_html(em.get("LicenseShortName", {}).get("value", ""))
    code = strip_html(em.get("License", {}).get("value", "")).lower()
    cats = info["categories"]
    rights_raw = f"LicenseShortName: {short}; License: {code}; UsageTerms: {strip_html(em.get('UsageTerms', {}).get('value', ''))}"
    lic_cats = [c for c in cats if re.match(r"^Category:(CC|PD|GFDL|KOGL|Public domain|Self-published|FAL|"
                                            r"License|Copyrighted free use|Attribution)", c, re.I)]
    rights_raw += f"; licence categories: {', '.join(c.replace('Category:', '') for c in lic_cats)}"
    has_bad = any(LIC_CAT_BAD.match(c) for c in cats)
    restr = strip_html(em.get("Restrictions", {}).get("value", ""))
    if restr:  # external review round 2 (E-30): non-copyright restrictions (ita-mibac, personality rights ...) = unclear -> out
        return None, f"non-copyright restriction on the file page ({restr}) -> review", rights_raw + f"; Restrictions: {restr}"
    is_kogl1 = "kogl" in short.lower() or "kogl" in code or any(re.match(r"^Category:KOGL", c, re.I) for c in cats)
    if is_kogl1:
        if re.search(r"type\s*1|제1유형|kogl-type1|kogl type 1", short + " " + code + " " + " ".join(cats), re.I):
            return "KOGL-Type-1", "KOGL Type 1", rights_raw
        return None, "KOGL but not type 1", rights_raw
    if code in ("cc0",) or short.upper() == "CC0":
        return "CC0-1.0", "CC0", rights_raw
    if code == "cc-by-4.0":
        return "CC-BY-4.0", "CC BY 4.0", rights_raw
    if code == "cc-by-3.0":
        return "CC-BY-3.0", "CC BY 3.0 unported", rights_raw
    if re.fullmatch(r"cc-by-3\.0-[a-z]+", code):
        return None, f"ported {code} (not unported 3.0) -> review", rights_raw
    if code == "pd" or short.lower().startswith("public domain"):
        if any(re.match(r"^Category:CC-Zero", c) for c in cats) and not has_bad:
            return "CC0-1.0", "photo CC0 (CC-Zero category)", rights_raw
        # Is the PD statement about the photo or about the object?
        if any(LIC_CAT_PD_PHOTO.match(c) for c in cats) and not has_bad:
            return "Public-Domain", "PD released by photographer", rights_raw
        # PD (object) + a free photo licence: decide by the photo licence categories
        if any(re.match(r"^Category:CC-BY-4\.0", c) for c in cats) and not has_bad:
            return "CC-BY-4.0", "object PD; photo CC BY 4.0", rights_raw
        if any(re.match(r"^Category:CC-Zero", c) for c in cats) and not has_bad:
            return "CC0-1.0", "object PD; photo CC0", rights_raw
        if has_bad:
            return None, "PD for object but photo licence is BY-SA/GFDL", rights_raw
        if any(LIC_CAT_PD_OBJECT.match(c) for c in cats):
            return None, "PD-old/PD-Art template on 3D-object photo: PD may hold for the object, not the photo -> review", rights_raw
        return None, "PD without determinable basis -> review", rights_raw
    return None, f"licence not in whitelist: {short or code}", rights_raw


ITEM_URL = re.compile(r"/art/collection/search/\d+|/collection/search/\d+|/collection_items/|/art/\d{4}\.|"
                      r"/object/|edanmdm|/artworks/\d+|/collection/object/|/objects/\d+|relicId=", re.I)


WIKITEXT = RAW / SRC / "wikitext"  # file-page wikitext of museum mirrors (exact item URL; E-30)


def mirror_of(info) -> tuple[str | None, str | None]:
    """If the file was uploaded from a museum's open-access images, return (museum item URL, source key).
    The extmetadata may only link the museum's policy page; the file page wikitext (fetched by --fetch-wikitext) has the
    exact item URL."""
    em = info["imageinfo"].get("extmetadata", {})
    txt = " ".join(str(em.get(k, {}).get("value", "")) for k in ("Credit", "ImageDescription", "Artist", "Attribution"))
    wt = WIKITEXT / f"{info.get('pageid')}.txt"
    if wt.exists():
        txt += " " + wt.read_text(encoding="utf-8")
    urls = [html.unescape(u).rstrip(".,;)") for u in re.findall(r"https?://[^\s\"'<>{}|\[\]]+", txt)]  # stop at wikitext markup
    for rx, src in MUSEUM_MIRRORS:
        hits = [u for u in urls if rx.search(u)]
        if hits:
            items = [u for u in hits if ITEM_URL.search(u)]
            # a link to the museum's open-access POLICY page (Met donation template) is not the item page
            return (items[0] if items else None), src
        if rx.search(txt):
            return None, src
    return None, None


MIRROR_HOLDER = {"met": "The Metropolitan Museum of Art", "cma": "The Cleveland Museum of Art", "aic": "Art Institute of Chicago",
                 "getty": "J. Paul Getty Museum", "walters": "Walters Art Museum", "smithsonian": "Smithsonian Institution"}
UNKNOWN_ARTIST = re.compile(r"^(?:unknown(?: author)?\s*)+$", re.I)


def credit(em: dict, mo_src: str | None) -> tuple[str, str]:
    """(name for the attribution line, rights_holder).
    External copyright review 2026-11-09 (PROGRESS E-29): the file page's 'Attribution' field is the credit the licence
    requires (e.g. img_001475 must credit 'Auckland Museum', not 'Unknown author'); files donated by a museum are credited
    to that museum; repeated 'Unknown author' strings are collapsed."""
    attr_field = strip_html(em.get("Attribution", {}).get("value", "")) or None
    artist = (strip_html(em.get("Artist", {}).get("value", "")) or "").strip()
    if UNKNOWN_ARTIST.match(artist):
        artist = "unknown author"
    if mo_src in MIRROR_HOLDER:
        return MIRROR_HOLDER[mo_src], MIRROR_HOLDER[mo_src]
    if attr_field:
        return attr_field, attr_field
    return (artist or "unknown author"), (artist or "unknown")


OBJECT_DATE = re.compile(r"centur|\bB\.?\s?C\.?|\bA\.?\s?D\.?\b|\bBCE\b|\bCE\b|dynast|period|世纪|世紀|時代|时代|朝", re.I)


def date_fields(em: dict) -> tuple[str | None, str | None]:
    """(object date text for date_raw, note). The Commons 'Date' field is usually when the PHOTOGRAPH was taken
    (EXIF / upload form: '2021.01.28', '15 March 2014, 18:22:24') -> note only, never date_raw (it used to leak into
    culture_period_raw). Museum mirrors put the OBJECT's date there ('late 6th century B.C.') -> kept as date_raw."""
    d = strip_html(em.get("DateTimeOriginal", {}).get("value", "")) or None
    if not d:
        return None, None
    d = re.sub(r"date QS:P\d+.*$", "", d).strip()  # Wikidata quick-statement residue of the {{date}} template
    if OBJECT_DATE.search(d) and not re.search(r"exif", d, re.I):
        return d, None
    return None, f"photograph taken {d} (file page date; not the object's date)"


def work_title(t: str) -> str:
    """the title the file page supplies (file name without 'File:' and extension)"""
    t = re.sub(r"^File:", "", t or "")
    return re.sub(r"\.(jpe?g|png|tiff?|gif|webp)$", "", t, flags=re.I).strip()


def fetch_wikitext() -> None:
    """Network: wikitext of the staged files that are museum mirrors (exact museum item URL), -> raw/commons/wikitext/."""
    WIKITEXT.mkdir(parents=True, exist_ok=True)
    h = Http(rate_per_sec=1.0, headers={"User-Agent": UA})
    rows = [json.loads(x) for x in (STAGING / f"{SRC}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    n = 0
    for r in rows:
        if not r.get("mirror_of"):
            continue
        out = WIKITEXT / f"{r['source_id']}.txt"
        if out.exists():
            continue
        d = h.json(API, params={"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main", "format": "json",
                                "pageids": r["source_id"]})
        page = next(iter(d["query"]["pages"].values()))
        out.write_text(page["revisions"][0]["slots"]["main"]["*"], encoding="utf-8")
        n += 1
    wlog("Stage1·commons", "取镜像文件页 wikitext（精确馆方条目 URL；外部版权复核第二轮 E-30）",
         "python -I scripts/collect_commons.py --fetch-wikitext", sum(1 for r in rows if r.get("mirror_of")), f"新取 {n} 个，存 raw/commons/wikitext/")
    print("wikitext fetched", n)


def photo_date_note(em: dict) -> str | None:
    return date_fields(em)[1]


def restage() -> None:
    """Offline (no network): recompute credit, mirror and date fields of staging/commons.jsonl from raw/commons/meta."""
    sp = STAGING / f"{SRC}.jsonl"
    rows = [json.loads(x) for x in sp.read_text(encoding="utf-8").splitlines() if x.strip()]
    ch = Counter()
    for r in rows:
        rec = json.loads((ROOT / r["record_file"]).read_text(encoding="utf-8"))
        ii = rec["imageinfo"][0] if isinstance(rec["imageinfo"], list) else rec["imageinfo"]
        em = ii.get("extmetadata", {})
        mo_url, mo_src = mirror_of({**rec, "imageinfo": ii})
        name, holder = credit(em, mo_src)
        lic = LICENSES[r["license"]]
        new = {"attribution_text": f"“{work_title(r['title_raw'])}”, {name}, {lic['en']} ({lic['url']}), via Wikimedia Commons: {r['source_url']}",
               "rights_holder": holder,
               "mirror_of": mo_url or (f"museum:{mo_src}" if mo_src else None),
               "date_raw": date_fields(em)[0]}
        if mo_src in MIRROR_HOLDER:
            new["holding_institution"] = MIRROR_HOLDER[mo_src]
        base = re.sub(r"(?:; )?photograph taken .*?\(file page date; not the object's date\)", "", r.get("notes") or "")
        base = re.sub(r"(?:; )?not the object's date\)", "", base)  # fragments left by an earlier split on '; '
        notes = [x for x in base.split("; ") if x]
        pn = photo_date_note(em)
        if pn:
            notes.append(pn)
        new["notes"] = "; ".join(notes) or None
        for k, v in new.items():
            if r.get(k) != v:
                ch[k] += 1
                r[k] = v
    write_jsonl(sp, rows)
    wlog("Stage1·commons", "离线重算 staging 的署名 / 权利人 / mirror_of / 日期（外部版权复核 E-29；不联网，读 raw/commons/meta）",
         "python -I scripts/collect_commons.py --restage", len(rows), "改动字段计数：" + "; ".join(f"{k} {v}" for k, v in ch.most_common()))
    print("restaged", len(rows), dict(ch))


def holding_from_cats(cats, desc) -> str | None:
    for c in cats:
        m = re.search(r"in (?:the )?([A-Z][\w' .\-]*(?:Museum|Musée|Museo|Museu|Antikensammlung|Glyptothek)[\w' .\-]*)", c)
        if m:
            return m.group(1).strip()
        m = re.search(r"^Category:([\w' .\-]*(?:Museum|Musée|Museo)[\w' .\-]*?) collection", c)
        if m:
            return m.group(1).strip()
    m = re.search(r"((?:[A-Z][\w'.\-]+ )+(?:National )?Museum(?: of [A-Z][\w' ]+)?)", desc or "")
    return m.group(1).strip() if m else None


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--restage", action="store_true", help="offline: recompute credit / mirror / date fields of staging")
    ap.add_argument("--fetch-wikitext", action="store_true", help="network: wikitext of museum-mirror files")
    a = ap.parse_args()
    if a.fetch_wikitext:
        setup_console()
        return fetch_wikitext()
    if a.restage:
        setup_console()
        return restage()
    h = Http(rate_per_sec=1.0, headers={"User-Agent": UA})
    OUT_META.mkdir(parents=True, exist_ok=True)
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    date = today()

    # evidence snapshot of the licensing policy page
    ev = EVIDENCE / f"commons_licensing_{date}.html"
    if not ev.exists():
        r = h.get(POLICY_URL)
        ev.write_bytes(r.content)

    # 1) category tree + counts
    cats, skipped = walk(h)
    lines = "\n".join(f"  - {c}  (depth {d}, files {f}{', 需馆藏/考古语境' if nc else ''})" for c, (d, nc, f) in cats.items())
    total_listed = sum(f for _, _, f in cats.values())
    wlog("Stage1·commons", "Commons 分类遍历（种子分类 + 子类，深度 ≤ 3，按名称跳过建筑/屋顶/日期类子类）",
         f"{API}?action=query&list=categorymembers&cmtype=subcat|file&cmtitle=<分类>", f"{len(cats)} 个分类，分类内文件数合计 {total_listed}（含重复）",
         f"跳过的子类 {len(skipped)} 个：{'; '.join(skipped[:80])}{' …' if len(skipped) > 80 else ''}",
         extra="- 用到的分类：\n" + lines)
    by_file = file_titles(h, cats)
    wlog("Stage1·commons", "列出全部分类中的文件（去重后）", f"{API}?action=query&list=categorymembers&cmtype=file", len(by_file), "原始文件数（去重）")
    if a.count_only:
        return

    # 2) metadata
    infos = fetch_info(h, list(by_file))
    for t, info in infos.items():
        (OUT_META / f"{info['pageid']}.json").write_text(json.dumps({**info, "seed_categories": sorted(by_file[t])},
                                                                    ensure_ascii=False, indent=1), encoding="utf-8")

    # 3) type filter
    type_ok, type_reasons = [], Counter()
    for t, info in infos.items():
        ii = info.get("imageinfo")
        if not ii:
            type_reasons["no imageinfo"] += 1
            continue
        if ii.get("mime") not in ("image/jpeg", "image/png", "image/tiff", "image/webp"):
            type_reasons[f"mime {ii.get('mime')}"] += 1
            continue
        em = ii.get("extmetadata", {})
        desc = strip_html(em.get("ImageDescription", {}).get("value", ""))
        objname = strip_html(em.get("ObjectName", {}).get("value", ""))
        vis_cats = [c for c in info["categories"]]
        text = " ".join([t, desc, objname, " ".join(vis_cats)])
        if MODERN_EXCLUDE.search(text):
            type_reasons["modern/replica/in-situ roof wording"] += 1
            continue
        need_ctx = all(cats[c][1] for c in by_file[t])  # strict only if every seed path is strict
        if need_ctx and not CONTEXT.search(text):
            type_reasons["no museum/archaeological context in mixed branch"] += 1
            continue
        if min(ii.get("width", 0), ii.get("height", 0)) < 300:
            type_reasons["too small (<300 px)"] += 1
            continue
        type_ok.append(t)
    wlog("Stage1·commons", "类型过滤：只留器物照片/拓片（排除近现代、复制品、建筑屋面实景；混合分支要求有馆藏或考古语境；短边 ≥ 300 px）",
         "scripts/collect_commons.py（MODERN_EXCLUDE / CONTEXT 正则）", f"{len(infos)} → {len(type_ok)}",
         "排除原因：" + "; ".join(f"{k} {v}" for k, v in type_reasons.most_common()))

    # 4) licence filter
    lic_ok, lic_reasons, n_review = [], Counter(), 0
    for t in type_ok:
        info = infos[t]
        key, why, rights_raw = classify_license(info)
        em0 = info["imageinfo"].get("extmetadata", {})
        red_txt = " ".join([t, strip_html(em0.get("ImageDescription", {}).get("value", "")), " ".join(info["categories"])])
        if key and RED_INST.search(red_txt):
            key, why = None, f"object held by red-light institution ({RED_INST.search(red_txt).group(0)}) -> review"
        info["_lic"] = (key, why, rights_raw)
        if key:
            lic_ok.append(t)
            lic_reasons[f"accept: {why}"] += 1
        else:
            lic_reasons[f"reject: {why.split(':')[0]}"] += 1
            rq_url = info.get("fullurl") or f"https://commons.wikimedia.org/wiki/{t.replace(' ', '_')}"
            if "review" in why:
                n_review += 1
            if "review" in why and rq_url not in REVIEW_QUEUE.read_text(encoding="utf-8"):  # idempotent re-runs
                review(info.get("fullurl") or f"https://commons.wikimedia.org/wiki/{t.replace(' ', '_')}",
                       rights_raw[:400], why,
                       ("默认不收；方案 §3 把该馆列为红灯，访客自摄照片是否可收由作者决定" if "red-light" in why else
                        "默认不收；如能确认照片本身的许可是 CC0/CC BY/PD（不仅是器物），再收入"), kind="license")
    wlog("Stage1·commons", "许可过滤（只收 CC0 / CC BY 3.0 unported / CC BY 4.0 / 照片本身 PD / KOGL 제1유형）",
         "scripts/collect_commons.py classify_license()", f"{len(type_ok)} → {len(lic_ok)}",
         "; ".join(f"{k} {v}" for k, v in lic_reasons.most_common()) + f"；进 REVIEW_QUEUE {n_review} 条")
    if a.no_download:
        return

    # 5) download + staging
    rows, fails = [], Counter()
    staged_old = {}
    sp = STAGING / f"{SRC}.jsonl"
    if sp.exists():
        for line in sp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                staged_old[r["image_key"]] = r
    for t in lic_ok:
        info = infos[t]
        ii = info["imageinfo"]
        em = ii.get("extmetadata", {})
        key, why, rights_raw = info["_lic"]
        pid = str(info["pageid"])
        image_key = f"{pid}#0"
        page_url = info.get("fullurl") or f"https://commons.wikimedia.org/wiki/{t.replace(' ', '_')}"
        stem = OUT_IMG / f"commons_{pid}"
        try:
            url = ii["url"].split("?")[0]  # drop utm_* tracking parameters
            existing = sorted(OUT_IMG.glob(f"commons_{pid}.*"))
            note = None
            if ii.get("size", 0) > 40_000_000 and ii.get("thumburl"):
                url = ii["thumburl"]
                note = "original > 40 MB; downloaded Wikimedia-rendered thumbnail"
            if existing:  # downloaded in an earlier run: reuse bytes on disk
                p = existing[0]
                sha, nb = sha256_file(p), p.stat().st_size
                old = staged_old.get(image_key)
                if old and old.get("image_url"):
                    url = old["image_url"].split("?")[0]
                elif ii.get("size", 0) > 40_000_000 and ii.get("thumburl"):
                    url, note = ii["thumburl"], "original > 40 MB; downloaded Wikimedia-rendered thumbnail"
            else:
                p, sha, nb = download_image(h, url, stem)
            w, hh = image_size(p)
        except Exception as e:
            fails[type(e).__name__] += 1
            continue
        desc = strip_html(em.get("ImageDescription", {}).get("value", "")) or None
        credit_raw = strip_html(em.get("Credit", {}).get("value", "")) or None
        mo_url, mo_src = mirror_of(info)
        name, holder = credit(em, mo_src)
        notes = [f"Commons licence decision: {why}"]
        if mo_src:
            notes.append(f"mirrored from museum source '{mo_src}'; museum record is authoritative")
        if note:
            notes.append(note)
        if photo_date_note(em):
            notes.append(photo_date_note(em))
        vis_cats = [c.replace("Category:", "") for c in info["categories"]]
        lic = LICENSES[key]
        attribution = f"“{work_title(t)}”, {name}, {lic['en']} ({lic['url']}), via Wikimedia Commons: {page_url}"
        try:
            rows.append(stage_row(
                source=SRC, source_id=pid, image_key=image_key, image_index=0, source_url=page_url, image_url=url,
                record_file=f"raw/{SRC}/meta/{pid}.json",
                title_raw=t.replace("File:", ""), object_name_raw=strip_html(em.get("ObjectName", {}).get("value", "")) or None,
                description_raw=desc, subjects_raw="; ".join(vis_cats),
                date_raw=date_fields(em)[0],
                holding_institution=MIRROR_HOLDER.get(mo_src) or holding_from_cats(info["categories"], desc)
                or "unknown (Wikimedia Commons file; holding institution not stated)",
                credit_line_raw=credit_raw,
                license=key, rights_raw=rights_raw,
                license_evidence=license_evidence(page_url, date, f"file page licence: {strip_html(em.get('LicenseShortName', {}).get('value', ''))} ({why})",
                                                  rights_raw[:300]),
                rights_holder=holder, attribution_text=attribution,
                raw_file=p.relative_to(ROOT).as_posix(), sha256=sha, bytes=nb, width=w, height=hh, date_accessed=date,
                mirror_of=mo_url or (f"museum:{mo_src}" if mo_src else None), notes="; ".join(notes),
            ))
        except Exception as e:
            fails[f"stage_row {e}"] += 1
    n = write_jsonl(sp, rows)
    wlog("Stage1·commons", "下载原图（原样保存，不重压缩）并写 staging", f"upload.wikimedia.org → raw/commons/img/；staging/{SRC}.jsonl",
         f"{len(lic_ok)} 待下载 → {n} 成功", ("失败：" + "; ".join(f"{k} {v}" for k, v in fails.items())) if fails else "无失败")
    print(f"commons: staged {n}, fails {dict(fails)}")


if __name__ == "__main__":
    main()
