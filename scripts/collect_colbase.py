"""Stage 1 collector: ColBase (National Institutes for Cultural Heritage, Japan).

ColBase has no documented public API; the public website is a Nuxt SPA that reads JSON from
https://colbase.nich.go.jp/colbaseapi/v2/ with the fixed header `x-api-key: aaa` (hard-coded in the
site's own front-end bundle, sent by every browser). We use exactly that, politely (1 req/s), which is
equivalent to browsing the site item by item.

Licence: ColBase 利用規約 (evidence/colbase_terms_ja_2026-11-08.json) — compatible with CC BY 4.0.
Conservative rule (PROGRESS E-2): only items whose 所蔵者 is an NICH institution; third-party holders
(temples, private owners, other bodies) go to REVIEW_QUEUE because of 規約 §2 (third-party rights).

Usage:
    python -I scripts/collect_colbase.py --count-only
    python -I scripts/collect_colbase.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RAW, ROOT, STAGING, Http, download_image, image_size, license_evidence, review,  # noqa: E402
                    setup_console, sha256_file, stage_row, today, wlog, write_jsonl)

API = "https://colbase.nich.go.jp/colbaseapi/v2"
SRC = "colbase"
TERMS_URL = "https://colbase.nich.go.jp/pages/term?locale=ja"
TERMS_QUOTE = ("本利用ルールは、クリエイティブ・コモンズ・ライセンスの表示4.0国際…と互換性があり、"
               "本利用ルールが適用されるコンテンツはCC BYに従うことでも利用することができます。")

# title queries (方案 Stage 1 搜索词 + 宇瓦/鴟尾/垂木先瓦 variants) ; broad "瓦" is run to measure recall
TITLE_QUERIES = ["軒丸瓦", "軒平瓦", "鬼瓦", "瓦当", "瓦當", "鐙瓦", "宇瓦", "鴟尾", "垂木先瓦", "瓦"]
FREE_QUERIES = ["瓦当", "瓦當", "軒丸瓦", "鬼面文", "人面文"]

INCLUDE = re.compile(r"軒丸瓦|軒平瓦|軒瓦|鬼瓦|瓦当|瓦當|鐙瓦|宇瓦|鴟尾|垂木先瓦|隅木蓋瓦|鬼板|巴瓦|瓦頭")
EXCLUDE = re.compile(r"瓦経|瓦硯|瓦塔|瓦器|瓦質|瓦笵|笵|模造|複製|レプリカ")
REPRO = re.compile(r"模造|複製|レプリカ|模写|模型")
NICH = re.compile(r"東京国立博物館|京都国立博物館|奈良国立博物館|九州国立博物館|奈良文化財研究所|東京文化財研究所|"
                  r"国立文化財機構|Tokyo National Museum|Kyoto National Museum|Nara National Museum|Kyushu National Museum|"
                  r"三の丸尚蔵館")
NICH_ORG_QID: dict = {}  # QIDs are filled centrally in build_metadata from a verified Wikidata lookup


def headers():
    return {"x-api-key": "aaa", "Referer": "https://colbase.nich.go.jp/", "Accept": "application/json"}


def search_all(http: Http, key: str, word: str, count_only: bool) -> tuple[int, list[dict]]:
    out, page = [], 1
    total = None
    while True:
        params = {"locale": "ja", key: word, "with_image_file": 1, "limit": 100, "page": page}
        d = http.json(f"{API}/collection_items", params=params)
        total = d["resultset"]["count"]
        out += d["results"]
        if count_only or len(out) >= total or not d["results"]:
            break
        page += 1
    return total, out


def century_to_years(c):
    if c is None:
        return None, None
    c = int(c)
    if c > 0:
        return (c - 1) * 100 + 1, c * 100
    return c * 100, (c + 1) * 100 - 1


def item_url(org: str, key: str) -> str:
    return f"https://colbase.nich.go.jp/collection_items/{org}/{urllib.parse.quote(key, safe='')}?locale=ja"


def photographer(remarks: str | None) -> str | None:
    """Photographer named in the museum's own image caption (image_files[].remarks), e.g.
    '奈良国立博物館所蔵 [森村欣司]撮影' or a bare name '金井杜男'. '{}' / empty -> None."""
    t = (remarks or "").strip()
    if not t or t == "{}":
        return None
    m = re.search(r"\[([^\]]+)\]\s*撮影", t) or re.search(r"([^\s\[\]]+)\s*撮影", t)
    return (m.group(1) if m else t).strip()


def attribution(url: str, remarks: str | None) -> str:
    """ColBase terms: credit '出典：ColBase' + item URL; the museum's photographer credit is appended
    (external copyright review 2026-11-09, PROGRESS E-29)."""
    ph = photographer(remarks)
    return f"出典：ColBase（https://colbase.nich.go.jp/）{url}" + (f"；撮影：{ph}" if ph else "")


def restage() -> None:
    """Offline (no network): rebuild attribution_text of staging/colbase.jsonl from raw/colbase/meta."""
    sp = STAGING / f"{SRC}.jsonl"
    rows = [json.loads(x) for x in sp.read_text(encoding="utf-8").splitlines() if x.strip()]
    n = 0
    for r in rows:
        d = json.loads((ROOT / r["record_file"]).read_text(encoding="utf-8"))
        f = next((f for f in d.get("image_files", []) if f.get("url") == r["image_url"]), {})
        new = attribution(r["source_url"], f.get("remarks"))
        if new != r["attribution_text"]:
            r["attribution_text"] = new
            n += 1
    write_jsonl(sp, rows)
    wlog("Stage1·colbase", "离线重算 staging 署名：加馆方图像说明里的摄影者（撮影：某某；外部版权复核 E-29；不联网，读 raw/colbase/meta）",
         "python -I scripts/collect_colbase.py --restage", len(rows), f"署名改动 {n} 行")
    print("restaged", len(rows), "changed", n)


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--restage", action="store_true", help="offline: rebuild attribution_text from raw/colbase/meta")
    a = ap.parse_args()
    if a.restage:
        return restage()
    http = Http(rate_per_sec=1.0, headers=headers())
    date = today()

    # ---------------- 1. count ----------------
    cands: dict[int, dict] = {}
    for key, words in (("title", TITLE_QUERIES), ("free_word", FREE_QUERIES)):
        for w in words:
            total, res = search_all(http, key, w, a.count_only)
            q = f"{API}/collection_items?locale=ja&{key}={w}&with_image_file=1&limit=100"
            wlog("Stage1", f"ColBase 检索（{key}={w}，仅带图条目）", q, total,
                 "只计数" if a.count_only else f"已翻页取回 {len(res)} 条")
            for r in res:
                cands.setdefault(r["id"], r)
    if a.count_only:
        return
    n_union = len(cands)
    typed = {i: r for i, r in cands.items() if INCLUDE.search(r["title"] or "") and not EXCLUDE.search(r["title"] or "")}
    wlog("Stage1", "ColBase 类型过滤：检索结果去重合并后，按标题正则保留瓦当类（軒丸瓦|軒平瓦|軒瓦|鬼瓦|瓦当|瓦當|鐙瓦|宇瓦|鴟尾|垂木先瓦|隅木蓋瓦|鬼板|巴瓦|瓦頭），"
         "排除 瓦経|瓦硯|瓦塔|瓦器|瓦質|瓦笵|笵|模造|複製|レプリカ", "scripts/collect_colbase.py", n_union, f"类型过滤后 {len(typed)} 件")

    # ---------------- 2. details + licence filter ----------------
    meta_dir = RAW / SRC / "meta"
    img_dir = RAW / SRC / "img"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rows, n_lic_ok, n_third, n_noimg, n_fail = [], 0, 0, 0, 0
    for i, (cid, r) in enumerate(sorted(typed.items())):
        org, key = r["organization_path_name"], r["organization_item_key"]
        mp = meta_dir / f"{cid}.json"
        try:
            if mp.exists():
                d = json.loads(mp.read_text(encoding="utf-8"))
            else:
                d = http.json(f"{API}/collection_items/{org}/{urllib.parse.quote(key, safe='')}", params={"locale": "ja"})
                mp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        except Exception as e:
            n_fail += 1
            print("detail fail", cid, e)
            continue
        url = item_url(org, key)
        holder = d.get("shozousha") or d.get("organization_title") or ""
        if not NICH.search(holder):
            n_third += 1
            review(url, f"所蔵者：{holder}；タイトル：{d.get('title')}",
                   "所蔵者不是国立文化财机构下属馆所；ColBase 利用規約 §2 指出第三者权利需由利用者自行处理",
                   "暂不收录；若作者确认图像由该馆拍摄且无第三者权利，可按 ColBase-Terms 收录")
            continue
        files = d.get("image_files") or []
        if not files:
            n_noimg += 1
            continue
        n_lic_ok += 1
        title = d.get("title")
        descs = d.get("descriptions") or []
        desc_txt = "\n".join(str(x.get("body") or x.get("description") or x) if isinstance(x, dict) else str(x) for x in descs) or None
        m = re.search(r"[（(]([^（）()]*出土[^（）()]*)[）)]", title or "")
        region_raw = d.get("shutudochi") or d.get("seisakuchi") or (f"[title] {m.group(1)}" if m else None)
        df, dt = century_to_years(d.get("jidai_from"))[0], century_to_years(d.get("jidai_to"))[1]
        modality = "rubbing" if re.search(r"拓本|拓影", title or "") else None
        files_sorted = sorted(files, key=lambda f: (not f.get("main"), f.get("name")))
        for n, f in enumerate(files_sorted):
            iurl = f["url"]
            stem = img_dir / f"{cid}_{n}"
            existing = list(img_dir.glob(f"{cid}_{n}.*")) if img_dir.exists() else []
            try:
                if existing:
                    p = existing[0]
                    sha, nb = sha256_file(p), p.stat().st_size
                else:
                    p, sha, nb = download_image(http, iurl, stem)
                w, h = image_size(p)
            except Exception as e:
                n_fail += 1
                print("img fail", cid, iurl, e)
                continue
            attr = attribution(url, f.get("remarks"))
            rows.append(stage_row(
                source=SRC, source_id=f"{org}/{key}", image_key=f"{org}/{key}#{f['name']}", image_index=n,
                source_url=url, image_url=iurl, record_file=str(mp.relative_to(ROOT)).replace("\\", "/"),
                title_raw=title, object_name_raw=d.get("bunrui"), description_raw=desc_txt,
                classification_raw=d.get("bunrui"), culture_raw=None, period_raw=d.get("jidai_seiki"),
                date_raw=d.get("jidai_seiki"), date_from=df, date_to=dt, region_raw=region_raw,
                medium_raw=d.get("hinshitu_keijo"), dimensions_raw=d.get("houryo"),
                credit_line_raw=d.get("kizousha"), subjects_raw=d.get("meibun"),
                holding_institution=d.get("organization_title"), holding_institution_qid=NICH_ORG_QID.get(org),
                accession_number=key, license="ColBase-Terms",
                rights_raw=f"ColBase 利用規約（政府標準利用規約2.0準拠、CC BY 4.0互換）; 所蔵者: {holder}",
                license_evidence=license_evidence(TERMS_URL, date, TERMS_QUOTE, f"所蔵者: {holder}"),
                rights_holder=d.get("organization_title"), attribution_text=attr,
                modality_hint=modality, is_reproduction=bool(REPRO.search(title or "")),
                raw_file=str(p.relative_to(ROOT)).replace("\\", "/"), sha256=sha, bytes=nb, width=w, height=h,
                date_accessed=date, notes=f"insuu={d.get('insuu')}" if d.get("insuu") else None,
            ))
        if i % 25 == 0:
            print(f"{i}/{len(typed)} items, {len(rows)} images")
    n = write_jsonl(STAGING / f"{SRC}.jsonl", rows)
    wlog("Stage1", "ColBase 许可过滤与下载：只收所蔵者属国立文化财机构的条目（规约 §2 第三者权利），每件取条目自身的 image_files 原图（不取上级组条目的图）",
         "scripts/collect_colbase.py", len(typed),
         f"许可通过 {n_lic_ok} 件；第三方所蔵进 REVIEW_QUEUE {n_third} 件；条目自身无图 {n_noimg} 件；失败 {n_fail} 次；写入 staging/colbase.jsonl {n} 张图")


if __name__ == "__main__":
    main()
