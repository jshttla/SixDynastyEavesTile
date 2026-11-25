"""Stage 1 collector: e뮤지엄 (emuseum.go.kr, National Museum of Korea portal for all Korean museums).

No data.go.kr key -> the public website is used exactly like a visitor would:
  search  GET /headerSearch?keyword=<kw>&rows=36&pageNum=<n>
  detail  GET /detail?relicId=<id>
  image   GET /IMG/DOWN/<token>?imageTitle=..&usagePurposeType=기타&usagePurposeDesc=..  (the page's own
          "이미지 다운로드" form; no login, no CAPTCHA; we declare the purpose honestly)
Licence: per item, ONLY items whose detail page states 공공누리 "제 1유형 : 출처표시" (link kogl.or.kr licenseType1)
are kept -> license = KOGL-Type-1. Types 2/3/4 are dropped (commercial-use ban / no-derivatives).
The site's KOGL-type search filter (mckoglsvOp) is not honoured by plain GET requests, so every candidate's detail
page is read. Keyword result lists are large (수막새 ~33k), so each keyword has a cap on the number of detail pages
inspected (CAPS); face-related keywords are taken in full.

Usage: python -I scripts/collect_emuseum.py [--count-only]
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EVIDENCE, RAW, ROOT, STAGING, image_size, license_evidence, setup_console,  # noqa: E402
                    read_jsonl, sha256_bytes, sniff_ext, stage_row, today, wlog, write_jsonl)

SRC = "emuseum"
B = "https://www.emuseum.go.kr"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) WadangDatasetBuilder/0.1 (research dataset of roof-tile images)"
ROWS = 36
# keyword -> max number of result entries whose detail page is inspected (None = all)
CAPS = [  # (keyword, max result entries inspected or None=all, max type-1 objects kept or None=all)
    ("얼굴무늬 수막새", None, None),
    ("귀면와", None, 120),
    ("연화문 수막새", 288, 80),
    ("수막새", 288, 80),
    ("암막새", 216, 60),
    ("막새", 144, 30),
]
KOGL1_URL = "https://www.kogl.or.kr/info/licenseType1.do"
KOGL_QUOTE = ("공공누리 제1유형: 출처표시 — 출처표시 조건만 지키면 상업적 이용, 2차적 저작물 작성 등 자유이용 가능"
              " (KOGL Type 1: attribution; commercial use and modification allowed)")
POLICY_QUOTE = ("e뮤지엄 저작권 정책: '저작권법 제24조의2(공공저작물의 자유이용)에 따라 e뮤지엄 홈페이지에서 제공하는 자료는 "
                "저작물별로 적용된 이용조건에 따라 … 무료로, 자유롭게 이용 가능합니다 … 공공누리 표시가 부착된 저작물인지를 확인한 이후에 "
                "자유롭게 이용하시기 바랍니다'")
TYPE_KEEP = re.compile(r"수막새|암막새|막새|귀면와|와당|瓦當|鬼面瓦|치미|망새|연목와|사래기와|부연와|녹유.*와|기와")
MODERN_PERIOD = re.compile(r"근대|현대|近代|現代|대한제국|일제강점")  # 1897- : modern in Korean periodisation
TYPE_DROP = re.compile(r"복제|모조|재현|레플리카|탁본|사진|도면|엽서|기념품|현대")


class Sess:
    def __init__(self, rate=1.2):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.dt = 1.0 / rate
        self.t = 0.0
        self.s.get(B + "/main", timeout=60)

    def get(self, url, tries=4, **kw):
        last = None
        for i in range(tries):
            d = time.time() - self.t
            if d < self.dt:
                time.sleep(self.dt - d)
            self.t = time.time()
            try:
                kw.setdefault("timeout", 120)
                r = self.s.get(url, **kw)
                if r.status_code == 200:
                    return r
                last = RuntimeError(f"HTTP {r.status_code} {url}")
            except Exception as e:
                last = e
            time.sleep(3 * (i + 1))
        raise last


def search(sess, kw, page):
    r = sess.get(B + "/headerSearch", params={"keyword": kw, "rows": ROWS, "pageNum": page})
    t = r.text
    tot = re.search(r"총 <b>([\d,]+)</b>", t)
    ids = []
    for rid in re.findall(r"fnMaker\('I','(PS\d+)'", t):
        if rid not in ids:
            ids.append(rid)
    return (int(tot.group(1).replace(",", "")) if tot else 0), ids


def page_text(t):
    t2 = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    s = html.unescape(re.sub(r"<[^>]+>", " | ", t2))
    return re.sub(r"(\s*\|\s*)+", " | ", s)


LABELS = ["명칭", "다른명칭", "국적/시대", "분류", "재질", "작가", "크기", "소장품번호", "출토지", "용도/기능", "지정구분", "촬영연도"]


def parse_detail(t: str) -> dict:
    d: dict = {}
    m = re.search(r'id="relicTitle">\s*(.*?)\s*</p>\s*<p class="text">\s*(.*?)\s*</p>', t, re.S)
    if m:
        d["title"] = html.unescape(m.group(1)).strip()
        # the line under the title is '기관명' or '기관명 / 국가지정유산 등급' (e.g. '국립경주박물관 / 보물'): the grade is
        # not part of the institution's name (external review round 2, E-30) -> kept separately
        parts_m = [x.strip() for x in html.unescape(m.group(2)).split(" / ")]
        d["museum"] = parts_m[0]
        if len(parts_m) > 1:
            d["designation_line"] = " / ".join(parts_m[1:])
    s = page_text(t)
    i = s.find("| 명칭 |")
    seg = s[i:] if i >= 0 else s
    parts = [p.strip() for p in seg.split("|") if p.strip()]
    k = 0
    while k < len(parts) and parts[k] != "좋아요 (0)":
        p = parts[k]
        if p in LABELS and p not in d and k + 1 < len(parts):
            d[p] = re.sub(r"\s+", " ", parts[k + 1]).strip()
            k += 2
            continue
        if p.startswith("제 ") and "유형" in p:
            break
        k += 1
    m = re.search(r"· 소장품설명 \| (.*?) \| · ", s)
    if m:
        d["설명"] = m.group(1).strip()
    m = re.search(r"· 연관단어 \| (.*?) \|", s)
    if m:
        d["연관단어"] = m.group(1).strip()
    m = re.search(r'<p class="f18 bold mb15">\s*(제\s*\d\s*유형[^<]*)</p>\s*<p class="mb20">([^<]*)</p>\s*<a href="([^"]*kogl[^"]*)"', t)
    if m:
        d["kogl_text"] = html.unescape(m.group(1)).strip()
        d["kogl_sentence"] = html.unescape(m.group(2)).strip()
        d["kogl_link"] = m.group(3)
    else:
        m = re.search(r"(제\s*\d\s*유형[^<|]*)", t)
        d["kogl_text"] = m.group(1).strip() if m else None
    m = re.search(r"원본 해상도 (\d+) \* (\d+)", t)
    if m:
        d["orig_res"] = f"{m.group(1)}x{m.group(2)}"
    sf = re.search(r'<div class="slider-for">(.*?)<div class="slider-nav">', t, re.S)
    seg = sf.group(1) if sf else ""
    d["images"] = re.findall(r'<img src="(/IMG/[^"]+)"[^>]*alt="([^"]*)"', seg)
    fm = re.search(r'<form id="frmHiddenDownload"[^>]*>(.*?)</form>', t, re.S)
    hidden = {}
    if fm:
        for inp in re.findall(r"<input[^>]*>", fm.group(1)):
            n = re.search(r'name="([^"]+)"', inp)
            v = re.search(r'value="([^"]*)"', inp)
            if n:
                hidden[n.group(1)] = html.unescape(v.group(1)) if v else ""
    d["download_form"] = hidden
    return d


def kogl_attribution(d: dict) -> str:
    """KOGL official attribution template (kogl.or.kr licenseType1 page, 출처 표시 의무 1):
    본 저작물은 '기관명'에서 'OO년' 작성하여 공공누리 제1유형으로 개방한 '저작물명(작성자:OOO)'을 이용하였으며,
    해당 저작물은 '기관명, 홈페이지 주소'에서 무료로 다운받으실 수 있습니다."""
    title = d.get("title") or d.get("명칭") or ""
    museum = d.get("museum") or "소장기관 미상"
    yr = None
    m = re.search(r"(1[89]\d\d|20\d\d)", d.get("촬영연도") or "")
    if m:
        yr = m.group(1)
    year_part = f"'{yr}년' 작성하여 " if yr else ""
    ko = (f"본 저작물은 '{museum}'에서 {year_part}공공누리 제1유형으로 개방한 '{title}(작성자:{museum})'을 이용하였으며, "
          f"해당 저작물은 '{museum}, e뮤지엄({d['detail_url']})'에서 무료로 다운받으실 수 있습니다.")
    # the official template is Korean only; no translation is added (author, PROGRESS E-36)
    return ko


def kogl_type(d) -> str | None:
    m = re.search(r"제\s*(\d)\s*유형", d.get("kogl_text") or "")
    return m.group(1) if m else None


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--restage", action="store_true", help="rebuild staging from raw/ only (no network)")
    a = ap.parse_args()
    date = today()
    if a.restage:
        return restage()
    (RAW / SRC / "meta").mkdir(parents=True, exist_ok=True)
    (RAW / SRC / "img").mkdir(parents=True, exist_ok=True)
    sess = Sess()
    ev = EVIDENCE / f"kogl_licenseType1_{date}.html"
    if not ev.exists():
        try:
            ev.write_bytes(requests.get(KOGL1_URL, headers={"User-Agent": UA}, timeout=60).content)
        except Exception:
            pass
    ev2 = EVIDENCE / f"emuseum_copyright_{date}.html"
    if not ev2.exists():
        try:
            ev2.write_bytes(sess.get(B + "/copyright").content)
        except Exception:
            pass

    # 1) counts + candidate lists (capped)
    cand: dict[str, set] = {}
    order: dict[str, list] = {}
    for kw, cap, _q in CAPS:
        total, ids = search(sess, kw, 1)
        want = total if cap is None else min(total, cap)
        pages = math.ceil(want / ROWS)
        got = list(ids)
        for pg in range(2, pages + 1):
            _, ids2 = search(sess, kw, pg)
            got += [x for x in ids2 if x not in got]
        got = got[:want]
        order[kw] = got
        for x in got:
            cand.setdefault(x, set()).add(kw)
        wlog("Stage1·emuseum", f"e뮤지엄 网页检索计数：{kw}",
             f"GET {B}/headerSearch?keyword={quote(kw)}&rows={ROWS}&pageNum=1..{pages}", total,
             f"按上限取前 {want} 条进入逐件核查（{'全取' if cap is None else f'上限 {cap}'}；保留제1유형上限 {_q or '不限'}），实际取得 relicId {len(got)} 个")
    wlog("Stage1·emuseum", "e뮤지엄：各检索词候选合并去重", "", len(cand), "待逐件看详情页（公共누리类型 + 类型过滤）")
    if a.count_only:
        return

    # 2) details: type filter + KOGL filter
    details, type_ok, lic_ok = {}, [], []
    treasons, lreasons = Counter(), Counter()
    seen = set()
    for kw, _cap, quota in CAPS:
        kept_kw = 0
        for rid in order.get(kw, []):
            if quota is not None and kept_kw >= quota:
                break
            if rid in seen:
                continue
            seen.add(rid)
            try:
                t = sess.get(B + "/detail", params={"relicId": rid}).text
            except Exception as e:
                treasons[f"detail fetch failed {type(e).__name__}"] += 1
                continue
            d = parse_detail(t)
            d["relicId"] = rid
            d["keywords"] = sorted(cand[rid])
            d["detail_url"] = f"{B}/detail?relicId={rid}"
            details[rid] = d
            name = " ".join(filter(None, [d.get("title"), d.get("명칭"), d.get("분류")]))
            if not TYPE_KEEP.search(name):
                treasons["name/classification not a roof tile"] += 1
                continue
            if TYPE_DROP.search(name):
                treasons["replica/rubbing/photo/modern"] += 1
                continue
            if not d.get("images"):
                treasons["no image"] += 1
                continue
            type_ok.append(rid)
            kt = kogl_type(d)
            lreasons[f"KOGL type {kt}"] += 1
            (RAW / SRC / "meta" / f"emuseum_{rid}.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
            if kt == "1":
                lic_ok.append(rid)
                kept_kw += 1
        lreasons[f"kept for '{kw}'"] = kept_kw
    wlog("Stage1·emuseum", "类型过滤：名称/分类须为막새·귀면와·와당·기와类，排除복제·모조·재현·탁본·사진·현대；须有图",
         f"GET {B}/detail?relicId=<id>", f"{len(details)} 个详情页 → {len(type_ok)}",
         "排除：" + "; ".join(f"{k} {v}" for k, v in treasons.most_common()))
    wlog("Stage1·emuseum", "许可过滤：只收详情页明确写'제 1유형 : 출처표시'的条目（KOGL-Type-1）；2/3/4 型不收",
         "detail 页 공공누리 区块", f"{len(type_ok)} → {len(lic_ok)}", "; ".join(f"{k} {v}" for k, v in sorted(lreasons.items())))

    # 3) download all images of kept objects
    rows, fails = [], Counter()
    for rid in lic_ok:
        d = details[rid]
        title = d.get("title") or d.get("명칭") or ""
        museum = d.get("museum") or ""
        clean_title = re.sub(r"[\{\}\[\]/?.,;:|\)*~`!^\-_+<>@#$%&\\=\('‘“’”]", "", title)
        for idx, (src, alt) in enumerate(d["images"]):
            down = src.replace("/IMG/", "/IMG/DOWN/")
            params = dict(d.get("download_form") or {})
            params.update({"imageTitle": quote(f"{clean_title}_{rid}"), "imageIndex": 0,
                           "usagePurposeType": quote("기타"),
                           "usagePurposeDesc": quote("open research dataset (non-profit, academic)"), "usageEtcDesc": ""})
            try:
                r = sess.get(B + down, params=params)
                b = r.content
                ext = sniff_ext(b)
                if not ext or len(b) < 5000:
                    raise ValueError(f"not image ({r.headers.get('content-type')}, {len(b)} B)")
                p = (RAW / SRC / "img" / f"emuseum_{rid}_{idx}").with_suffix(ext)
                p.write_bytes(b)
                w, h = image_size(p)
            except Exception as e:
                fails[f"{type(e).__name__}: {str(e)[:50]}"] += 1
                continue
            attribution = kogl_attribution(d)
            rows.append(stage_row(
                source=SRC, source_id=rid, image_key=f"{rid}#{idx}", image_index=idx,
                source_url=d["detail_url"], image_url=B + down,
                record_file=f"raw/{SRC}/meta/emuseum_{rid}.json",
                title_raw=title, object_name_raw=d.get("다른명칭"), description_raw=d.get("설명"),
                classification_raw=d.get("분류"), period_raw=d.get("국적/시대"), date_raw=d.get("국적/시대"),
                region_raw=d.get("출토지"), medium_raw=d.get("재질"), dimensions_raw=d.get("크기"),
                subjects_raw=d.get("연관단어"), accession_number=d.get("소장품번호"),
                holding_institution=museum or "unknown (e뮤지엄)",
                license="KOGL-Type-1",
                rights_raw=f"{d.get('kogl_text')} | {d.get('kogl_sentence')} | {d.get('kogl_link')}",
                license_evidence=license_evidence(KOGL1_URL, date, KOGL_QUOTE, f"{d['detail_url']}: {d.get('kogl_text')}"),
                rights_holder=museum or None, attribution_text=attribution,
                raw_file=p.relative_to(ROOT).as_posix(), sha256=sha256_bytes(b), bytes=len(b), width=w, height=h,
                date_accessed=date, notes=f"image alt: {alt}; page original resolution {d.get('orig_res')}",
            ))
    n = write_jsonl(STAGING / f"{SRC}.jsonl", rows)
    wlog("Stage1·emuseum", "e뮤지엄：经详情页'이미지 다운로드'表单（用途 기타：open research dataset）下载제1유형条目全部图像，写 staging",
         f"GET {B}/IMG/DOWN/<token>?imageTitle=..&usagePurposeType=기타", f"{len(lic_ok)} 件 → {n} 张图",
         ("失败：" + "; ".join(f"{k} {v}" for k, v in fails.items())) if fails else "无失败")
    print(f"emuseum: {n} images from {len(lic_ok)} objects; fails {dict(fails)}")


def build_row(d: dict, idx: int, src: str, alt: str, p: Path, b_sha: str, nbytes: int, date: str) -> dict:
    title = d.get("title") or d.get("명칭") or ""
    museum = d.get("museum") or ""
    down = src.replace("/IMG/", "/IMG/DOWN/")
    w, h = image_size(p)
    rid = d["relicId"]
    return stage_row(
        source=SRC, source_id=rid, image_key=f"{rid}#{idx}", image_index=idx,
        source_url=d["detail_url"], image_url=B + down,
        record_file=f"raw/{SRC}/meta/emuseum_{rid}.json",
        title_raw=title, object_name_raw=d.get("다른명칭"), description_raw=d.get("설명"),
        classification_raw=d.get("분류"), period_raw=d.get("국적/시대"), date_raw=d.get("국적/시대"),
        region_raw=d.get("출토지"), medium_raw=d.get("재질"), dimensions_raw=d.get("크기"),
        subjects_raw=d.get("연관단어"), accession_number=d.get("소장품번호"),
        holding_institution=museum or "unknown (e뮤지엄)",
        license="KOGL-Type-1",
        rights_raw=f"{d.get('kogl_text')} | {d.get('kogl_sentence')} | {d.get('kogl_link')}",
        license_evidence=license_evidence(B + "/copyright", date, POLICY_QUOTE + " ; " + KOGL_QUOTE + f" ({KOGL1_URL})",
                                          f"{d['detail_url']}: {d.get('kogl_text')}"),
        rights_holder=museum or None, attribution_text=kogl_attribution(d),
        raw_file=p.relative_to(ROOT).as_posix(), sha256=b_sha, bytes=nbytes, width=w, height=h,
        date_accessed=d.get("date_accessed") or date, notes=f"image alt: {alt}; page original resolution {d.get('orig_res')}",
    )


def restage():
    from common import sha256_file
    date = today()
    rows = []
    n_modern = 0
    old = {}
    if (STAGING / f"{SRC}.jsonl").exists():  # keep the real access date / evidence of rows already collected
        old = {r["image_key"]: r for r in read_jsonl(STAGING / f"{SRC}.jsonl")}
    for mf in sorted((RAW / SRC / "meta").glob("emuseum_*.json")):
        d = json.loads(mf.read_text(encoding="utf-8"))
        if " / " in (d.get("museum") or ""):  # '국립경주박물관 / 보물': grade is not part of the name (E-30)
            parts_m = [x.strip() for x in d["museum"].split(" / ")]
            d["museum"], d["designation_line"] = parts_m[0], " / ".join(parts_m[1:])
        if kogl_type(d) != "1":
            continue
        if MODERN_PERIOD.search(d.get("국적/시대") or ""):
            n_modern += 1  # 方案: 近现代件不收
            continue
        for idx, (src, alt) in enumerate(d.get("images", [])):
            fs = sorted((RAW / SRC / "img").glob(f"emuseum_{d['relicId']}_{idx}.*"))
            if not fs:
                continue
            p = fs[0]
            row = build_row(d, idx, src, alt, p, sha256_file(p), p.stat().st_size, date)
            o = old.get(row["image_key"])
            if o:
                row["date_accessed"], row["license_evidence"] = o["date_accessed"], o["license_evidence"]
            rows.append(row)
    if n_modern:
        wlog("Stage1·emuseum", "e뮤지엄：排除국적/시대为근대·현대（近现代）的제1유형条目", "restage 过滤 MODERN_PERIOD", n_modern, "不入 staging")
    n = write_jsonl(STAGING / f"{SRC}.jsonl", rows)
    wlog("Stage1·emuseum", "e뮤지엄：按 KOGL 官方出处表示模板重建 staging 的 attribution_text（离线，基于 raw/emuseum/meta 与已下载图像）",
         "python -I scripts/collect_emuseum.py --restage", n, "staging 行数；署名格式见 kogl.or.kr licenseType1 页'출처 표시 의무'例句")
    print(f"restaged {n}")


if __name__ == "__main__":
    main()
