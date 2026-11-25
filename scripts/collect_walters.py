"""Walters Art Museum collector (static open-data dump on GitHub; API v1 closed in 2023).

Source of records: https://github.com/WaltersArtMuseum/api-thewalters-org (art.csv, media.csv), stored in raw/walters/dump/.
Licence: CC0 for images and metadata (policy page saved to evidence/walters_policy_2026-11-08.html). The policy says the
*longer text descriptions* on the website are GFDL, so description text is NOT copied into description_raw.

Usage: python -I scripts/collect_walters.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RAW, ROOT, STAGING, Http, download_image, image_size, license_evidence, safe_ascii,  # noqa: E402
                    setup_console, stage_row, today, wlog, write_jsonl)

DUMP = RAW / "walters" / "dump"
REPO = "https://raw.githubusercontent.com/WaltersArtMuseum/api-thewalters-org/main"
POLICY = "https://thewalters.org/about/policies/rights-reproductions/"
QUOTE = ("it has adopted the Creative Commons Zero: No Rights Reserved or CC0 license to waive copyright and allow for "
         "unrestricted use of digital images and metadata by any person, for any purpose")
TYPE_RE = re.compile(r"antefix|roof[\s-]*tile|tile[\s-]*end|end[\s-]*tile|eaves?[\s-]*tile|onigawara|ridge[\s-]*tile|"
                     r"roof[\s-]*end|wadang|瓦", re.I)
# Final scope decisions on the type-matched records (record text only). Reason strings go to WORKLOG.
EXCLUDE = {
    "11041": "Roof Tile with Bodhisattva seated on lion-dragon：屋脊人物俑（与 SMVK 屋脊骑士俑同样处理），不属瓦当/antefix/鬼瓦",
    "25346": "标题只写 'Roof Tile'，分类 'tiles, fragments'，无瓦当/檐头证据（平瓦）",
    "22771": "标题只写 'Roof Tile'（Roman），无瓦当/檐头证据（平瓦）",
}


def main():
    setup_console()
    csv.field_size_limit(10 ** 8)
    http = Http(rate_per_sec=2, timeout=120)
    DUMP.mkdir(parents=True, exist_ok=True)
    for f in ("art.csv", "media.csv", "README.md"):
        if not (DUMP / f).exists():
            (DUMP / f).write_bytes(http.get(f"{REPO}/{f}").content)
    art = list(csv.DictReader(open(DUMP / "art.csv", encoding="utf-8")))
    media = defaultdict(list)
    for m in csv.DictReader(open(DUMP / "media.csv", encoding="utf-8")):
        media[m["ObjectID"]].append(m)
    wlog("Stage1·walters", "下载 Walters 官方静态数据转储（API v1 已于 2023 年关闭）并计数",
         f"{REPO}/art.csv ; {REPO}/media.csv", f"art.csv {len(art)} 条记录；media.csv 涉及 {len(media)} 件",
         "README 与政策页均声明图像与元数据 CC0；政策页另称网站上较长的文字说明为 GFDL → 不抄 Description 字段")
    hits = [r for r in art if TYPE_RE.search(" ".join([r["ObjectName"], r["Title"], r["Keywords"], r["Classification"]]))]
    wlog("Stage1·walters", "关键词计数（ObjectName/Title/Keywords/Classification 含 antefix、roof tile、eaves tile、tile end 等）",
         "scripts/collect_walters.py", len(hits),
         "命中：" + "；".join(f"{r['ObjectID']} {r['Title']}" for r in hits))
    typed = [r for r in hits if r["ObjectID"] not in EXCLUDE]
    wlog("Stage1·walters", "类型二次过滤（排除屋脊人物俑与无檐头证据的平瓦）", "scripts/collect_walters.py", len(hits),
         f"保留 {len(typed)}；排除：" + "；".join(f"{k}: {v}" for k, v in EXCLUDE.items()))
    lic = [r for r in typed if media.get(r["ObjectID"])]
    wlog("Stage1·walters", "许可过滤：全馆 CC0（政策页），条目须有图像；馆方说明 CC0 例外仅限借展品与因年代未进入公有领域者，这些条目均为本馆藏古代器物",
         "scripts/collect_walters.py", len(typed), f"有图可下 {len(lic)} 件（无图：{[r['ObjectID'] + ' ' + r['Title'] for r in typed if not media.get(r['ObjectID'])]}）")

    out, fails = [], Counter()
    (RAW / "walters" / "meta").mkdir(parents=True, exist_ok=True)
    for r in lic:
        sid = r["ObjectID"]
        # no dots in the stem: download_image() uses Path.with_suffix(), which would eat ".354_0" in "48.354_0"
        safe = safe_ascii(f"{sid}_{r['AccessionNumber']}").replace(".", "-")
        ms = sorted(media[sid], key=lambda m: (-int(m["IsPrimary"] or 0), int(m["Rank"] or 99)))
        (RAW / "walters" / "meta" / f"{safe}.json").write_text(json.dumps({"art": r, "media": ms}, ensure_ascii=False, indent=1), encoding="utf-8")
        idx = 0
        for m in ms:
            if m["MediaType"] != "Image":
                continue
            if (m["MediaView"] or "").lower() in ("group",):
                fails["group_view_skipped"] += 1
                continue
            url = m["ImageURL"]
            try:
                p, sha, nb = download_image(http, url, RAW / "walters" / "img" / f"{safe}_{idx}")
            except Exception as e:  # noqa: BLE001
                fails["download"] += 1
                print("fail", url, e)
                continue
            w, h = image_size(p)
            out.append(stage_row(
                source="walters", source_id=sid, image_key=f"{sid}#{idx}", image_index=idx,
                source_url=r["ResourceURL"] or f"https://art.thewalters.org/object/{r['AccessionNumber']}", image_url=url,
                record_file=f"raw/walters/meta/{safe}.json",
                title_raw=r["Title"], object_name_raw=r["ObjectName"], description_raw=None,
                classification_raw=r["Classification"] or None, culture_raw=r["Culture"] or None,
                period_raw=" | ".join(x for x in [r["Period"], r["Dynasty"], r["Reign"], r["Style"]] if x) or None,
                date_raw=r["DateText"] or None,
                date_from=int(r["DateBeginYear"]) if re.fullmatch(r"-?\d+", r["DateBeginYear"] or "") else None,
                date_to=int(r["DateEndYear"]) if re.fullmatch(r"-?\d+", r["DateEndYear"] or "") else None,
                region_raw=None, medium_raw=r["Medium"] or None, dimensions_raw=r["Dimensions"] or None,
                credit_line_raw=(r["CreditLine"] or "").strip() or None, subjects_raw=r["Keywords"] or None,
                holding_institution="Walters Art Museum", holding_institution_qid="Q210081",
                accession_number=r["AccessionNumber"],
                license="CC0-1.0", rights_raw="Walters open data (art.csv/media.csv): images and metadata CC0 (README; rights-reproductions policy)",
                license_evidence=license_evidence(POLICY, today(), QUOTE, "github.com/WaltersArtMuseum/api-thewalters-org README: 'made available to the public under a CC0 license'"),
                rights_holder="Walters Art Museum",
                attribution_text=f"{r['Title']}, Walters Art Museum, {r['AccessionNumber']}, CC0 1.0",
                modality_hint="photo", is_reproduction=False,
                raw_file=p.relative_to(ROOT).as_posix(), sha256=sha, bytes=nb, width=w, height=h,
                date_accessed=today(),
                notes=f"MediaView: {m['MediaView']}; IsPrimary={m['IsPrimary']}; 馆方文字说明为 GFDL，未抄录",
            ))
            idx += 1
    write_jsonl(STAGING / "walters.jsonl", out)
    wlog("Stage1·walters", "下载原图（art.thewalters.org/images/raw/，即 media.csv 给出的 ImageURL）",
         "scripts/collect_walters.py", len(lic),
         f"成功 {len({o['source_id'] for o in out})} 件 / {len(out)} 张；{dict(fails)}；输出 staging/walters.jsonl")
    print("walters", len(out), dict(fails))


if __name__ == "__main__":
    main()
