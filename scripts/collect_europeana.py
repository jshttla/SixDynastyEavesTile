"""Collect Europeana items (antefixes / eaves-tile ends / ridge-end tiles) with open rights.

Rules (方案 §3): only CC0 / PDM / CC BY 3.0 / CC BY 4.0 rights URIs. BY-SA, NC, ND, InC, NoC-* are dropped.
Ported CC BY licences (e.g. by/3.0/nl) go to REVIEW_QUEUE, not into the dataset.

Usage: python -I scripts/collect_europeana.py [--count-only]
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EVIDENCE, RAW, ROOT, STAGING, Http, download_image, image_size, license_evidence,  # noqa: E402
                    review, safe_ascii, setup_console, stage_row, today, wlog, write_jsonl, read_jsonl)

KEY = "api2demo"  # Europeana public demo key
API = "https://api.europeana.eu/record/v2"
TERMS = ["antefix", "antefixes", "Stirnziegel", "antéfixe", "antefissa", "antefija", "antefixa",
         '"tile end"', '"eaves tile"', '"roof tile end"', "軒丸瓦", "軒平瓦", "鬼瓦", "onigawara", "瓦当", "막새",
         '"ridge tile"', '"roof tile"', "dakpan", "Dachziegel"]

# Pre-filter (broad, on search-result text) and final decision (on the full record text). Record text only, never file names.
PROVIDER_QUERIES = [
    ('tegel OR taktegel OR ändtegel OR takpanna OR tegelpanna OR takprydnad OR "roof-end tile" OR "roof tile"',
     'DATA_PROVIDER:"Museum of Far Eastern Antiquities"'),
]
PRE_RE = re.compile(
    r"antefi|stirnziegel|antéfix|antefiss|antefij|tile[\s-]*end|end[\s-]*tile|eaves?[\s-]*tile|roof[\s-]*tile|"
    r"軒丸瓦|軒平瓦|鬼瓦|onigawara|瓦当|瓦當|막새|ändtegel|taktegel|takpanna|tegelpanna|takprydnad|kalypter", re.I)
STRONG_RE = re.compile(
    r"antefi|stirnziegel|antéfix|antefiss|antefij|roof[\s-]*end[\s-]*tile|end[\s-]*tile|tile[\s-]*end|eaves?[\s-]*tile|"
    r"ändtegel|軒丸瓦|軒平瓦|鬼瓦|onigawara|瓦当|瓦當|막새", re.I)  # NB: Swedish takpanna/tegelpanna = any roof tile
DESC_POS_RE = re.compile(
    r"(mould|mold)-made antefix|terracotta antefix|the object is an antefix|is an antefix|antefix with|"
    r"framplatta(n)? till (ett )?taktegel|konkavt taktegel|ändtegel som suttit|takpanna i form av|fragment av taktegel", re.I)
NEG_TITLE_RE = re.compile(r"tsuba|parerplåt|netsuke|gravfigurin|figurine|skärva|flaska|pyxis|skizzenbuch|"
                          r"\bmodel\b|maquette|postcard|ansichtkaart", re.I)
HEDGE_RE = re.compile(r"könnte|possibly|may have been|perhaps|enligt inventariet är", re.I)
FIGURE_RE = re.compile(r"i form av (ryttare|munk|man\b)|takryttare|form av ryttare", re.I)


def decide(title: str, typ: str, desc: str, subj: str) -> tuple[bool, str]:
    t = f"{title} | {typ}"
    if NEG_TITLE_RE.search(t):
        return False, "excluded object type in title/type"
    if re.search(r"skulptur|figur", t, re.I) and FIGURE_RE.search(desc or ""):
        return False, "roof figurine (rider/monk), outside eaves-tile scope"
    if STRONG_RE.search(t):
        return True, "title/type names the object type"
    m = DESC_POS_RE.search(desc or "")
    if m:
        # hedge check only within the sentence that carries the assertion
        d = desc or ""
        a = max(d.rfind(".", 0, m.start()), d.rfind("|", 0, m.start())) + 1
        b = min([x for x in (d.find(".", m.end()), d.find("|", m.end())) if x >= 0] or [len(d)])
        if HEDGE_RE.search(d[a:b]):
            return False, "description only, hedged"
        return True, "description asserts the object type"
    return False, "no type evidence in record"


OPEN = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/publicdomain/mark/1.0/": "Public-Domain",
    "http://creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
}
POLICY_URL = "https://api.europeana.eu/record/v2/{id}.json (edmRights per aggregation)"
LIC_NAME = {"CC-BY-4.0": "CC BY 4.0", "CC-BY-3.0": "CC BY 3.0", "CC0-1.0": "CC0 1.0", "Public-Domain": "Public Domain Mark 1.0"}


def norm_rights(u: str) -> str:
    u = u.strip().replace("https://", "http://")
    if not u.endswith("/"):
        u += "/"
    return u


def image_rights(agg: dict, url: str) -> tuple[str, str, str | None]:
    """(effective rights URI, aggregation edm:rights, this file's webResourceEdmRights or None).
    EDM: a WebResource's own edm:rights overrides the aggregation's for that file. External copyright review
    2026-11-09 (PROGRESS E-29): Museum of Far Eastern Antiquities records say CC0 at aggregation level, but every image
    says CC BY 4.0 -> the image is CC BY 4.0. A Europeana-cached thumbnail inherits the rights of the file it was made from."""
    agg_r = norm_rights(lm(agg.get("edmRights")) or "")
    target = url
    if "api.europeana.eu/thumbnail" in url:
        target = agg.get("edmIsShownBy") or agg.get("edmObject") or url

    def same(a: str, b: str) -> bool:
        return a.replace("https://", "http://") == b.replace("https://", "http://")
    for w in agg.get("webResources") or []:
        if same(w.get("about") or "", target):
            wr = ((w.get("webResourceEdmRights") or {}).get("def") or [None])[0]
            if wr:
                return norm_rights(wr), agg_r, norm_rights(wr)
            break
    return agg_r, agg_r, None


CREDITS = RAW / "europeana" / "credits"  # provider pages fetched for photographer / copyright notices (E-30)
LIC_URL = {"CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/", "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/",
           "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/", "Public-Domain": "https://creativecommons.org/publicdomain/mark/1.0/"}


def _clean(v: str) -> str:
    v = re.sub(r'<span class="hidden">.*?</span>', "", v, flags=re.S)
    return re.sub(r"\s+", " ", __import__("html").unescape(re.sub("<[^>]+>", " ", v))).strip()


def fetch_credits() -> None:
    """Network: the provider's own pages for the credit lines Europeana does not carry (external review round 2, E-30).
    SMVK (Museum of Far Eastern Antiquities): image record -> parent 'Digital bild' record: 'Copyright - media',
    'Namn - foto (digital bild)'. Landesarchiv Baden-Württemberg: finding-aid page, 'Autor/Fotograf/Künstler' of the item.
    Raw pages -> raw/europeana/credits/ ; parsed -> raw/europeana/credits/credits.json (image_key -> fields)."""
    CREDITS.mkdir(parents=True, exist_ok=True)
    http = Http(rate_per_sec=1.0)
    rows = [json.loads(x) for x in (STAGING / "europeana.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    out, fails = {}, Counter()
    for r in rows:
        key = r["image_key"]
        safe = safe_ascii(key.replace("/", "__").replace("#", "_"), 120)
        try:
            m = re.search(r"collections\.smvk\.se/carlotta-om/web/image/zoom/(\d+)/", r["image_url"])
            if m:
                zp = CREDITS / f"{safe}_image.html"
                if not zp.exists():
                    zp.write_text(http.get(f"https://collections.smvk.se/carlotta-om/web/object/{m.group(1)}").text, encoding="utf-8")
                d = re.search(r'Belongs to Digital bild</dt><dd><a[^>]+href="([^";]+)', zp.read_text(encoding="utf-8"))
                if not d:
                    fails["smvk_no_digital_bild"] += 1
                    continue
                dp = CREDITS / f"{safe}_digital.html"
                if not dp.exists():
                    dp.write_text(http.get("https://collections.smvk.se" + d.group(1)).text, encoding="utf-8")
                pairs = {_clean(a): _clean(b) for a, b in re.findall(r"<dt[^>]*>(.*?)</dt><dd[^>]*>(.*?)</dd>", dp.read_text(encoding="utf-8"), re.S)}
                ph = (pairs.get("Namn - foto (digital bild)") or "").split(" Beskrivning")[0].strip() or None
                cr = re.sub(r"\s+\S*%\S*.*$", "", pairs.get("Copyright - media") or "").strip() or None  # drop the url-encoded duplicate
                out[key] = {"photographer": ph, "copyright": cr, "licence_note": pairs.get("Licens - media"),
                            "page": "https://collections.smvk.se" + d.group(1), "evidence": dp.relative_to(ROOT).as_posix()}
                continue
            rec = json.loads((ROOT / r["record_file"]).read_text(encoding="utf-8"))
            shown = (rec["object"]["aggregations"][0].get("edmIsShownAt") or "")
            if "landesarchiv-bw.de" in shown:
                lp = CREDITS / f"{safe}_landesarchiv.html"
                if not lp.exists():
                    lp.write_text(http.get(shown).text, encoding="utf-8")
                txt = _clean(lp.read_text(encoding="utf-8", errors="replace"))
                title = (r["title_raw"] or "").strip()
                i = txt.find(title)
                seg = txt[i:i + 400] if i >= 0 else ""
                fm = re.search(r"Autor/Fotograf/Künstler:\s*([^|]+?)\s+(?:1 Foto|\d)", seg)
                out[key] = {"photographer": fm.group(1).strip() if fm else None, "copyright": "Landesarchiv Baden-Württemberg",
                            "page": shown, "evidence": lp.relative_to(ROOT).as_posix(), "title_found": i >= 0}
        except Exception as e:  # noqa: BLE001
            fails[type(e).__name__] += 1
            print("credit fail", key, e)
    (CREDITS / "credits.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ph = sum(1 for v in out.values() if v.get("photographer"))
    wlog("Stage1·europeana", "抓取提供方原页面上的摄影者 / 版权主体（Europeana 记录不带；外部版权复核第二轮 E-30）：SMVK 数字图片记录、Landesarchiv BW 档案条目",
         "python -I scripts/collect_europeana.py --fetch-credits", len(rows),
         f"取得 {len(out)} 行，其中有摄影者 {n_ph}；失败 {dict(fails)}；原始页面存 raw/europeana/credits/")
    print("credits", len(out), "photographer", n_ph, dict(fails))


def rights_fields(rec: dict, sid: str, url: str, title: str, data_provider: str, provider: str, date: str,
                  credit: dict | None = None) -> dict | None:
    """Licence-dependent staging fields for one image; None if the image's own rights are not on the whitelist.
    credit (from fetch_credits): the provider's own photographer / copyright notice, kept verbatim in the credit line."""
    obj = rec["object"]
    prov = next((p for p in obj["proxies"] if not p.get("europeanaProxy")), obj["proxies"][0])
    eff, agg_r, img_r = image_rights(obj["aggregations"][0], url)
    if eff not in OPEN:
        return None
    lic = OPEN[eff]
    dc = lm(prov.get("dcRights"))
    layers = f"edm:rights (aggregation) = {agg_r}" + (f"; webResourceEdmRights (this image) = {img_r}" if img_r else "")
    credit = credit or {}
    who = []
    if credit.get("photographer"):
        who.append(f"Foto / photo: {credit['photographer']}")
    if credit.get("copyright") and credit["copyright"] != data_provider:
        who.append(f"© {credit['copyright']}")
    holder = credit.get("copyright") or data_provider or None
    att = (f"{title} — " + ("; ".join(who) + "; " if who else "") + f"{data_provider}"
           + (f" (via {provider})" if provider and provider != data_provider else "")
           + f", {LIC_NAME[lic]} ({LIC_URL[lic]}), https://www.europeana.eu/item/{sid}"
           + (f"; {credit['page']}" if credit.get("page") else ""))
    return dict(
        license=lic,
        rights_raw=layers + (f"; dc:rights = {dc}" if dc else "")
        + (f"; provider page: Copyright - media = {credit['copyright']}" if credit.get("copyright") else ""),
        license_evidence=license_evidence(f"https://api.europeana.eu/record/v2/{sid}.json", date,
                                          layers + ("; image-level rights take precedence (EDM)" if img_r and img_r != agg_r else ""),
                                          dc or ""),
        attribution_text=att,
        rights_holder=holder,
    )


def restage() -> None:
    """Offline (no network): recompute licence fields of staging/europeana.jsonl per image from raw/europeana/meta."""
    sp = STAGING / "europeana.jsonl"
    rows = [json.loads(x) for x in sp.read_text(encoding="utf-8").splitlines() if x.strip()]
    out, ch, dropped = [], Counter(), []
    cp = CREDITS / "credits.json"
    credits = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
    for r in rows:
        rec = json.loads((ROOT / r["record_file"]).read_text(encoding="utf-8"))
        old_date = r["license_evidence"].split(" | ")[1] if " | " in r["license_evidence"] else today()
        via = re.search(r" \(via (.+?)\), ", r["attribution_text"] or "")
        dp = r["holding_institution"] if not str(r["holding_institution"]).startswith("unknown") else ""
        new = rights_fields(rec, r["source_id"], r["image_url"], r["title_raw"], dp,
                            via.group(1) if via else "", old_date, credits.get(r["image_key"]))
        if new is None:
            dropped.append(r["image_key"])
            continue
        if new["license"] != r["license"]:
            ch[f"{r['license']} -> {new['license']} ({r['holding_institution']})"] += 1
        r.update(new)
        out.append(r)
    write_jsonl(sp, out)
    wlog("Stage1·europeana", "离线按图片层 rights 重算许可（EDM：webResourceEdmRights 优先于记录层 edm:rights；外部版权复核 E-29；不联网，读 raw/europeana/meta）",
         "python -I scripts/collect_europeana.py --restage", len(rows),
         "许可变化：" + ("; ".join(f"{k} {v}" for k, v in ch.most_common()) or "无") + f"；图片层 rights 不在白名单而移除 {len(dropped)} 行")
    print("restaged", len(rows), dict(ch), "dropped", dropped)


def lm(v) -> str | None:
    """Flatten a Europeana language map verbatim (unique values, original order)."""
    if not v:
        return None
    vals = []
    if isinstance(v, dict):
        for k in (["en", "def"] + [k for k in v if k not in ("en", "def")]):
            for x in v.get(k, []) or []:
                if x not in vals:
                    vals.append(x)
    elif isinstance(v, list):
        vals = list(dict.fromkeys(v))
    else:
        vals = [str(v)]
    vals = [x for x in vals if not str(x).startswith("http://data.europeana.eu/")]
    return " | ".join(vals) if vals else None


def search_all(http: Http, term: str, qf: str = "TYPE:IMAGE") -> tuple[int, list[dict], str]:
    params = {"wskey": KEY, "query": term, "rows": 100, "cursor": "*", "profile": "standard", "qf": qf}
    first_url = f"{API}/search.json?" + urllib.parse.urlencode({k: v for k, v in params.items() if k not in ("cursor", "wskey")})
    items, total = [], 0
    while True:
        d = http.json(f"{API}/search.json", params=params)
        total = d.get("totalResults", 0)
        items += d.get("items", []) or []
        nxt = d.get("nextCursor")
        if not nxt or not d.get("items"):
            break
        params["cursor"] = nxt
    return total, items, first_url


def main():
    setup_console()
    if "--fetch-credits" in sys.argv:  # network: provider pages with photographer / copyright notices
        return fetch_credits()
    if "--restage" in sys.argv:  # offline: per-image licence fields from raw/europeana/meta (+ credits.json)
        return restage()
    count_only = "--count-only" in sys.argv
    plan_only = "--plan" in sys.argv
    plan_file = RAW / "europeana" / "plan.json"
    http = Http(rate_per_sec=5)
    meta_dir = RAW / "europeana" / "meta"
    img_dir = RAW / "europeana" / "img"
    search_dir = RAW / "europeana" / "search"
    for d in (meta_dir, img_dir, search_dir):
        d.mkdir(parents=True, exist_ok=True)

    if plan_file.exists() and not count_only and not plan_only:
        keep = json.loads(plan_file.read_text(encoding="utf-8"))
        print("loaded plan", len(keep))
    else:
        keep = plan(http, search_dir, count_only)
        if keep is None:
            return
        plan_file.write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    if plan_only:
        for iid, it in sorted(keep.items()):
            print(iid, "|", (it.get("title") or [""])[0][:70], "|", (it.get("dataProvider") or [""])[0][:40], "|", (it.get("rights") or [""])[0])
        return
    run_download(http, keep, meta_dir, img_dir)


def plan(http, search_dir, count_only):
    # ---- 1. count / search
    union: dict[str, dict] = {}
    hits_by_term = {}
    if "--reuse-search" in sys.argv:
        files = sorted(search_dir.glob("search_*.json"))
        for f in files:
            for it in json.loads(f.read_text(encoding="utf-8"))["items"]:
                union.setdefault(it["id"], it)
        wlog("Stage1·europeana", "复用本轮已保存的检索结果（raw/europeana/search/*.json），不重新检索，只重做过滤与判定",
             "scripts/collect_europeana.py --plan --reuse-search", len(union), f"{len(files)} 个检索结果文件")
    for t, qf in ([] if "--reuse-search" in sys.argv else [(t, "TYPE:IMAGE") for t in TERMS] + PROVIDER_QUERIES):
        total, items, url = search_all(http, t, qf)
        hits_by_term[t] = total
        (search_dir / f"search_{safe_ascii(t)}_{__import__('hashlib').md5(t.encode()).hexdigest()[:8]}.json").write_text(
            json.dumps({"term": t, "url": url, "totalResults": total, "items": items}, ensure_ascii=False), encoding="utf-8")
        rights_c = Counter(norm_rights((it.get("rights") or ["?"])[0]) for it in items)
        wlog("Stage1·europeana", f"检索计数 term={t}（qf={qf}）", url + "&wskey=api2demo", total,
             f"取回 {len(items)} 条；rights 分布前 5：{rights_c.most_common(5)}")
        for it in items:
            union.setdefault(it["id"], it)
    if count_only:
        return None

    # ---- 2. broad type pre-filter on search-result text
    pre = {}
    for iid, it in union.items():
        txt = " ".join(filter(None, [" ".join(it.get("title") or []), " ".join(it.get("dcDescription") or [])]))
        if PRE_RE.search(txt):
            pre[iid] = it
    wlog("Stage1·europeana", "各检索词结果去重合并后做类型预过滤（标题/描述含屋瓦类词汇，含瑞典语 ändtegel/taktegel/takpanna）",
         "scripts/collect_europeana.py", len(union), f"类型预过滤后 {len(pre)} 条")

    # ---- 3. rights filter
    lic_ok, by_rights = {}, Counter()
    for iid, it in pre.items():
        r = norm_rights((it.get("rights") or [""])[0])
        by_rights[r] += 1
        if r in OPEN:
            lic_ok[iid] = it
        elif re.match(r"http://creativecommons.org/licenses/by/[34]\.0/[a-z]{2}/$", r):
            review(f"https://www.europeana.eu/item{iid}", f"edm:rights = {r}",
                   "CC BY 本地化移植版（ported），不在方案白名单的 CC BY 3.0/4.0 国际版之列", "建议不收；如作者认可移植版等价于 CC BY 3.0，可在明早手动加入")
    wlog("Stage1·europeana", "许可过滤：只收 CC0 / PDM / CC BY 3.0 / CC BY 4.0（国际版）",
         "scripts/collect_europeana.py", len(pre), f"许可过滤后 {len(lic_ok)} 条；预过滤后的 rights 分布：{by_rights.most_common()}")

    # ---- 4. final type decision on full record text
    meta_dir = RAW / "europeana" / "meta"
    keep, reasons = {}, Counter()
    decisions = []
    for iid, it in sorted(lic_ok.items()):
        sid = iid.strip("/")
        safe = safe_ascii(sid.replace("/", "__"), 120)
        mf = meta_dir / f"{safe}.json"
        if mf.exists():
            rec = json.loads(mf.read_text(encoding="utf-8"))
        else:
            try:
                rec = http.json(f"{API}/{sid}.json", params={"wskey": KEY})
            except Exception as e:  # noqa: BLE001
                reasons["record_fetch_failed"] += 1
                print("record fail", sid, e)
                continue
            mf.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        prov = next((p for p in rec["object"]["proxies"] if not p.get("europeanaProxy")), rec["object"]["proxies"][0])
        ok, why = decide(lm(prov.get("dcTitle")) or "", lm(prov.get("dcType")) or "", lm(prov.get("dcDescription")) or "",
                         lm(prov.get("dcSubject")) or "")
        reasons[why] += 1
        decisions.append({"id": iid, "title": lm(prov.get("dcTitle")), "keep": ok, "why": why})
        if ok:
            keep[iid] = it
    (RAW / "europeana" / "type_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=1), encoding="utf-8")
    wlog("Stage1·europeana", "按完整记录文本做最终类型判定（标题/类型字段强匹配，或描述明确断言；排除刀镡、根付、陶俑、屋脊骑士/僧人俑、有保留措辞者）",
         "scripts/collect_europeana.py → raw/europeana/type_decisions.json", len(lic_ok), f"保留 {len(keep)} 条；理由分布 {dict(reasons)}")
    return keep


def run_download(http, keep, meta_dir, img_dir):

    # ---- 4. evidence
    ev_file = EVIDENCE / "europeana_policy_2026-11-08_api_rights.txt"
    ev_quote = ("Europeana API per-item edm:rights (aggregation) — e.g. IIIF manifest attribution: "
                "'antefix - https://www.europeana.eu/item/2021606/resource_document_rijksmuseum_van_oudheden_RG10. "
                "Rijksmuseum van Oudheden - https://hdl.handle.net/21.12126/60115. CC BY - http://creativecommons.org/licenses/by/4.0/'")
    if not ev_file.exists():
        ev_file.write_text(
            "Europeana rights evidence, fetched 2026-11-08 (UTC)\n"
            "www.europeana.eu/en/rights and pro.europeana.eu rights pages returned HTTP 403 (bot protection) to scripted requests,\n"
            "so the per-item rights statement returned by the Europeana Record API (edm:rights of the provider aggregation)\n"
            "is used as evidence for every item; each item's full API record is stored under raw/europeana/meta/.\n\n"
            + ev_quote + "\n", encoding="utf-8")

    # ---- 5. records + images
    rows, fails = [], Counter()
    prev = {r["image_key"]: r for r in read_jsonl(STAGING / "europeana.jsonl")}
    for n, (iid, it) in enumerate(sorted(keep.items()), 1):
        sid = iid.strip("/")
        safe = safe_ascii(sid.replace("/", "__"), 120)
        key = f"{sid}#0"
        if key in prev and (ROOT / prev[key]["raw_file"]).exists():
            rows.extend(r for r in prev.values() if r["source_id"] == sid)
            continue
        try:
            rec = http.json(f"{API}/{sid}.json", params={"wskey": KEY})
        except Exception as e:  # noqa: BLE001
            fails["record_fetch"] += 1
            print("record fail", sid, e)
            continue
        (meta_dir / f"{safe}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        obj = rec["object"]
        prov = next((p for p in obj["proxies"] if not p.get("europeanaProxy")), obj["proxies"][0])
        agg = obj["aggregations"][0]
        rights = norm_rights(lm(agg.get("edmRights")) or "")
        if rights not in OPEN:
            fails["rights_changed_in_record"] += 1
            continue
        lic = OPEN[rights]
        orgs = {o.get("about"): lm(o.get("prefLabel")) for o in obj.get("organizations", [])}
        dp_raw = lm(agg.get("edmDataProvider")) or ""
        dp_uri = (agg.get("edmDataProvider") or {}).get("def", [""])[0]
        data_provider = orgs.get(dp_uri) or (it.get("dataProvider") or [dp_raw])[0]
        data_provider = (data_provider or "").split(" | ")[0]
        provider = (it.get("provider") or [""])[0]
        title = lm(prov.get("dcTitle")) or " | ".join(it.get("title") or [])
        # images: primary (edmIsShownBy / edmObject) + additional views (hasView)
        def fix(u):
            return u.replace("http://img.rmo.nl/", "https://img.rmo.nl/")
        primary_cands = [fix(u) for u in dict.fromkeys([agg.get("edmIsShownBy"), agg.get("edmObject")]) if u]
        view_urls = [fix(u) for u in (agg.get("hasView") or []) if u and fix(u) not in primary_cands]
        images = []  # (index, path, sha, nbytes, url, note)
        notes = []
        got = None
        for u in primary_cands:
            try:
                p, sha, nb = download_image(http, u, img_dir / f"{safe}_0")
                got = (0, p, sha, nb, u, None)
                break
            except Exception as e:  # noqa: BLE001
                notes.append(f"原图源失败 {u}: {str(e)[:80]}")
        if got is None and primary_cands:
            tu = "https://api.europeana.eu/thumbnail/v3/400/" + __import__("hashlib").md5(
                (agg.get("edmIsShownBy") or agg.get("edmObject")).encode()).hexdigest()
            try:
                p, sha, nb = download_image(http, tu, img_dir / f"{safe}_0", min_bytes=2000)
                got = (0, p, sha, nb, tu, "原图源不可达，使用 Europeana 缓存缩略图（400 px 宽），分辨率低")
            except Exception as e:  # noqa: BLE001
                notes.append(f"缩略图也失败: {str(e)[:80]}")
        if got:
            images.append(got)
        for u in view_urls:
            k = len(images) if images else 1
            try:
                p, sha, nb = download_image(http, u, img_dir / f"{safe}_{k}")
                images.append((k, p, sha, nb, u, "hasView 附加视图"))
            except Exception as e:  # noqa: BLE001
                notes.append(f"附加视图失败 {u}: {str(e)[:60]}")
        if not images:
            fails["image_download"] += 1
            print("image fail", sid, notes)
            continue
        desc = lm(prov.get("dcDescription"))
        date_raw = lm(prov.get("dctermsCreated")) or lm(prov.get("dcDate"))
        region = lm(prov.get("dctermsSpatial")) or lm(prov.get("dcCoverage"))
        tt = f"{title} | {lm(prov.get('dcType')) or ''}"
        is_rep = bool(re.search(r"\b(replica|reproduction|copy|cast|kopie|abguss|nachbildung|gipsabguss|replik)\b", tt, re.I)
                      or re.search(r"\b(is|är|ist) (a|an|en|ett|eine?) (modern )?(copy|replica|cast|kopia|replik|abguss)\b", desc or "", re.I))

        if data_provider and any(k in data_provider.lower() for k in ["metropolitan", "smithsonian", "cleveland", "art institute of chicago", "getty", "walters"]):
            notes.append(f"dataProvider={data_provider}：与直接采集的馆可能重复，留给去重环节判")
        if agg.get("edmIsShownAt"):
            notes.append(f"provider page: {agg.get('edmIsShownAt')}")
        modality = "drawing" if re.search(r"Architekturmuseum|Grammatik der Ornamente|Zeichnung|drawing", f"{title} {desc} {data_provider}", re.I) else None
        for (idx, p, sha, nb, used, inote) in images:
            rf = rights_fields(rec, sid, used, title, data_provider, provider, today())
            if rf is None:
                fails["image_rights_not_open"] += 1
                continue
            w, h = image_size(p)
            rows.append(stage_row(
                source="europeana", source_id=sid, image_key=f"{sid}#{idx}", image_index=idx,
                source_url=f"https://www.europeana.eu/item/{sid}", image_url=used,
                iiif_manifest=f"https://iiif.europeana.eu/presentation/{sid}/manifest",
                record_file=f"raw/europeana/meta/{safe}.json",
                title_raw=title, object_name_raw=lm(prov.get("dcType")), description_raw=desc,
                classification_raw=lm(prov.get("dcFormat")), culture_raw=None, period_raw=None, date_raw=date_raw,
                region_raw=region, medium_raw=lm(prov.get("dctermsMedium")), dimensions_raw=lm(prov.get("dctermsExtent")),
                credit_line_raw=lm(prov.get("dcRights")), subjects_raw=lm(prov.get("dcSubject")),
                holding_institution=data_provider or "unknown (see record)", accession_number=lm(prov.get("dcIdentifier")),
                **rf,
                modality_hint=modality, is_reproduction=is_rep,
                raw_file=p.relative_to(ROOT).as_posix(), sha256=sha, bytes=nb, width=w, height=h,
                date_accessed=today(), notes="; ".join([x for x in [inote] + notes if x]) or None,
            ))
        if n % 20 == 0:
            print(n, "/", len(keep))
    write_jsonl(STAGING / "europeana.jsonl", rows)
    low = sum(1 for r in rows if r["notes"] and "缩略图" in r["notes"])
    n_obj = len({r["source_id"] for r in rows})
    wlog("Stage1·europeana", "逐条取完整记录并下载图像", "scripts/collect_europeana.py", len(keep),
         f"下载成功 {n_obj} 件 / {len(rows)} 张图；失败 {dict(fails)}；其中 {low} 张因原图源失效改用 Europeana 400px 缓存缩略图；输出 staging/europeana.jsonl")
    print("done", len(rows), dict(fails))


if __name__ == "__main__":
    main()
