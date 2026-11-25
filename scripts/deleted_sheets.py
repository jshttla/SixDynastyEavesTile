"""Contact sheets of the images the eye review deleted, one per kind (author asked to see them). Public images only.
Output: qa/review_v3/deleted_<kind>.jpg (local, not committed). Usage: python -I scripts/deleted_sheets.py
"""
import csv
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent.parent
RV = ROOT / "reports" / "review_v3"
WC = ROOT / "cloud" / "work" / "upload" / "images"
OUT = ROOT / "qa" / "review_v3"
ZH = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 13)
KO = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 13)
FB = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 22)
KINDS = [("xuanpingwa", "軒平瓦类（滴水 / 암막새，平瓦的檐头）", r"軒平|滴水|암막|平瓦|板瓦"),
         ("guiwa", "鬼瓦类（屋脊两端的大块装饰瓦，含韩国 귀면와 鬼面瓦）", r"鬼瓦|鬼面|귀면"),
         ("chiwei", "鸱尾类（屋脊两端的鱼尾形大构件）", r"鸱|鴟|蚩"),
         ("jishi", "脊饰类（脊兽、骑马人物、屋脊上的装饰）", r"脊|兽|獸|骑|騎|立人|屋顶|屋脊"),
         ("taosu", "陶塑 / 雕刻", r"陶塑|人物|俑|头像|頭像|雕"),
         ("other", "其他（垂木先瓦、普通筒瓦 / 板瓦残片、隅木蓋瓦等）", None)]


KOB = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 22)


def text(d, xy, t, fill, big=False):
    """Mixed Chinese / Hangul text: Hangul glyphs from Malgun, everything else from YaHei (one font lacks the other)."""
    x, y = xy
    for ch in t:
        f = (KOB if big else KO) if "\uac00" <= ch <= "\ud7a3" else (FB if big else ZH)
        d.text((x, y), ch, fill=fill, font=f)
        x += d.textlength(ch, font=f)


def sheet(title, items, path, cols=5, C=250):
    rows = (len(items) + cols - 1) // cols
    sh = Image.new("RGB", (cols * (C + 10) + 10, 50 + rows * (C + 52)), "white")
    d = ImageDraw.Draw(sh)
    text(d, (12, 10), title, "black", big=True)
    for k, (iid, l1, l2) in enumerate(items):
        x, y = 10 + (k % cols) * (C + 10), 50 + (k // cols) * (C + 52)
        im = ImageOps.contain(Image.open(WC / f"{iid}.jpg").convert("RGB"), (C, C))
        sh.paste(im, (x + (C - im.width) // 2, y + (C - im.height) // 2))
        text(d, (x, y + C + 2), f"{iid}  {l1}"[:30], "black")
        text(d, (x, y + C + 20), l2[:20], "#b00000")
    sh.save(path, quality=86)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    first = {}
    for p in RV.glob("out_*.tsv"):
        for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
            first[r["image_id"]] = r
    dec = list(csv.DictReader(open(RV / "decisions.csv", encoding="utf-8-sig")))
    title = {r["image_id"]: (r["title_raw"] or "") for r in dec}
    src = {r["image_id"]: r["source"] for r in dec}
    pools = {k: [] for k, _, _ in KINDS}
    for r in dec:
        if r["category"] != "not_wadang":
            continue
        w = first[r["image_id"]]["what"]
        k = next((k for k, _, rx in KINDS if rx and re.search(rx, w)), "other")
        pools[k].append(r["image_id"])
    rnd = random.Random(20261109)
    for k, label, _ in KINDS:
        ids = sorted(pools[k])
        pick = sorted(rnd.sample(ids, min(25, len(ids))))
        items = [(i, f"{src[i]} {title[i]}", "复核：" + first[i]["what"]) for i in pick]
        sheet(f"已删除 · {label} · 共 {len(ids)} 张，随机显示 {len(pick)} 张", items, OUT / f"deleted_{k}.jpg")
        print(k, len(ids))
    nv = sorted(r["image_id"] for r in dec if r["category"] == "no_research_value")
    items = [(i, f"{src[i]} {title[i]}", "复核：" + first[i]["value_note"]) for i in nv]
    sheet(f"已删除 · 没有研究价值 · 全部 {len(nv)} 张", items, OUT / "deleted_no_value.jpg")
    print("no_value", len(nv))
    json.dump(pools, open(OUT / "deleted_pools.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
