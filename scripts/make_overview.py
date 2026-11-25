"""One-page visual overview of the finished dataset (public images only; the author's photos are never used).
Rows = region groups; each cell = standardised crop with the segmentation contour, caption = motif / period / holder / licence.
Output: qa/overview.jpg (local review material, not committed: contains third-party images)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from common import REPORTS, ROOT, read_jsonl, setup_console  # noqa: E402

setup_console()
import re as _re
_FC = {}


def F(size, text=""):
    face = "C:/Windows/Fonts/malgun.ttf" if _re.search("[가-힣]", text or "") else "C:/Windows/Fonts/msyh.ttc"
    if (face, size) not in _FC:
        _FC[(face, size)] = ImageFont.truetype(face, size)
    return _FC[(face, size)]
recs = read_jsonl(ROOT / "metadata.jsonl")
vocab = {r["code"]: r["zh"] for r in csv.DictReader(open(ROOT / "vocab" / "motif.csv", encoding="utf-8"))}
per = {r["code"]: r["zh"] for r in csv.DictReader(open(ROOT / "vocab" / "period.csv", encoding="utf-8"))}
lic_zh = {"CC0-1.0": "CC0", "Public-Domain": "公有领域", "CC-BY-4.0": "CC BY 4.0", "CC-BY-3.0": "CC BY 3.0",
          "ColBase-Terms": "ColBase 规约", "KOGL-Type-1": "KOGL 1型"}
# views judged 'back' by both models are skipped for the showcase (selection only; labels untouched)
back = defaultdict(int)
for p in (REPORTS / "vlm").glob("*.jsonl"):
    for x in read_jsonl(p):
        if (x.get("result") or {}).get("view") in ("back", "side"):
            back[x["image_id"]] += 1
q = {r["image_id"]: r for r in csv.DictReader(open(REPORTS / "quality.csv", encoding="utf-8"))}


suspect = set(json.load(open(REPORTS / "std_suspect_masks.json"))) if (REPORTS / "std_suspect_masks.json").exists() else set()


def good(r):
    return (r["image_id"] not in suspect and r["crop_file"] and back[r["image_id"]] == 0 and not r["quality_flags"] and r["motif"] != "unknown"
            and r["view"] in (None, "front") and float(q[r["image_id"]]["lap_var"]) > 60)


GROUPS = [("china", "中国"), ("korea", "朝鲜半岛"), ("japan", "日本"), ("mediterranean", "地中海"), ("other", "其他地区")]
ORDER = ["human_face", "beast_face", "demon_face", "gorgoneion", "silenus", "maenad", "four_spirits", "text", "lotus",
         "cloud", "animal", "palmette", "geometric", "other"]
N, CELL = 8, 270
rows = []
for g, gz in GROUPS:
    cand = [r for r in recs if r["region_group"] == g and good(r)]
    by = defaultdict(list)
    for r in cand:
        by[r["motif"]].append(r)
    pick, used_obj = [], set()
    while len(pick) < N and any(by.values()):
        for m in ORDER:  # round-robin over motifs -> diversity
            lst = [r for r in by.get(m, []) if r["object_id"] not in used_obj]
            if lst and len(pick) < N:
                lst.sort(key=lambda r: (r["tilt_deg"] is None, r["tilt_deg"] or 0, r["image_id"]))
                r = lst[len(pick) % max(1, min(3, len(lst)))]
                pick.append(r)
                used_obj.add(r["object_id"])
                by[m] = [x for x in by[m] if x["object_id"] != r["object_id"]]
        if not any(by.values()):
            break
    rows.append((gz, sum(1 for r in recs if r["region_group"] == g), pick))

W = 40 + 150 + N * (CELL + 14)
H = 230 + len(rows) * (CELL + 96)
canvas = Image.new("RGB", (W, H), (250, 248, 244))
d = ImageDraw.Draw(canvas)
n_img, n_obj = len(recs), len({r["object_id"] for r in recs})
d.text((40, 28), "瓦当开放图像数据集 v0.1.0 · 成品一览", font=F(40), fill=(40, 30, 20))
d.text((40, 88), f"{n_img} 张图 / {n_obj} 件器物 / 10 个开放来源 / 60 家收藏机构 · 每张图都有馆方原文、许可证据与中英文引文 · 标签全部待人工复核",
       font=F(22), fill=(90, 80, 70))
d.text((40, 124), "每格：标准化裁剪（1024 px）+ 自动分割轮廓（绿）；下方依次为母题 / 时代 / 收藏机构 / 许可。只展示公开来源的图像。",
       font=F(19), fill=(120, 110, 100))
y = 180
for gz, total, pick in rows:
    d.text((40, y + CELL // 2 - 30), gz, font=F(30), fill=(60, 40, 20))
    d.text((40, y + CELL // 2 + 12), f"{total} 张", font=F(20), fill=(120, 110, 100))
    x = 190
    for r in pick:
        im = cv2.imdecode(np.fromfile(str(ROOT / r["crop_file"]), np.uint8), cv2.IMREAD_COLOR)
        mk = cv2.imdecode(np.fromfile(str(ROOT / r["mask_file"]), np.uint8), cv2.IMREAD_GRAYSCALE)
        cs, _ = cv2.findContours((mk == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (60, 220, 60), 5)
        if (mk == 128).any():
            im[mk == 128] = (0.5 * im[mk == 128] + [0, 0, 127]).astype(np.uint8)
        tile = Image.fromarray(cv2.cvtColor(cv2.resize(im, (CELL, CELL), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        canvas.paste(tile, (x, y))
        inst = (r["holding_institution"] or "").split(" / ")[0]
        inst = inst if len(inst) <= 16 else inst[:15] + "…"
        d.text((x, y + CELL + 4), f"{vocab.get(r['motif'], r['motif'])} · {per.get(r['period_code'], '时代未写') if r['period_code'] else '时代未写'}",
               font=F(17), fill=(40, 30, 20))
        d.text((x, y + CELL + 28), inst, font=F(15, inst), fill=(100, 90, 80))
        d.text((x, y + CELL + 50), f"{lic_zh[r['license']]} · {r['image_id']}", font=F(15), fill=(130, 120, 110))
        x += CELL + 14
    y += CELL + 96
out = ROOT / "qa" / "overview.jpg"
canvas.save(out, quality=88)
print(out, canvas.size, [(g, len(p)) for g, _, p in rows])
