"""Local page for the author to check the rectified images (PROGRESS E-55). Not published.
Each card: standard crop | rectified crop (rim outline drawn on both), tilt, stretch, outline roundness before -> after.
Cards whose outline got flatter after rectification are framed red. Sort and filter in the page.
Output: qa/rect_review.html (+ qa/rect_review_img/ with the small side-by-side images). Usage: python -I scripts/make_rect_review.py
"""
from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS, ROOT, read_jsonl, setup_console  # noqa: E402

OUT = ROOT / "qa" / "rect_review.html"
IMG = ROOT / "qa" / "rect_review_img"
S = 300


def panel(img_p: Path, mask_p: Path) -> np.ndarray:
    im = cv2.imdecode(np.fromfile(str(img_p), np.uint8), cv2.IMREAD_COLOR)
    mk = cv2.imdecode(np.fromfile(str(mask_p), np.uint8), cv2.IMREAD_GRAYSCALE)
    cs, _ = cv2.findContours((mk == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cs, -1, (0, 200, 0), 6)
    return cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)


def main():
    setup_console()
    R = {r["image_id"]: r for r in read_jsonl(ROOT / "metadata.jsonl")}
    X = [x for x in csv.DictReader(open(REPORTS / "rect_index.csv", encoding="utf-8-sig")) if x["image_id"] in R and R[x["image_id"]].get("rect_file")]
    IMG.mkdir(parents=True, exist_ok=True)
    data = []
    for x in X:
        i, r = x["image_id"], R[x["image_id"]]
        p = IMG / f"{i}.jpg"
        if not p.exists():
            a = panel(ROOT / r["crop_file"], ROOT / r["mask_file"])
            b = panel(ROOT / r["rect_file"], ROOT / r["rect_mask_file"])
            pair = np.hstack([a, np.full((S, 6, 3), 255, np.uint8), b])
            cv2.imencode(".jpg", pair, [cv2.IMWRITE_JPEG_QUALITY, 84])[1].tofile(str(p))
        before, after = float(x["mask_axis_ratio_before"] or 0), float(x["mask_axis_ratio_after"] or 0)
        data.append({"id": i, "tilt": round(float(x["tilt_deg_ellipse"]), 1), "k": round(float(x["stretch"]), 3), "edge": x["rim_edge_used"],
                     "b": before, "a": after, "worse": after < before - 0.02, "front": bool(r.get("frontal")), "src": r["source"]})
    n_worse = sum(d["worse"] for d in data)
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>拍正图检查（{len(data)} 张）</title>
<style>body{{font-family:"Microsoft YaHei",sans-serif;margin:0;background:#f6f5f2}}
.top{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:8px 14px;z-index:9;display:flex;gap:12px;flex-wrap:wrap;align-items:center}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(620px,1fr));gap:10px;padding:10px 14px}}
.c{{background:#fff;border:2px solid #ddd;padding:6px}}.c.worse{{border-color:#d22}}.c img{{width:100%;display:block}}
.t{{font-size:13px;color:#333;margin-top:4px}}.m{{color:#888}}.w{{color:#d22;font-weight:bold}}select,button{{font:inherit}}</style></head><body>
<div class="top"><b>拍正图检查：{len(data)} 张</b><span class="m">每张左为标准裁剪图、右为拍正图，绿线是瓦当轮廓。红框 = 拍正后外廓反而更扁（{n_worse} 张）。</span>
<label>排序 <select id="sort" onchange="draw()"><option value="k">拉伸从大到小</option><option value="tilt">倾角从大到小</option><option value="d">外廓变化（变差的在前）</option><option value="id">图号</option></select></label>
<label>只看 <select id="only" onchange="draw()"><option value="">全部</option><option value="worse">变扁的</option><option value="front">正面</option><option value="big">拉伸 ≥ 1.15</option></select></label>
<span id="n" class="m"></span></div>
<main id="g"></main>
<script>
var D={json.dumps(data, ensure_ascii=False)};
function draw(){{var s=document.getElementById('sort').value,o=document.getElementById('only').value;
 var L=D.filter(function(d){{return !o||(o==='worse'&&d.worse)||(o==='front'&&d.front)||(o==='big'&&d.k>=1.15)}});
 L.sort(function(x,y){{return s==='k'?y.k-x.k:s==='tilt'?y.tilt-x.tilt:s==='d'?(x.a-x.b)-(y.a-y.b):(x.id<y.id?-1:1)}});
 document.getElementById('n').textContent='显示 '+L.length+' 张';
 document.getElementById('g').innerHTML=L.map(function(d){{return '<div class="c'+(d.worse?' worse':'')+'"><a href="../'+'rect/'+d.id+'_crop.jpg" target="_blank"><img loading="lazy" src="rect_review_img/'+d.id+'.jpg"></a>'+
  '<div class="t"><b>'+d.id+'</b> · '+(d.front?'正面':'非正面')+' · 倾角约 '+d.tilt+'° · 拉伸 ×'+d.k+' · 用'+(d.edge==='inner'?'内缘':'外缘')+
  ' · 外廓圆度 '+d.b.toFixed(3)+' → <span class="'+(d.worse?'w':'')+'">'+d.a.toFixed(3)+'</span></div></div>'}}).join('')}}
draw();
</script></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(OUT, len(data), "pairs,", n_worse, "flatter after rectification")


if __name__ == "__main__":
    main()
