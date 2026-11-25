"""Stage 2: dedup candidates. pHash (Hamming <= 8) + DINOv2 cosine (top 1 % of cross-object pairs).

Tonight's automatic actions are limited to (方案 Stage 2): identical sha256 merged (build_metadata) and same
source+source_id -> same object_id (build_metadata). Everything else is ONLY listed for the author.

Usage: python -I scripts/dedup.py [--emb reports/dinov2_emb.npz]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imagehash  # noqa: E402

from common import REPORTS, ROOT, read_jsonl, setup_console, wlog  # noqa: E402

Image.MAX_IMAGE_PIXELS = 400_000_000


def small(path: Path, side=512) -> Image.Image:
    im = ImageOps.exif_transpose(Image.open(path))
    im.draft("RGB", (side, side))
    im = im.convert("RGB")
    im.thumbnail((side, side))
    return im


def pair_thumb(a: dict, b: dict, out: Path, label: str):
    W = 300
    canvas = Image.new("RGB", (2 * W + 10, W + 40), "white")
    d = ImageDraw.Draw(canvas)
    for k, r in enumerate((a, b)):
        im = small(ROOT / r["file_name"], W)
        canvas.paste(im, (k * (W + 10) + (W - im.width) // 2, 36 + (W - im.height) // 2))
        d.text((k * (W + 10) + 3, 3), f"{r['image_id']} {r['source']}", fill="black")
        d.text((k * (W + 10) + 3, 17), f"{r['object_id']}", fill="gray")
    d.text((2 * W - 60, 3), label, fill="red")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=82)


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(REPORTS / "dinov2_emb.npz"))
    a = ap.parse_args()
    recs = read_jsonl(ROOT / "metadata.jsonl")
    by = {r["image_id"]: r for r in recs}
    # ---- pHash
    ph_path = REPORTS / "phash.csv"
    cache = {}
    if ph_path.exists():
        cache = {row["image_id"]: row for row in csv.DictReader(open(ph_path, encoding="utf-8"))}
    rows = []
    for r in recs:
        c = cache.get(r["image_id"])
        if c and c["sha256"] == r["sha256"]:
            rows.append(c)
            continue
        h = imagehash.phash(small(ROOT / r["file_name"]))
        rows.append({"image_id": r["image_id"], "sha256": r["sha256"], "phash": str(h)})
    with open(ph_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "sha256", "phash"])
        w.writeheader()
        w.writerows(rows)
    ids = [x["image_id"] for x in rows]
    H = np.array([[int(b) for b in bin(int(x["phash"], 16))[2:].zfill(64)] for x in rows], np.uint8)
    obj = np.array([by[i]["object_id"] for i in ids])
    cand = {}
    n = len(ids)
    for i in range(n):
        d = (H[i + 1:] != H[i]).sum(1)
        for j in np.nonzero(d <= 8)[0] + i + 1:
            if obj[i] != obj[j]:
                cand[(ids[i], ids[j])] = {"phash_dist": int(d[j - i - 1])}
    n_ph = len(cand)
    # ---- DINOv2 cosine
    thr = None
    n_cos = 0
    if Path(a.emb).exists():
        z = np.load(a.emb)
        eids = list(z["ids"])
        E = z["emb"]
        pos = {k: v for v, k in enumerate(eids)}
        keep = [i for i in ids if i in pos]
        X = E[[pos[i] for i in keep]]
        S = X @ X.T
        o = np.array([by[i]["object_id"] for i in keep])
        iu = np.triu_indices(len(keep), 1)
        cross = o[iu[0]] != o[iu[1]]
        sims = S[iu][cross]
        thr = float(np.percentile(sims, 99)) if len(sims) else 1.0
        sel = np.nonzero(cross & (S[iu] >= thr))[0]
        n_cos = len(sel)
        # full top-1 % list (not reviewable by hand: tens of thousands of pairs) -> CSV without thumbnails
        import gzip
        with gzip.open(REPORTS / "dedup_cos_top1pct.csv.gz", "wt", encoding="utf-8", newline="") as gz:
            gw = csv.writer(gz)
            gw.writerow(["image_id_a", "image_id_b", "cos_sim"])
            for k in sel:
                gw.writerow([keep[iu[0][k]], keep[iu[1][k]], round(float(S[iu][k]), 4)])
        # review queue (PROGRESS E-23): top-1 % pairs that are mutual nearest neighbours across objects, plus cos >= 0.95
        S2 = S.copy()
        np.fill_diagonal(S2, -1)
        S2[o[:, None] == o[None, :]] = -1
        nn = S2.argmax(1)
        for i in range(len(keep)):
            j = int(nn[i])
            if (nn[j] == i and i < j and S[i, j] >= thr) or S2[i, j] >= 0.95:
                p = (keep[min(i, j)], keep[max(i, j)])
                cand.setdefault(p, {})["cos_sim"] = round(float(S[i, j]), 4)
        for i, j in zip(*np.nonzero(np.triu(S2 >= 0.95, 1))):
            cand.setdefault((keep[i], keep[j]), {})["cos_sim"] = round(float(S[i, j]), 4)
        for (x, y), v in cand.items():
            if "cos_sim" not in v and x in pos and y in pos:
                v["cos_sim"] = round(float(E[pos[x]] @ E[pos[y]]), 4)
    # ---- write queue with pair thumbnails (sorted: most similar first)
    out = []
    for (x, y), v in sorted(cand.items(), key=lambda kv: (-(kv[1].get("cos_sim") or 0), kv[1].get("phash_dist", 99))):
        A, B = by[x], by[y]
        reason = ";".join(k for k in ("phash_dist", "cos_sim") if k in v and (k != "cos_sim" or (thr and v[k] >= thr)))
        pid = f"pair_{len(out) + 1:05d}"
        thumb = Path("qa") / "dedup" / f"{pid}.jpg"
        out.append({"pair_id": pid, "image_id_a": x, "image_id_b": y, "object_id_a": A["object_id"], "object_id_b": B["object_id"],
                    "source_a": A["source"], "source_b": B["source"], "phash_dist": v.get("phash_dist", ""),
                    "cos_sim": v.get("cos_sim", ""), "reason": reason, "title_a": A["title_raw"], "title_b": B["title_raw"],
                    "url_a": A["source_url"], "url_b": B["source_url"], "thumb": thumb.as_posix(),
                    "decision": "", "decision_note": ""})
    for row in out:
        pair_thumb(by[row["image_id_a"]], by[row["image_id_b"]], ROOT / row["thumb"],
                   f"pH {row['phash_dist']} cos {row['cos_sim']}")
    with open(REPORTS / "dedup_candidates.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) if out else ["pair_id"])
        w.writeheader()
        w.writerows(out)
    wlog("Stage2", "去重候选：pHash（64 位，汉明距离 ≤ 8）+ DINOv2 ViT-L 余弦相似度（不同 object_id 之间的配对取前 1%）；只列队列，不合并",
         "python -I scripts/dedup.py", f"{n} 张图",
         f"pHash 候选 {n_ph} 对；余弦前 1% 阈值 = {thr if thr is None else round(thr, 4)}，前 1% 共 {n_cos} 对（全表 reports/dedup_cos_top1pct.csv.gz，无缩略图）；"
         f"复核队列 = pHash 候选 ∪ 前 1% 中互为最近邻 ∪ cos≥0.95，共 {len(out)} 对（reports/dedup_candidates.csv，每行有缩略图 qa/dedup/）")
    print(len(out), n_ph, n_cos, thr)


if __name__ == "__main__":
    main()
