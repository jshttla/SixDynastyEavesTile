"""Stage 7 developer entry: Hugging Face Parquet shards (<= ~450 MB each), images embedded as datasets.Image.

configs (README YAML):
  default : one row per image (original bytes) + the published fields (common.PUBLIC_FIELDS, PROGRESS E-38)
  std     : standardised crop (1024 px) + 3-value mask, for images that have them (mask_wrong_review excluded)
metadata.jsonl / metadata.csv on the Hub are written here from the same projection (never the internal full record).
Usage: python -I scripts/build_hf.py  -> dist/hf/data/<config>/train-XXXXX-of-XXXXX.parquet
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from common import PUBLIC_FIELDS, ROOT, public, read_jsonl, setup_console, wlog  # noqa: E402

OUT = ROOT / "dist" / "hf"
SHARD_BYTES = 450 * 1024 * 1024
META_COLS = None  # filled from the first record


def meta_value(v):
    if isinstance(v, list):
        return v
    return v


def schema(extra_images: list[str]) -> pa.Schema:
    img_t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    fields = [pa.field(n, img_t) for n in extra_images]
    for k in META_COLS:
        if k == "quality_flags":
            fields.append(pa.field(k, pa.list_(pa.string())))
        elif k in ("date_from", "date_to", "object_count", "width", "height"):
            fields.append(pa.field(k, pa.int64()))
        elif k in ("diameter_cm", "completeness_ratio", "rim_arc_deg", "tilt_deg"):
            fields.append(pa.field(k, pa.float64()))
        elif k in ("has_face", "is_reproduction", "attribution_required", "frontal"):
            fields.append(pa.field(k, pa.bool_()))
        else:
            fields.append(pa.field(k, pa.string()))
    return pa.schema(fields)


def features_meta(extra_images: list[str]) -> dict:
    """HF `datasets` feature metadata so the viewer shows images."""
    feats = {n: {"_type": "Image"} for n in extra_images}
    for f in schema([]):
        if pa.types.is_list(f.type):
            feats[f.name] = {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"}
        else:
            dt = {pa.int64(): "int64", pa.float64(): "float64", pa.bool_(): "bool"}.get(f.type, "string")
            feats[f.name] = {"dtype": dt, "_type": "Value"}
    return {"info": {"features": feats}}


def coerce(r: dict) -> dict:
    out = {}
    r = public(r)
    for k in META_COLS:
        v = r.get(k)
        if k in ("date_from", "date_to", "object_count", "width", "height") and v is not None:
            v = int(v)
        elif k in ("diameter_cm", "completeness_ratio", "rim_arc_deg", "tilt_deg") and v is not None:
            v = float(v)
        elif k not in ("quality_flags", "has_face", "is_reproduction", "attribution_required", "frontal") and v is not None:
            v = str(v)
        out[k] = v
    return out


def write_config(name: str, recs: list[dict], images: dict) -> tuple[int, int]:
    """images: column -> function(rec) -> (path_on_disk, name_in_parquet)"""
    d = OUT / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    sch = schema(list(images)).with_metadata({"huggingface": json.dumps(features_meta(list(images)))})
    shards, cur, cur_bytes = [], [], 0
    for r in recs:
        size = sum(Path(f(r)[0]).stat().st_size for f in images.values())
        if cur and cur_bytes + size > SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(r)
        cur_bytes += size
    if cur:
        shards.append(cur)
    total = 0
    for i, chunk in enumerate(shards):
        rows = []
        for r in chunk:
            row = coerce(r)
            for col, f in images.items():
                p, nm = f(r)
                row[col] = {"bytes": Path(p).read_bytes(), "path": nm}
            rows.append(row)
        tbl = pa.Table.from_pylist(rows, schema=sch)
        fp = d / f"train-{i:05d}-of-{len(shards):05d}.parquet"
        pq.write_table(tbl, fp, compression="zstd", row_group_size=64)
        total += fp.stat().st_size
    return len(shards), total


def sync_docs() -> list[str]:
    """dist/hf/ is the complete upload folder: the dataset card plus every file it links to (external copyright recheck
    2026-11-09: a stale README of 11-08 was left here, with a licence_link to a file that was not in the folder).
    Copies the current root files, then checks that every relative link of the card resolves inside dist/hf/."""
    import re
    keep = set(HF_DOCS) | {dst for _s, dst in HF_LINKED} | set(HF_DIRS) | {"data", "std", "rect", "metadata.jsonl", "metadata.csv",
                                                                            "files.csv", ".gitattributes"}
    for q in list(OUT.iterdir()):  # anything else left over from earlier releases goes
        if q.name not in keep:
            shutil.rmtree(q) if q.is_dir() else q.unlink()
    for f in HF_DOCS:
        if (ROOT / f).exists():
            shutil.copy2(ROOT / f, OUT / f)
    write_public_metadata(read_jsonl(ROOT / "metadata.jsonl"))
    if (ROOT / "site" / "files.csv").exists():
        shutil.copy2(ROOT / "site" / "files.csv", OUT / "files.csv")
    for src, dst in HF_LINKED:  # large files: hard link (same disk), no second 3.4 GB copy
        q = OUT / dst
        q.unlink(missing_ok=True)
        try:
            import os
            os.link(ROOT / src, q)
        except OSError:
            shutil.copy2(ROOT / src, q)
    for d in ("std", "rect"):  # segmentation results and rectified pairs as plain files: the paths in metadata resolve on the Hub
        if (OUT / d).exists():
            shutil.rmtree(OUT / d)
        (OUT / d).mkdir()
    import os
    for r in read_jsonl(ROOT / "metadata.jsonl"):
        for k in ("crop_file", "mask_file", "rect_file", "rect_mask_file"):
            if r.get(k):
                try:
                    os.link(ROOT / r[k], OUT / r[k])
                except OSError:
                    shutil.copy2(ROOT / r[k], OUT / r[k])
    for d in HF_DIRS:
        if (OUT / d).exists():
            shutil.rmtree(OUT / d)
        shutil.copytree(ROOT / d, OUT / d)
    card = (OUT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)#\s]+)\)", card) + re.findall(r'^license_link:\s*"?([^"\n]+)"?', card, re.M)
    missing = sorted({x for x in links if not re.match(r"[a-z]+://", x) and not (OUT / x).exists()})
    if missing:
        raise SystemExit(f"dist/hf/README.md links to files not in dist/hf/: {missing}")
    return links


def write_public_metadata(recs: list[dict]) -> None:
    """metadata.jsonl / metadata.csv for the Hub: the published fields only (PROGRESS E-38)."""
    import csv
    with open(OUT / "metadata.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(public(r), ensure_ascii=False) + "\n")
    shutil.copy2(ROOT / "metadata.csv", OUT / "metadata.csv")  # build_catalog.py writes the same projection
    head = next(csv.reader(open(OUT / "metadata.csv", encoding="utf-8")))
    assert head == PUBLIC_FIELDS, head


_CFG = json.loads((ROOT / "release_config.json").read_text(encoding="utf-8"))
HF_DOCS = ["README.md", "ATTRIBUTION.md", "许可说明.md", "免责声明.md", "CHANGELOG.md"]  # E-35 / E-38 / E-39
HF_LINKED = [(f"dist/{_CFG['seg_zip_name']}", _CFG["seg_zip_name"]),  # segmentation results, one click (E-38); no full ZIP (E-39)
             (f"dist/{_CFG['seg_front_zip_name']}", _CFG["seg_front_zip_name"]),  # frontal views only (AI judgement, E-48)
             (f"dist/{_CFG['rect_zip_name']}", _CFG["rect_zip_name"]),  # rectified crops and masks (E-55)
             (f"dist/db/wadang_v{_CFG['version']}.sqlite", f"wadang_v{_CFG['version']}.sqlite")]  # catalogue database (E-87)
HF_DIRS: list[str] = []  # no vocab/ (labels are not published), no LICENSES/ (each row links its licence); never evidence/


def main():
    global META_COLS
    setup_console()
    if "--docs-only" in sys.argv:
        links = sync_docs()
        print("docs synced; card links checked:", len(links))
        return
    recs = read_jsonl(ROOT / "metadata.jsonl")
    META_COLS = list(PUBLIC_FIELDS)
    if (OUT / "data").exists():
        shutil.rmtree(OUT / "data")
    orig = {"image": lambda r: (ROOT / r["file_name"], Path(r["file_name"]).name)}
    n1, b1 = write_config("default", recs, orig)
    # a mask the eye review judged wrong (quality_flags=mask_wrong_review, PROGRESS E-28) is not served as a standard crop
    stdr = [r for r in recs if r.get("crop_file") and (ROOT / r["crop_file"]).exists() and (ROOT / r["mask_file"]).exists()
            and "mask_wrong_review" not in r["quality_flags"]]
    n3, b3 = write_config("std", stdr, {"image": lambda r: (ROOT / r["crop_file"], Path(r["crop_file"]).name),
                                        "mask": lambda r: (ROOT / r["mask_file"], Path(r["mask_file"]).name)})
    rectr = [r for r in stdr if r.get("rect_file") and (ROOT / r["rect_file"]).exists() and (ROOT / r["rect_mask_file"]).exists()]
    n4, b4 = write_config("rect", rectr, {"image": lambda r: (ROOT / r["rect_file"], Path(r["rect_file"]).name),  # E-55
                                          "mask": lambda r: (ROOT / r["rect_mask_file"], Path(r["rect_mask_file"]).name)})
    links = sync_docs()
    wlog("Stage7", "构建 HF Parquet 分片（图像以 datasets.Image 结构嵌入，zstd 压缩，单片 ≤450 MB）；同步数据卡及其链接文件到 dist/hf/",
         "python -I scripts/build_hf.py", len(recs),
         f"default {len(recs)} 行 / {n1} 片 / {b1 / 1e6:.0f} MB；"
         f"std {len(stdr)} 行 / {n3} 片 / {b3 / 1e6:.0f} MB；rect {len(rectr)} 行 / {n4} 片 / {b4 / 1e6:.0f} MB；"
         f"数据卡相对链接 {len(links)} 个全部可解析")
    print(len(recs), len(stdr), len(rectr))


if __name__ == "__main__":
    main()
