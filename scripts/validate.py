"""validate.py — the gatekeeper. Run before every commit; must be all green.

Usage:
    python scripts/validate.py                 # metadata.jsonl in repo root
    python scripts/validate.py --hash          # also re-hash every image file present
    python scripts/validate.py --ci            # + repo hygiene (no data_local, no 4096x3072 jpg tracked)
Exit code 0 = green.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENUMS, FACE_MOTIFS, LICENSES, ROOT, setup_console, sha256_file  # noqa: E402

FIELDS = [  # PROGRESS E-34: the release carries no classification / interpretation labels
    "image_id", "object_id", "mold_id", "file_name", "motif", "frontal", "culture_period_raw", "region_raw",
    "holding_institution", "holding_institution_qid", "accession_number",
    "diameter_cm", "dimensions_raw", "completeness", "source",
    "source_id", "source_url", "iiif_manifest", "license", "license_url", "license_evidence", "rights_holder",
    "attribution_text", "attribution_text_derived", "attribution_required", "modifications", "date_accessed",
    "sha256", "mask_file", "crop_file", "rectified_file", "rect_file", "rect_mask_file", "completeness_ratio", "rim_arc_deg", "tilt_deg",
    "tilt_source", "tilt_validation", "quality_flags",
    "title_raw", "object_name_raw", "description_raw", "medium_raw", "image_url", "rights_raw", "mirror_of",
    "width", "height",
]
LABEL_FIELDS = {"citation_zh", "citation_en", "notes", "object_type", "motif_raw", "has_face", "object_count", "view", "is_reproduction", "date_from", "date_to",
                "region_code", "country_iso3166", "modality", "label_source", "qa_status", "region_group", "period_code"}
REQUIRED_NONEMPTY = ["image_id", "object_id", "file_name", "source", "source_id", "source_url", "license", "license_url",
                     "license_evidence", "modifications", "date_accessed", "sha256", "motif",
                     "holding_institution"]
NULLABLE_ENUMS = {"completeness", "tilt_source"}
NON_FACE = {"lotus", "cloud", "text", "palmette", "geometric"}
ASCII_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")
REGION_GROUPS = {"china", "korea", "japan", "mediterranean", "other", "unknown"}


def check_rows(rows: list[dict], do_hash: bool) -> list[str]:
    errs: list[str] = []
    seen_img, seen_sha = Counter(), Counter()
    for i, r in enumerate(rows, 1):
        tag = f"line {i} ({r.get('image_id')})"
        missing = [k for k in FIELDS if k not in r]
        if missing:
            errs.append(f"{tag}: missing keys {missing}")
        extra = [k for k in r if k not in FIELDS]
        if extra:
            errs.append(f"{tag}: unknown keys {extra}")
        for k in REQUIRED_NONEMPTY:
            if r.get(k) in (None, "", []):
                errs.append(f"{tag}: empty required field {k}")
        if not re.fullmatch(r"img_\d{6}", str(r.get("image_id"))):
            errs.append(f"{tag}: bad image_id")
        if not re.fullmatch(r"obj_\d{6}", str(r.get("object_id"))):
            errs.append(f"{tag}: bad object_id")
        if r.get("mold_id") is not None:
            errs.append(f"{tag}: mold_id must be null in v0.1")
        seen_img[r.get("image_id")] += 1
        seen_sha[r.get("sha256")] += 1
        for k, allowed in ENUMS.items():
            if k not in FIELDS:
                continue
            v = r.get(k)
            if v is None and k in NULLABLE_ENUMS:
                continue
            if v not in allowed:
                errs.append(f"{tag}: {k}={v!r} not in enum")
        # licence
        lic = r.get("license")
        if lic in LICENSES:
            if r.get("license_url") != LICENSES[lic]["url"]:
                errs.append(f"{tag}: license_url does not match table for {lic}")
            if r.get("attribution_required") is not LICENSES[lic]["attribution_required"]:
                errs.append(f"{tag}: attribution_required must be {LICENSES[lic]['attribution_required']} for {lic}")
            if LICENSES[lic]["attribution_required"] and not r.get("attribution_text"):
                errs.append(f"{tag}: attribution_text required for {lic}")
            if not LICENSES[lic]["attribution_required"] and r.get("attribution_text"):
                errs.append(f"{tag}: no credit line unless the licence requires one (PROGRESS E-35)")
        if " | " not in str(r.get("license_evidence", "")):
            errs.append(f"{tag}: license_evidence must be 'URL | date | quote'")
        if not str(r.get("source_url", "")).startswith(("http://", "https://")):
            errs.append(f"{tag}: source_url not http(s)")
        if not re.fullmatch(r"[0-9a-f]{64}", str(r.get("sha256"))):
            errs.append(f"{tag}: bad sha256")
        # paths
        for k in ("file_name", "mask_file", "crop_file", "rect_file", "rect_mask_file"):
            v = r.get(k)
            if v is not None and not ASCII_PATH.match(v):
                errs.append(f"{tag}: {k} not ASCII-safe: {v}")
            if v is not None and ("data_local" in v or "local_only" in v):
                errs.append(f"{tag}: {k} points into data_local (local-only data must never be published)")
        if not str(r.get("file_name", "")).startswith("images/"):
            errs.append(f"{tag}: file_name must be under images/")
        if r.get("rectified_file") is not None:  # legacy field; the published rectified pair is rect_file / rect_mask_file (E-55)
            errs.append(f"{tag}: rectified_file must be null (use rect_file / rect_mask_file)")
        if bool(r.get("rect_file")) != bool(r.get("rect_mask_file")) or (r.get("rect_file") and not r.get("crop_file")):
            errs.append(f"{tag}: rect_file / rect_mask_file must come as a pair, next to a standard crop")
        if r.get("tilt_deg") is not None and r.get("tilt_validation") != "synthetic_only":
            errs.append(f"{tag}: tilt_deg published without tilt_validation=synthetic_only")
        # no labels in the release (PROGRESS E-34)
        lab = sorted(LABEL_FIELDS & set(r))
        if lab:
            errs.append(f"{tag}: label fields must not be published: {lab}")
        for k in ("width", "height"):
            if not isinstance(r.get(k), int):
                errs.append(f"{tag}: {k} must be int")
        for k in ("diameter_cm", "completeness_ratio", "rim_arc_deg", "tilt_deg"):
            if r.get(k) is not None and not isinstance(r.get(k), (int, float)):
                errs.append(f"{tag}: {k} must be numeric or null")
        for k in ():
            if r.get(k) is not None and not isinstance(r.get(k), bool):
                errs.append(f"{tag}: {k} must be bool or null")
        if not isinstance(r.get("quality_flags"), list):
            errs.append(f"{tag}: quality_flags must be a list")
        if r.get("date_accessed") and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["date_accessed"]):
            errs.append(f"{tag}: date_accessed not YYYY-MM-DD")
        if do_hash:
            p = ROOT / r.get("file_name", "")
            if p.exists():
                if sha256_file(p) != r.get("sha256"):
                    errs.append(f"{tag}: sha256 mismatch for {p.name}")
            else:
                errs.append(f"{tag}: file missing {r.get('file_name')}")
    for k, n in seen_img.items():
        if n > 1:
            errs.append(f"duplicate image_id {k} x{n}")
    for k, n in seen_sha.items():
        if n > 1:
            errs.append(f"duplicate sha256 {k[:12]}… x{n}")
    return errs


def check_vocab() -> list[str]:
    errs = []
    vf = ROOT / "vocab" / "motif.csv"
    if vf.exists():
        with open(vf, encoding="utf-8") as f:
            codes = {row["code"] for row in csv.DictReader(f)}
        if codes != ENUMS["motif"]:
            errs.append(f"vocab/motif.csv codes differ from enum: {sorted(codes ^ ENUMS['motif'])}")
    vf = ROOT / "vocab" / "object_type.csv"
    if vf.exists():
        with open(vf, encoding="utf-8") as f:
            codes = {row["code"] for row in csv.DictReader(f)}
        if codes != ENUMS["object_type"]:
            errs.append(f"vocab/object_type.csv codes differ from enum: {sorted(codes ^ ENUMS['object_type'])}")
    return errs


def check_repo() -> list[str]:
    errs = []
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    except Exception as e:
        return [f"git ls-files failed: {e}"]
    for f in files:
        if f.startswith("data_local/") or ("own_" in f and f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".npz"))):
            errs.append(f"tracked local-only file: {f}")
        if f.lower().endswith((".jpg", ".jpeg")):
            try:
                from PIL import Image
                with Image.open(ROOT / f) as im:
                    if im.size in ((4096, 3072), (3072, 4096)):
                        errs.append(f"tracked 4096x3072 jpg (author photo size): {f}")
            except Exception:
                pass
    return errs


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "metadata.jsonl"))
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--ci", action="store_true")
    a = ap.parse_args()
    p = Path(a.file)
    rows = []
    errs: list[str] = []
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        errs.append(f"line {i}: invalid JSON {e}")
    errs += check_rows(rows, a.hash)
    errs += check_vocab()
    if a.ci:
        errs += check_repo()
    if errs:
        for e in errs[:200]:
            print("FAIL", e)
        print(f"\nRED: {len(errs)} problem(s) in {len(rows)} rows")
        return 1
    print(f"GREEN: {len(rows)} rows, 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
