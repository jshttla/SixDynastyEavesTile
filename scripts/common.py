"""Shared constants and helpers for the wadang dataset build.

Everything that encodes a *rule* (licence whitelist, enums, paths) lives here so that
collectors cannot invent values. validate.py imports the same tables.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
STAGING = ROOT / "staging"
IMAGES = ROOT / "images"
STD = ROOT / "std"
REPORTS = ROOT / "reports"
EVIDENCE = ROOT / "evidence"
IDS = ROOT / "ids"
VOCAB = ROOT / "vocab"
DATA_LOCAL = ROOT / "data_local"
WORKLOG = ROOT / "WORKLOG.md"
REVIEW_QUEUE = ROOT / "REVIEW_QUEUE.md"
QA_LOG = ROOT / "QA_LOG.md"

UA = "wadang-dataset-builder/0.1 (open research dataset of roof-tile images; contact via repository issues)"

# ---------------------------------------------------------------------------
# Licence whitelist (方案 4.4). Collectors may ONLY emit these keys.
# ---------------------------------------------------------------------------
LICENSES: dict[str, dict[str, Any]] = {
    "CC0-1.0": {
        "url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution_required": False,
        "zh": "CC0 1.0 公有领域贡献",
        "en": "CC0 1.0 Universal Public Domain Dedication",
    },
    "CC-BY-4.0": {
        "url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution_required": True,
        "zh": "知识共享 署名 4.0 国际",
        "en": "Creative Commons Attribution 4.0 International",
    },
    "CC-BY-3.0": {
        "url": "https://creativecommons.org/licenses/by/3.0/",
        "attribution_required": True,
        "zh": "知识共享 署名 3.0",
        "en": "Creative Commons Attribution 3.0",
    },
    "Public-Domain": {
        "url": "https://creativecommons.org/publicdomain/mark/1.0/",
        "attribution_required": False,
        "zh": "公有领域（Public Domain Mark 1.0）",
        "en": "Public Domain (Public Domain Mark 1.0)",
    },
    "ColBase-Terms": {
        # Verified 2026-11-08: SPA route /pages/term (documents API path "term"); evidence/colbase_terms_*.json
        "url": "https://colbase.nich.go.jp/pages/term?locale=ja",
        "attribution_required": True,
        "zh": "ColBase 利用规约（与 CC BY 4.0 兼容）",
        "en": "ColBase Terms of Use (compatible with CC BY 4.0)",
    },
    "KOGL-Type-1": {
        "url": "https://www.kogl.or.kr/info/licenseType1.do",
        "attribution_required": True,
        "zh": "韩国公共信息开放许可 第1类型（공공누리 제1유형，须署名）",
        "en": "Korea Open Government License Type 1 (attribution)",
    },
}

ENUMS: dict[str, set] = {
    "object_type": {"eaves_tile", "antefix", "ridge_tile", "mask", "figurine_head", "sculpture_head", "other"},
    "motif": {"human_face", "beast_face", "demon_face", "lotus", "cloud", "text", "four_spirits", "animal",
              "palmette", "gorgoneion", "silenus", "maenad", "geometric", "other", "unknown"},
    "view": {"front", "back", "side", "oblique"},
    "modality": {"photo", "rubbing", "drawing"},
    "completeness": {"complete", ">70%", "<70%"},
    "label_source": {"institution_record", "institution_title_keyword", "inferred"},
    "qa_status": {"pending", "human_checked"},
    "tilt_source": {"ellipse", "normal", "both", "rubbing", "none"},
    "license": set(LICENSES),
}

FACE_MOTIFS = {"human_face", "beast_face", "demon_face", "gorgoneion", "silenus", "maenad"}

# Source keys and display names (holding institution may differ from source, e.g. Commons).
SOURCES = {
    "met": "The Metropolitan Museum of Art Open Access",
    "smithsonian": "Smithsonian Open Access",
    "colbase": "ColBase (National Institutes for Cultural Heritage, Japan)",
    "emuseum": "e-Museum (Korea)",
    "harvard": "Harvard Library",
    "commons": "Wikimedia Commons",
    "cma": "Cleveland Museum of Art Open Access",
    "aic": "Art Institute of Chicago",
    "europeana": "Europeana",
    "getty": "J. Paul Getty Museum",
    "rijks": "Rijksmuseum",
    "walters": "Walters Art Museum",
}


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def setup_console() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WORKLOG
# ---------------------------------------------------------------------------
_wlock_t = threading.Lock()


class _XLock:
    """Cross-process + cross-thread lock using an exclusive lock file (several collectors run in parallel)."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        _wlock_t.acquire()
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() - t0 > 30:  # stale lock
                    try:
                        os.remove(self.path)
                    except OSError:
                        pass
                    t0 = time.time()
                time.sleep(0.05)

    def __exit__(self, *a):
        os.close(self.fd)
        try:
            os.remove(self.path)
        except OSError:
            pass
        _wlock_t.release()


_wlock = _XLock(ROOT / ".worklog.lock")


def _next_wid() -> str:
    n = 0
    if WORKLOG.exists():
        for m in re.finditer(r"^### W(\d{4})", WORKLOG.read_text(encoding="utf-8"), re.M):
            n = max(n, int(m.group(1)))
    return f"W{n + 1:04d}"


def wlog(stage: str, action: str, cmd: str = "", raw_count: Any = "", conclusion: str = "", extra: str = "") -> str:
    """Append one WORKLOG entry; returns its id (e.g. W0042)."""
    with _wlock:
        wid = _next_wid()
        lines = [f"### {wid} · {utcnow()} · {stage}", f"- 动作：{action}"]
        if cmd:
            lines.append(f"- 命令/URL：`{cmd}`" if "\n" not in cmd and len(cmd) < 400 else f"- 命令/URL：\n```\n{cmd}\n```")
        if raw_count != "":
            lines.append(f"- 原始返回数：{raw_count}")
        if conclusion:
            lines.append(f"- 结论：{conclusion}")
        if extra:
            lines.append(extra)
        with open(WORKLOG, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(lines) + "\n")
        return wid


def review(item_url: str, seen_text: str, why: str, suggestion: str, kind: str = "license") -> None:
    with _wlock:
        with open(REVIEW_QUEUE, "a", encoding="utf-8") as f:
            f.write(f"\n- [{kind}] {utcnow()} · {item_url}\n  - 原文：{seen_text}\n  - 拿不准的原因：{why}\n  - 建议：{suggestion}\n")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Http:
    """requests.Session with polite rate limiting and retries."""

    def __init__(self, rate_per_sec: float = 2.0, headers: Optional[dict] = None, timeout: int = 60):
        import requests

        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        if headers:
            self.s.headers.update(headers)
        self.min_dt = 1.0 / rate_per_sec
        self.t_last = 0.0
        self.timeout = timeout
        self.lock = threading.Lock()

    def _wait(self):
        with self.lock:
            dt = time.time() - self.t_last
            if dt < self.min_dt:
                time.sleep(self.min_dt - dt)
            self.t_last = time.time()

    def get(self, url: str, params: Optional[dict] = None, tries: int = 4, stream: bool = False, **kw):
        last = None
        for i in range(tries):
            self._wait()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout, stream=stream, **kw)
                if r.status_code in (429, 500, 502, 503, 504):
                    last = RuntimeError(f"HTTP {r.status_code} {url}")
                    time.sleep(2 ** (i + 1))
                    continue
                return r
            except Exception as e:  # network
                last = e
                time.sleep(2 ** (i + 1))
        raise last  # type: ignore[misc]

    def json(self, url: str, params: Optional[dict] = None, **kw):
        r = self.get(url, params=params, **kw)
        r.raise_for_status()
        return r.json()


def sha256_file(path: os.PathLike | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def safe_ascii(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_")
    return s[:maxlen] or "x"


def sniff_ext(b: bytes) -> Optional[str]:
    if b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if b[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return None


def download_image(http: Http, url: str, dest_stem: Path, min_bytes: int = 5000) -> tuple[Path, str, int]:
    """Download bytes verbatim (no re-encode). Returns (path, sha256, nbytes). Raises on non-image."""
    r = http.get(url)
    r.raise_for_status()
    b = r.content
    ext = sniff_ext(b)
    if ext is None:
        raise ValueError(f"not an image ({r.headers.get('content-type')}) {url}")
    if len(b) < min_bytes:
        raise ValueError(f"too small ({len(b)} B) {url}")
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    p = dest_stem.parent / (dest_stem.name + ext)  # not with_suffix(): stems may contain dots (accession numbers)
    p.write_bytes(b)
    return p, sha256_bytes(b), len(b)


def read_jsonl(path: os.PathLike | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: os.PathLike | str, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
            n += 1
    os.replace(tmp, p)
    return n


# ---------------------------------------------------------------------------
# Staging rows: what every collector writes to staging/<source>.jsonl (one row per downloaded image).
# build_metadata.py turns these into metadata.jsonl (IDs, label mapping, citations).
# Collectors copy museum text VERBATIM into *_raw fields; they never set motif/region labels themselves.
# ---------------------------------------------------------------------------
STAGE_FIELDS = {
    # identity / provenance
    "source": True,            # key of SOURCES
    "source_id": True,         # object-level id at the source (str)
    "image_key": True,         # unique per image within source, e.g. f"{source_id}#{n}"
    "image_index": True,       # 0 = primary image of the object, 1.. = additional views
    "source_url": True,        # human-viewable item page
    "image_url": True,         # exact URL the bytes were downloaded from
    "iiif_manifest": False,
    "record_file": True,       # raw/<source>/meta/<safe id>.json (verbatim API record / page extract)
    # verbatim museum text (None if absent)
    "title_raw": False, "object_name_raw": False, "description_raw": False, "classification_raw": False,
    "culture_raw": False, "period_raw": False, "date_raw": False, "date_from": False, "date_to": False,
    "region_raw": False,       # find spot / place of origin as written by the museum
    "medium_raw": False, "dimensions_raw": False, "credit_line_raw": False, "subjects_raw": False,
    "holding_institution": True, "holding_institution_qid": False, "accession_number": False,
    # licence
    "license": True,           # key of LICENSES
    "rights_raw": True,        # the item-level rights statement exactly as the source gives it
    "license_evidence": True,  # license_evidence(page_url, date, quote, item_rights)
    "rights_holder": False, "attribution_text": False,
    # hints taken from the record (not from file names)
    "modality_hint": False,    # "photo" | "rubbing" | "drawing" | None
    "is_reproduction": False,  # True only if the record says copy/replica/cast/modern
    # file
    "raw_file": True,          # path relative to ROOT, ASCII
    "sha256": True, "bytes": True, "width": True, "height": True,
    "date_accessed": True,
    "mirror_of": False,        # e.g. Commons file mirrored from a museum: museum item URL
    "notes": False,
}


def stage_row(**kw) -> dict:
    unknown = set(kw) - set(STAGE_FIELDS)
    if unknown:
        raise KeyError(f"unknown staging fields {unknown}")
    row = {k: kw.get(k) for k in STAGE_FIELDS}
    for k, req in STAGE_FIELDS.items():
        if req and row[k] in (None, ""):
            raise ValueError(f"staging field {k} required ({row.get('source')}:{row.get('source_id')})")
    if row["source"] not in SOURCES:
        raise ValueError(f"unknown source {row['source']}")
    if row["license"] not in LICENSES:
        raise ValueError(f"license {row['license']} not in whitelist")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", row["raw_file"]):
        raise ValueError(f"raw_file not ASCII: {row['raw_file']}")
    if LICENSES[row["license"]]["attribution_required"] and not row["attribution_text"]:
        raise ValueError(f"attribution_text required for {row['license']}")
    return row


def image_size(path: os.PathLike | str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def license_evidence(page_url: str, date: str, quote: str, item_rights: str = "") -> str:
    """方案格式：许可页 URL | 抓取日期 | 原文关键句 （+ 条目级权利字段原文）"""
    s = f"{page_url} | {date} | {quote}"
    if item_rights:
        s += f" | item: {item_rights}"
    return s


# Official names, exactly as the sources / licensors write them (author, PROGRESS E-36: never translate official names).
OFFICIAL_SOURCE = {"colbase": "ColBase", "emuseum": "e뮤지엄", "commons": "Wikimedia Commons", "europeana": "Europeana",
                   "getty": "J. Paul Getty Museum", "met": "The Metropolitan Museum of Art", "walters": "The Walters Art Museum",
                   "cma": "The Cleveland Museum of Art"}
OFFICIAL_LICENSE = {"CC0-1.0": "CC0 1.0", "Public-Domain": "Public domain", "CC-BY-4.0": "CC BY 4.0", "CC-BY-3.0": "CC BY 3.0",
                    "ColBase-Terms": "ColBase 利用規約", "KOGL-Type-1": "공공누리 제1유형"}

# PROGRESS E-38 (author, 2026-11-10): only the images are given. A licence asks for attribution, not for the object's name,
# date or place; so every published table / file / page carries these fields and nothing else. The full record
# (metadata.jsonl at the root) stays internal, for takedown and correction requests.
# E-39: nothing is written unless the licence asks for it — no per-image licence name or link either (a licence that asks
# for them has them inside its attribution text).
PUBLIC_FIELDS = ["image_id", "object_id", "motif", "frontal", "attribution_text", "attribution_text_derived", "crop_file", "mask_file",
                 "rect_file", "rect_mask_file",  # E-55: rectified crop + mask (null when the rim was not fitted as an ellipse)
                 "width", "height", "sha256"]  # E-40: attribution_text = original, attribution_text_derived = every derived file


def public(r: dict) -> dict:
    out = {k: r.get(k) for k in PUBLIC_FIELDS}
    if not r.get("attribution_required"):
        out["attribution_text"] = out["attribution_text_derived"] = None
    return out


def local_settings() -> dict:
    """Machine-specific or private values: local file paths, extra leak-check patterns, where an API key file lives.
    Read from local_settings.json at the repo root, which is git-ignored and never published (see local_settings.example.json)."""
    p = ROOT / "local_settings.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
