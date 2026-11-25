"""Stage 7: README.md (HF dataset card), DATASHEET.md, CITATION.cff, .zenodo.json, croissant.json, CHANGELOG.md,
ATTRIBUTION.md — all numbers computed from metadata.jsonl and logged to ONE WORKLOG entry that the card cites.

Usage: python -I scripts/build_card.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import LICENSES, REPORTS, ROOT, SOURCES, read_jsonl, setup_console, today, wlog  # noqa: E402

GROUPS = ["china", "korea", "japan", "mediterranean", "other", "unknown"]
GROUP_ZH = {"china": "中国", "korea": "朝鲜半岛", "japan": "日本", "mediterranean": "地中海", "other": "其他", "unknown": "未定"}
MOTIFS = ["human_face", "beast_face", "demon_face", "gorgoneion", "silenus", "maenad", "lotus", "cloud", "text",
          "four_spirits", "animal", "palmette", "geometric", "other", "unknown"]


def cfg() -> dict:
    p = ROOT / "release_config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def stats(recs):
    s = {"n_images": len(recs), "n_objects": len({r["object_id"] for r in recs})}
    s["by_source_images"] = dict(Counter(r["source"] for r in recs))
    s["by_source_objects"] = {k: len({r["object_id"] for r in recs if r["source"] == k}) for k in s["by_source_images"]}
    s["by_license"] = dict(Counter(r["license"] for r in recs))
    s["by_group"] = dict(Counter(r["region_group"] for r in recs))
    s["by_motif"] = dict(Counter(r["motif"] for r in recs))
    s["by_modality"] = dict(Counter(r["modality"] for r in recs))
    s["by_object_type"] = dict(Counter(r["object_type"] for r in recs))
    s["by_label_source"] = dict(Counter(r["label_source"] for r in recs))
    s["group_x_motif"] = {g: {m: sum(1 for r in recs if r["region_group"] == g and r["motif"] == m) for m in MOTIFS} for g in GROUPS}
    s["group_x_modality"] = {g: {m: sum(1 for r in recs if r["region_group"] == g and r["modality"] == m)
                                 for m in ("photo", "rubbing", "drawing")} for g in GROUPS}
    s["by_period"] = dict(Counter(r["period_code"] or "none" for r in recs).most_common())
    s["has_face_true"] = sum(1 for r in recs if r["has_face"] is True)
    s["faces_config"] = 0  # faces config withdrawn (E-34)
    s["std_config"] = sum(1 for r in recs if r.get("crop_file"))
    s["tilt_source"] = dict(Counter(r["tilt_source"] or "null" for r in recs))
    s["completeness"] = dict(Counter(r["completeness"] or "null" for r in recs))
    s["qa_pending"] = sum(1 for r in recs if r["qa_status"] == "pending")
    s["quality_flags"] = dict(Counter(f for r in recs for f in r["quality_flags"]))
    s["institutions"] = len({r["holding_institution"] for r in recs})
    s["diameter_known"] = sum(1 for r in recs if r["diameter_cm"] is not None)
    return s


def readme(recs, s, wid, c) -> str:
    yaml = f"""---
license: other
license_name: mixed-see-dataset-card
license_link: "{c.get('hf_url', '')}/blob/main/%E8%AE%B8%E5%8F%AF%E8%AF%B4%E6%98%8E.md"
pretty_name: "{c.get('dataset_title_en', 'Open Roof-Tile End Image Dataset')}"
language:
- zh
- en
- ja
- ko
task_categories:
- image-classification
- image-segmentation
tags:
- cultural-heritage
- archaeology
- museum
- roof-tile
- wadang
- antefix
- digital-humanities
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/default/*.parquet
- config_name: faces
  data_files:
  - split: train
    path: data/faces/*.parquet
- config_name: std
  data_files:
  - split: train
    path: data/std/*.parquet
---
"""
    gm = [[GROUP_ZH[g] + f" ({g})"] + [s["group_x_motif"][g][m] for m in MOTIFS] + [sum(s["group_x_motif"][g].values())] for g in GROUPS]
    gmod = [[GROUP_ZH[g] + f" ({g})"] + [s["group_x_modality"][g][m] for m in ("photo", "rubbing", "drawing")] for g in GROUPS]
    src_rows = []
    for k in sorted(s["by_source_images"], key=lambda k: -s["by_source_images"][k]):
        lic = Counter(r["license"] for r in recs if r["source"] == k)
        src_rows.append([SOURCES.get(k, k), s["by_source_objects"][k], s["by_source_images"][k],
                         ", ".join(f"{a} {b}" for a, b in lic.most_common())])
    funnel = ""
    fp = REPORTS / "source_funnel.csv"
    if fp.exists():
        rows = list(csv.DictReader(open(fp, encoding="utf-8")))
        funnel = "\n### 各来源检索漏斗 / Collection funnel per source\n\n" + table(list(rows[0]), [list(r.values()) for r in rows]) + "\n"
    lic_rows = [[k, LICENSES[k]["en"], "yes" if LICENSES[k]["attribution_required"] else "no", s["by_license"].get(k, 0)] for k in LICENSES]
    hf = c.get("hf_repo", "USER/wadang")
    return yaml + f"""
# {c.get('dataset_title_zh', '瓦当开放图像数据集')} · {c.get('dataset_title_en', 'Open Roof-Tile End Image Dataset')} (v0.1.0)

**如有侵权或勘误，请随时联系 / Copyright concerns or corrections: {c.get('contact_email', '')}**

本数据集为纯学术项目，遵守适用的法律法规，不含有、也不代表任何政治观点；馆方原文不代表维护者观点。详见[声明与免责](免责声明.md)。 Purely academic; contains and represents no political views; museum text does not represent the maintainer's views. See the [Statement and Disclaimer](免责声明.md).

**研究者入口 / For humanities researchers:** 图库网站（可按地区、时代、母题、馆藏筛选，每件附出处注释与许可说明）{c.get('site_url', '')} ·
整库下载 ZIP 与 Excel 目录：{c.get('hf_file_url_template', '').format(hf_repo=c.get('hf_repo'), file_name=c.get('zip_name')) if 'TODO' in c.get('zenodo_url', 'TODO') else c.get('zenodo_url')} · [中文说明](README_zh.md) · [如何引用](如何引用.md) · [许可说明](许可说明.md) · [数据来源与可信度](数据来源与可信度.md)

A cross-regional image dataset of ancient roof-tile ends (Chinese *wadang* incl. semicircular *ban-wadang*, Japanese *nokimarugawara*,
Korean *sumaksae*, Mediterranean antefixes; photographs, rubbings and fragments) with per-image provenance and licence, built only from
open-access museum and archive sources. Scope: the tile closing the front end of the cover tile at the eaves (zh.wikipedia 瓦當);
flat eaves tiles (*nokihiragawara*), ridge-end tiles (*onigawara*) and ridge ornaments were reviewed image by image and excluded. Every number on this card is computed from `metadata.jsonl` (maintainer's build log entry **{wid}**).

- **{s['n_images']} images of {s['n_objects']} objects** from {len(s['by_source_images'])} sources and {s['institutions']} holding institutions.
- All labels are **pending human review** (`qa_status = pending` for {s['qa_pending']} / {s['n_images']} rows): this is a v0.1 release.
- Labels come only from the holding institution's own record (`label_source`); where the institution says nothing, `motif = unknown`.
  Model outputs were used only to queue items for human review, never written into label fields.

## Coverage / 覆盖

### Region × motif (images; zeros shown) / 地区 × 母题

{table(['region_group'] + MOTIFS + ['total'], gm)}

### Region × modality / 地区 × 模态

{table(['region_group', 'photo', 'rubbing', 'drawing'], gmod)}

### Sources and licences / 来源与许可

{table(['source', 'objects', 'images', 'licences (images)'], src_rows)}
{funnel}
## Licence / 许可

The dataset is **mixed-licence**. Each row carries `license`, `license_url`, `license_evidence`, `rights_holder`,
`attribution_required` and a ready-made `attribution_text`.

{table(['license', 'meaning', 'attribution required', 'images'], lic_rows)}

- **Our contributions** (metadata normalisation, labels, QA records, derived masks/crops where the source image is CC0/PD, reports) are
  released under **CC0 1.0** (`LICENSE-DATA`; it covers these parts only, not the images). Code is **MIT** (`LICENSE`). We do not require attribution for our parts.
- Derived images (`std/`) inherit the licence of their source image; ColBase-derived crops are marked "一部加工" in `attribution_text`.
- KOGL Type 1 items must not be relabelled as CC BY. Full licence texts: `LICENSES/`. Per-item attribution list: `ATTRIBUTION.md`.

Filter by licence in three lines:

```python
from datasets import load_dataset
ds = load_dataset("{hf}", "default", split="train")
cc0_only = ds.filter(lambda lic: lic in ("CC0-1.0", "Public-Domain"), input_columns="license")  # no attribution needed
```

## Configs

| config | rows | content |
|---|---|---|
| `default` | {s['n_images']} | one row per image: original image bytes + all metadata |
| `faces` | {s['faces_config']} | `has_face = true`, excluding reproductions, back/side views, group records and multi-object photos |
| `std` | {s['std_config']} | standardised 1024 px crop (`image`) + 3-value mask (`mask`: 0 background, 255 tile front, 128 occluder) |

```python
from datasets import load_dataset
faces = load_dataset("{hf}", "faces", split="train")
silla = faces.filter(lambda pc: pc == "KR-UnifiedSilla", input_columns="period_code")  # input_columns: no image decoding
```

Field definitions: `SCHEMA.md`. Controlled vocabularies (zh / en / ja / ko): `vocab/`.

## How labels were made / 标签怎么来的

| label_source | meaning | images |
|---|---|---|
| institution_record | from a structured subject/keyword field of the holding institution | {s['by_label_source'].get('institution_record', 0)} |
| institution_title_keyword | from a keyword in the institution's own title / object name | {s['by_label_source'].get('institution_title_keyword', 0)} |
| inferred | the institution does not state the motif → `motif = unknown` (no guessing) | {s['by_label_source'].get('inferred', 0)} |

Region (`region_group`, `region_code`) and dates come only from institution records. File names (e.g. Wikimedia Commons titles) are never used.

Geometry (`tilt_deg`, `rim_arc_deg`, `completeness_ratio`) is automatic (Grounding DINO + SAM 2.1 segmentation, RANSAC ellipse,
MoGe-2 plane normal). `tilt_validation = synthetic_only`: tilt accuracy has been checked on synthetic affine tilts only
(`reports/tilt_validation.md`); physical validation is pending. `rectified_file` is null in this version on purpose.

## Known biases and limitations / 已知偏差

- Sampling follows what museums have released under open licences, not what exists: Japanese national museums (ColBase) and
  Mediterranean antefixes are over-represented; mainland-Chinese museum holdings are absent (their images are not openly licensed).
- Modalities are unbalanced (photo {s['by_modality'].get('photo', 0)}, rubbing {s['by_modality'].get('rubbing', 0)}, drawing {s['by_modality'].get('drawing', 0)}).
- Six Dynasties (Nanjing area) human-face tiles are scarce in openly licensed collections.
- Many objects have several views; back views are common in ColBase. `view` is filled only where the museum states it.
- Museum metadata errors are kept verbatim in `*_raw` fields; normalised fields may inherit them.
- Tilt from ellipses has a floor of several degrees because hand-made tiles are not perfectly circular.

## Citation / 引用

See `如何引用.md` and `CITATION.cff`. DOI: not yet assigned (Zenodo deposit pending); until then cite the Hugging Face dataset page {c.get('hf_url', '')}.

## Maintenance

Report errors or copyright concerns (no account needed): {c.get('contact_email', '')}. Changes: `CHANGELOG.md`.
"""


def datasheet(s, wid, c) -> str:
    return f"""# DATASHEET（Datasheets for Datasets, Gebru et al.）— v0.1.0

All numbers are computed from `metadata.jsonl` (build log entry {wid}).

## Motivation
- **Purpose.** A cross-regional, provenance-complete image corpus of ancient roof-tile ends (wadang / nokimarugawara / sumaksae / antefixes) for humanities research
  (comparison of motifs across China, the Korean peninsula, Japan and the Mediterranean) and for computer-vision work on worn,
  relief-decorated, non-planar heritage objects. Created first to support a study of facial conventions on eaves tiles (Q2).
- **Creators.** {c.get('author_en', 'the author')} (pseudonymous maintainer; contact {c.get('contact_email', '')}), with an automated build pipeline. No funding.

## Composition
- **Instances.** {s['n_images']} images (rows) of {s['n_objects']} objects; one object may have several views (shared `object_id`).
- **Sample, not complete.** Only items whose images are released under CC0 / Public Domain / CC BY 3.0 / 4.0 / ColBase Terms /
  KOGL Type 1. BY-SA, NC, ND and unknown licences were excluded; doubtful items are listed in `REVIEW_QUEUE.md` and not included.
- **Per instance.** Original image bytes, verbatim museum text (`*_raw`), normalised labels, licence and provenance fields,
  automatic segmentation and geometry. Schema: `SCHEMA.md`.
- **Labels.** From institution records only (`label_source`); {s['by_label_source'].get('inferred', 0)} rows have `motif = unknown`.
  All rows `qa_status = pending` in v0.1.
- **Missing information.** `view`, `object_count`, `region_code`, dates, diameters are null where the institution gives nothing.
- **Relationships.** `object_id` groups views; `mold_id` (same mould) is empty in v0.1. Duplicate candidates: `reports/dedup_candidates.csv`.
- **Splits.** None (single `train` split); users should split by `object_id` to avoid leakage between views of the same object.
- **Errors / noise.** Museum metadata errors are preserved verbatim; automatic segmentation can fail on fragments, back views and
  group photos (`quality_flags`). Group records (several objects under one museum record) are flagged `group_record`.
- **Confidential / offensive / personal data.** None: objects are ancient artefacts; no people are depicted.
- **Not included.** Images whose rights chain or use conditions are unclear, duplicates, and images whose segmentation the eye review
  judged wrong were excluded before release.

## Collection process
- **Acquisition.** Public APIs and website search of: {', '.join(sorted(s['by_source_images']))}. Each search was first run in count-only
  mode; raw hit counts, filters and downloads are logged in the maintainer's build log (UTC; not distributed, it contains local system details). Images were saved byte-for-byte (no re-encoding).
- **Timeframe.** Accessed 2026-11-07/08 (UTC); `date_accessed` per row.
- **Ethics.** Only open-licence museum images; licence evidence (policy page + quote + item rights field) is stored per row (`license_evidence`); the licence-page snapshots are kept by the maintainer as audit evidence and are not redistributed (they carry their own text licences).

## Preprocessing / labelling
- Exact duplicates (same SHA-256) merged keeping the widest licence; near-duplicates only listed for review.
- Label normalisation rules: `scripts/labelmap.py` (unit-tested in `tests/`).
- Two multimodal models gave independent opinions used only to build
  `reports/label_disagreements.csv`; their outputs are not in the label fields.
- Segmentation: Grounding DINO base + SAM 2.1 hiera-large (Apache-2.0); geometry: OpenCV RANSAC ellipse + MoGe-2 (MIT). No model was trained
  or fine-tuned. Raw data and every script are kept so that `metadata.jsonl` can be rebuilt from scratch (`scripts/`).

## Uses
- Suitable: browsing and comparing motifs; teaching; retrieval and classification baselines; segmentation of relief objects.
- Not suitable: claims about the *distribution* of motifs in antiquity (the sample follows open-licence availability); dating or
  attribution of individual objects without consulting the holding institution.

## Distribution
- Static gallery website (free subdomain), Zenodo (ZIP + Excel catalogue + metadata + code, DOI), Wikidata + Commons (planned),
  Hugging Face (Parquet). Mixed licences per row; our contributions CC0,
  code MIT. Attribution requirements: `ATTRIBUTION.md`.

## Maintenance
- Maintainer: {c.get('author_en', 'the author')} (pseudonymous); corrections and takedown requests: {c.get('contact_email', '')}.
- Versioned releases with CHANGELOG; corrections are logged in `QA_LOG.md`.
"""


REGION_ZH = {"china": "中国", "korea": "朝鲜半岛", "japan": "日本", "mediterranean": "地中海", "other": "其他地区", "unknown": "地区未定"}
SOURCE_ZH = {"colbase": "ColBase（日本国立文化财机构）", "emuseum": "韩国 e뮤지엄", "commons": "维基共享资源",
             "europeana": "Europeana", "getty": "盖蒂博物馆", "met": "大都会艺术博物馆", "walters": "沃尔特斯艺术博物馆", "cma": "克利夫兰艺术博物馆"}
LIC_ZH = {"ColBase-Terms": "ColBase 利用规约（与 CC BY 4.0 兼容，须注明出典）", "CC-BY-4.0": "CC BY 4.0（须署名）", "CC0-1.0": "CC0（不要求署名）",
          "KOGL-Type-1": "韩国公共信息开放许可第 1 类型（须署名）", "CC-BY-3.0": "CC BY 3.0（须署名）", "Public-Domain": "公有领域（摄影者本人声明放弃）"}


def readme_zh(s: dict, c: dict) -> str:
    """Chinese dataset card (domestic mirror on ModelScope, and linked from the HF card). Author 2026-11-09, PROGRESS E-32."""
    src = "\n".join(f"| {SOURCE_ZH.get(k, k)} | {s['by_source_images'][k]} | {s['by_source_objects'][k]} |"
                    for k in sorted(s["by_source_images"], key=lambda k: -s["by_source_images"][k]))
    lic = "\n".join(f"| {LIC_ZH.get(k, k)} | {v} |" for k, v in sorted(s["by_license"].items(), key=lambda kv: -kv[1]))
    reg = "、".join(f"{REGION_ZH.get(k, k)} {v}" for k, v in sorted(s["by_group"].items(), key=lambda kv: -kv[1]))
    zipurl = c.get("hf_file_url_template", "").format(hf_repo=c.get("hf_repo"), file_name=c.get("zip_name"))
    return f"""**如有侵权或勘误，请随时联系：{c.get('contact_email', '')}**

本数据集为纯学术项目，遵守适用的法律法规，不含有、也不代表任何政治观点；馆方原文不代表维护者观点。详见[声明与免责](免责声明.md)。

# {c.get('dataset_title_zh', '瓦当开放图像数据集')}（v{c.get('version', '0.1.0')}）

各地区古代瓦当的开放图像：中国瓦当（含半瓦当）、日本軒丸瓦、韩国수막새、地中海 antefix，包括照片、拓片和残片。
只收博物馆与档案馆以开放许可发布的图像，每张图附馆方原文、整理后的规范字段、许可与证据、中英文出处注释，以及自动生成的标准裁剪图与分割 mask。

收录标准依中文维基「瓦當」：**屋檐最前端、挡在筒瓦头上的那块瓦**。軒平瓦 / 滴水、鬼瓦、屋脊装饰不收。全部图像经过逐张视觉复核与两轮外部版权复核；
权利或使用条件不明确的图、重复图、自动分割被判错误的图都已排除。

- **{s['n_images']} 张图 / {s['n_objects']} 件器物**，来自 {len(s['by_source_images'])} 个开放来源、{s['institutions']} 个收藏机构。
- 地区：{reg}。拓片 {s['by_modality'].get('rubbing', 0)} 张；带人脸 / 神面的 {s['has_face_true']} 张。
- 全部标签都是**待人工复核**（`qa_status = pending`）：地区、时代、母题只取馆方记录，馆方没写就记“未定”，不猜。

## 浏览与下载

| 想做什么 | 去哪里 |
|---|---|
| 在线浏览、筛选、看许可、复制出处 | 浏览网站（见本页链接；国际版：{c.get('space_url', '')}） |
| 下载全部原图 + Excel 目录 | `{c.get('zip_name', '')}`（本仓库文件；国际版：{zipurl}） |
| 全部字段表 | `metadata.csv`、`metadata.jsonl`、`catalog.xlsx` |
| 写程序读取 | `data/` 下的 Parquet：default（全部 {s['n_images']} 张）、faces（人脸类 {s['faces_config']} 张）、std（标准裁剪图 + mask {s['std_config']} 张） |

## 来源

| 来源 | 图数 | 器物数 |
|---|---:|---:|
{src}

## 许可（逐图不同）

每张图沿用来源给的许可，记录在 `license`、`license_url`、`attribution_text` 字段；需要署名的图请照 `attribution_text` 原样写全（也可看 `ATTRIBUTION.md`）。
通俗说明见《许可说明.md》。本数据集作者自己整理的元数据与标注为 CC0 1.0（见 `LICENSE-DATA` 开头的适用范围）。

| 许可 | 图数 |
|---|---:|
{lic}

## 如何引用

{c.get('author_zh', '')}. {c.get('dataset_title_zh', '')}（版本 {c.get('version', '')}）[DS/OL]. Hugging Face, {c.get('release_year', '')}. {c.get('hf_url', '')}.

引用其中一张图：用该图的 `citation_zh` 字段（GB/T 7714 格式），并照 `attribution_text` 写署名。详见《如何引用.md》。

## 已知局限

- 偏向开放许可的馆藏：欧美和日本的馆藏多，中国大陆博物馆的藏品几乎没有（其图像多未以开放许可发布）。
- 拓片少；地区未定的图占比高（馆方没写，不推断）。
- 倾角等几何字段只做过合成验证；自动分割由模型完成。
- 版权复核由 AI 完成，不是律师意见。如有侵权或错误，请联系上面的邮箱，核实后会在下一版改正或下架。
"""


from common import OFFICIAL_LICENSE as LIC_ZH_SHORT  # official licence names (E-36)  # noqa: E402
from provenance import credit  # noqa: E402


def readme_min(recs, c, hf_yaml: bool = True) -> str:
    """Chinese-only, minimal dataset card (author, PROGRESS E-35: say as little as possible, no English; E-38: only the images)."""
    from collections import Counter
    n_img, n_obj = len(recs), len({r["object_id"] for r in recs})
    n_std = sum(1 for r in recs if r.get("crop_file"))
    n_rect = sum(1 for r in recs if r.get("rect_file"))
    lic_link = f"{c.get('hf_url', '')}/blob/main/%E8%AE%B8%E5%8F%AF%E8%AF%B4%E6%98%8E.md"
    db = f"wadang_v{c.get('version', '')}.sqlite"  # E-87: catalogue database
    # E-66: links back to the GitHub mirror of the website and to the pipeline code (only when set in release_config.json)
    mirror = "".join(f"\n{label}：{c[k]}\n" for label, k in (("网页镜像", "github_site_url"), ("处理代码", "github_code_url")) if c.get(k))
    yaml = f"""---
license: other
license_name: mixed
license_link: "{lic_link}"
pretty_name: "{c.get('dataset_title_zh', '')}"
language:
- zh
size_categories:
- 1K<n<10K
task_categories:
- image-segmentation
tags:
- cultural-heritage
- archaeology
- roof-tile
- antefix
- wadang
configs:
- config_name: default
  data_files:
  - split: train
    path: data/default/*.parquet
- config_name: std
  data_files:
  - split: train
    path: data/std/*.parquet
- config_name: rect
  data_files:
  - split: train
    path: data/rect/*.parquet
---
""" if hf_yaml else ""
    return yaml + f"""**如有侵权或勘误，请联系：{c.get('contact_email', '')}**

学术图像资料，不代表任何政治立场，不作现代主权与领土归属判断。详见[声明](免责声明.md)。

# {c.get('dataset_title_zh', '')}（v{c.get('version', '')}）

古代瓦当的开放许可图像，共 {n_img} 张、{n_obj} 件，每张附自动分割结果（标准裁剪图与掩膜，可能有误）。
其中 {n_rect} 张另有拍正图：按瓦当边缘的椭圆（多为外缘，少数为内缘），把斜拍的图沿短轴拉回正圆（自动估计，可能不准）。
除程序自动归类的“母题类别”（可能有误）、AI 自动判断的“正面”（`frontal` 字段，是否拍到瓦当正面，不保证准确）和目录数据库里的自动测量值外，不作任何标注。

浏览：{c.get('space_url', '')}
{mirror}
程序与智能体下载说明：{c.get('site_url', '')}llms.txt

| 文件 | 内容 |
|---|---|
| `{c.get('seg_zip_name', '')}` | 一键下载：全部分割结果（标准裁剪图与掩膜） |
| `{c.get('seg_front_zip_name', '')}` | 同上，只含正面图（AI 自动判断，不保证准确） |
| `{c.get('rect_zip_name', '')}` | 拍正图与拍正掩膜，{n_rect} 张（自动估计，可能不准） |
| `files.csv` | 每行一张图，附署名与下载链接 |
| `metadata.jsonl`、`metadata.csv` | 同上，供程序读取 |
| `{db}` | 目录数据库（`SQLite`），见下 |
| `std/` | 分割结果文件夹，{n_std} 张（自动生成，可能有误） |
| `rect/` | 拍正图文件夹，{n_rect} 张（自动估计，可能不准） |
| `data/std`、`data/rect`、`data/default` | 分割结果、拍正图、原图（Parquet）：`datasets.load_dataset("{c.get('hf_repo', '')}", "std")`，拍正图用 `"rect"` |

掩膜为 1024×1024 灰度 PNG：0 = 背景，255 = 瓦当，128 = 遮挡物（少数图才有），与同名标准裁剪图逐像素对应；拍正掩膜与拍正图逐像素对应。同一件器物可能有多张照片（器物编号相同），划分训练集与测试集时请按器物编号分组。

许可要求署名的图须照原样署名：原图见“attribution_text”字段，分割结果与拍正图见“attribution_text_derived”字段，也可查 [ATTRIBUTION.md](ATTRIBUTION.md)；字段为空的图不要求署名。详见[许可说明](许可说明.md)。

## 目录数据库

`{db}` 初步具备关系型数据库的功能：来源（`sources`）、许可（`licences`）、收藏机构（`institutions`）、器物（`objects`）、图像（`images`）五张表以编号相互关联，收藏机构、馆藏号、母题建有索引，视图 `catalogue` 把常用的列合为一张；每一列的含义与性质（编号、来源给出、自动生成）见库内的 `fields` 表。可用任何 `SQLite` 工具打开，例如：

```sql
SELECT image_id, institution, accession_number, source_url FROM catalogue WHERE object_id = 'obj_000057';
```

收藏机构、馆藏号、来源页面为来源给出的原样，直径从来源给出的尺寸读出；母题、完整度、边缘弧度、倾角为程序自动生成，可能有误。不收来源给出的时代、地区、出土地、题名、材质与说明文字，需要时请按来源页面查看。网站首页可按馆藏号、收藏机构、图像编号查找。
"""


def main():
    setup_console()
    recs = read_jsonl(ROOT / "metadata.jsonl")
    c = cfg()
    n_img, n_obj = len(recs), len({r["object_id"] for r in recs})
    wid = wlog("Stage7", "数据卡（纯中文极简，PROGRESS E-35）、ATTRIBUTION、CHANGELOG", "python -I scripts/build_card.py", n_img,
               f"{n_img} 张 / {n_obj} 件")
    (ROOT / "README.md").write_text(readme_min(recs, c), encoding="utf-8", newline="\n")
    (ROOT / "README_zh.md").unlink(missing_ok=True)  # the card itself is Chinese now
    att = [r for r in recs if r["attribution_required"]]
    lines = ["# 需要署名的图", "", f"共 {len(att)} 张。用原图时照“原图署名”，用分割结果或拍正图时照“分割结果、拍正图署名”，原样照写；其余图不要求署名。", "",
             "| 图像编号 | 原图署名 | 分割结果、拍正图署名 |", "|---|---|---|"]
    lines += [f"| {r['image_id']} | {credit(r).replace('|', '/')} | {credit(r, True).replace('|', '/')} |" for r in att]
    (ROOT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    rd, ud = c.get("release_date", ""), c.get("updated_date", "")
    n_rect = sum(1 for r in recs if r.get("rect_file"))
    (ROOT / "CHANGELOG.md").write_text(
        f"# 更新记录\n\n## v{c.get('version', '')}\n\n- {rd}：首个公开版本。\n"
        + (f"- {ud}：增加拍正图（{n_rect} 张）。\n" if ud and ud != rd else "")
        + "- 2026-11-26：增加目录数据库（SQLite）与网站查找；单件页列出收藏机构、馆藏号与馆方页面；改正拍正图的说明（多为外缘，少数为内缘）。图像与表格文件未变。\n"  # E-87 / E-89
        + f"- 现有 {n_img} 张图、{n_obj} 件。\n", encoding="utf-8", newline="\n")
    print("card ok", wid)


def croissant(recs, s, c):
    fields = [("image_id", "sc:Text"), ("object_id", "sc:Text"), ("file_name", "sc:Text"), ("object_type", "sc:Text"),
              ("motif", "sc:Text"), ("motif_raw", "sc:Text"), ("has_face", "sc:Boolean"), ("region_group", "sc:Text"),
              ("region_code", "sc:Text"), ("period_code", "sc:Text"), ("date_from", "sc:Integer"), ("date_to", "sc:Integer"),
              ("holding_institution", "sc:Text"), ("accession_number", "sc:Text"), ("modality", "sc:Text"),
              ("label_source", "sc:Text"), ("qa_status", "sc:Text"), ("source", "sc:Text"), ("source_url", "sc:URL"),
              ("license", "sc:Text"), ("license_url", "sc:URL"), ("attribution_text", "sc:Text"), ("sha256", "sc:Text"),
              ("diameter_cm", "sc:Float"), ("tilt_deg", "sc:Float"), ("completeness_ratio", "sc:Float")]
    doc = {
        "@context": {"@language": "en", "@vocab": "https://schema.org/", "citeAs": "cr:citeAs", "column": "cr:column",
                     "conformsTo": "dct:conformsTo", "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
                     "data": {"@id": "cr:data", "@type": "@json"}, "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
                     "dct": "http://purl.org/dc/terms/", "examples": {"@id": "cr:examples", "@type": "@json"},
                     "extract": "cr:extract", "field": "cr:field", "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject",
                     "fileSet": "cr:fileSet", "format": "cr:format", "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset",
                     "jsonPath": "cr:jsonPath", "key": "cr:key", "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path",
                     "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex", "repeated": "cr:repeated",
                     "replace": "cr:replace", "sc": "https://schema.org/", "separator": "cr:separator", "source": "cr:source",
                     "subField": "cr:subField", "transform": "cr:transform"},
        "@type": "sc:Dataset",
        "name": "open-roof-tile-end-image-dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "description": f"{s['n_images']} open-licence images of {s['n_objects']} ancient roof-tile ends (wadang, antefixes) with per-image provenance and licence (mixed licences per item).",
        "license": [LICENSES[k]["url"] for k in LICENSES if s["by_license"].get(k)],
        "url": c.get("hf_url", "https://huggingface.co/"),
        "version": "0.1.0",
        "citeAs": f"{c.get('author_en', 'TODO')} (2026). {c.get('dataset_title_en', 'Open Roof-Tile End Image Dataset')} v0.1.0. "
                  + (c.get("hf_url", "") if "TODO" in c.get("zenodo_version_doi", "TODO") else f"doi:{c['zenodo_version_doi']}"),
        "distribution": [
            {"@type": "cr:FileObject", "@id": "metadata.jsonl", "name": "metadata.jsonl", "contentUrl": "metadata.jsonl",
             "encodingFormat": "application/jsonlines", "sha256": __import__("hashlib").sha256((ROOT / "metadata.jsonl").read_bytes()).hexdigest()},
            {"@type": "cr:FileSet", "@id": "images", "name": "images", "description": "original image files",
             "encodingFormat": "image/jpeg", "includes": "images/*"},
        ],
        "recordSet": [{
            "@type": "cr:RecordSet", "@id": "items", "name": "items", "key": {"@id": "items/image_id"},
            "field": [{"@type": "cr:Field", "@id": f"items/{n}", "name": n, "dataType": t,
                       "source": {"fileObject": {"@id": "metadata.jsonl"}, "extract": {"column": n}}} for n, t in fields],
        }],
    }
    (ROOT / "croissant.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
