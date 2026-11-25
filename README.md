# 瓦当开放图像数据集 · 处理代码

本仓库是 [瓦当开放图像数据集](https://huggingface.co/datasets/jshttla/wadang) 的处理代码。数据文件挂在本仓库的 [Releases](https://github.com/jshttla/SixDynastyEavesTile/releases)（v0.1.0），与数据集页面上的相同；另有目录数据库 `wadang_v0.1.0.sqlite`（SQLite，Releases 与数据集页面都有），可按收藏机构、馆藏号、器物编号查询。数据不适用本仓库的 MIT 许可，图像的许可逐张不同。

如有侵权或勘误，请随时联系：sixdynastyliuchao@163.com

## 目录数据库

`wadang_v0.1.0.sqlite` 初步具备关系型数据库的功能：来源（sources）、许可（licences）、收藏机构（institutions）、器物（objects）、图像（images）五张表以编号相互关联（外键），收藏机构、馆藏号、母题建有索引，`catalogue` 视图把常用的列合为一张；每一列的含义与性质（编号、来源给出、自动生成）见库内的 `fields` 表。可用 SQL 按收藏机构、馆藏号、器物编号查询，例如：

```sql
SELECT image_id, institution, crop_file FROM catalogue WHERE accession_number = 'TJ-1547';
```

收藏机构、馆藏号、来源页面为来源给出的原样，直径从来源给出的尺寸读出；母题、完整度、边缘弧度、倾角为程序自动生成，可能有误。不收来源给出的时代、地区、出土地、题名、材质与说明文字。网站首页可按馆藏号、收藏机构、图像编号查找，单件页列出收藏机构、馆藏号与来源页面。

## 内容

| 目录 | 内容 |
|---|---|
| `scripts/collect_*.py` | 从开放馆藏接口采集记录与图像（Art Institute of Chicago、Cleveland Museum of Art、ColBase、Wikimedia Commons、e뮤지엄、Europeana、J. Paul Getty Museum、Harvard Library、The Metropolitan Museum of Art、Rijksmuseum、Smithsonian、Walters Art Museum） |
| `scripts/` 其余 | 去重候选、质量标记、署名文字、溯源信息、拍正、校验、数据卡与 Hugging Face 文件的生成 |
| `cloud/` | GPU 上运行的分割与几何估计，及其运行脚本 |
| `tests/` | 干净环境下的加载检查 |

本仓库不是完整可运行的流程：归类与网站生成部分、审核材料没有包含在内。

## 用到的开放模型

| 用途 | 模型 | 许可 |
|---|---|---|
| 去重候选 | facebook/dinov2-large | Apache-2.0 |
| 分割 | IDEA-Research/grounding-dino-base、SAM 2.1 Hiera-L、BiRefNet | Apache-2.0 / Apache-2.0 / MIT |
| 几何估计 | Ruicheng/moge-3-vitg | MIT |

没有训练或微调任何模型。

## 运行环境

- 本地：`requirements.txt`；云端 GPU：`cloud/requirements-cloud.txt`。
- 本机专用的路径写在 `local_settings.json`（模板见 `local_settings.example.json`）；云端登录信息写在 `cloud/cloud.env`（模板见 `cloud/cloud.env.example`）。这两个文件不进仓库。

## 许可

代码：MIT（见 `LICENSE`）。Releases 里的数据文件不适用 MIT：图像的许可逐张不同，见每个压缩包里的 `LICENSE_GUIDE.md` 或数据集页面的《许可说明》。

代码在 AI 编程助手协助下完成。

---

# Open Roof-Tile End Image Dataset · pipeline code

This repository holds the processing code of the [Open Roof-Tile End Image Dataset](https://huggingface.co/datasets/jshttla/wadang). The data files are attached to this repository's [Releases](https://github.com/jshttla/SixDynastyEavesTile/releases) (v0.1.0), identical to those on the dataset page; a catalogue database, `wadang_v0.1.0.sqlite` (SQLite, in Releases and on the dataset page), can be queried by holding institution, accession number and object ID. The data are not under this repository's MIT licence, and image licences differ per image.

For infringement reports or corrections, contact: sixdynastyliuchao@163.com

Catalogue database: `wadang_v0.1.0.sqlite` provides basic relational-database features. Five tables (sources, licences, institutions, objects, images) are linked by IDs (foreign keys), institutions, accession numbers and motifs are indexed, the `catalogue` view joins the common columns, and the `fields` table documents every column and whether it is an ID, given by the source or generated automatically. Institution names, accession numbers and source pages are as given by the sources; motifs, completeness, rim arc and tilt are generated automatically and may be wrong. The sources' period, region, findspot, title, material and description text are not included. The website's home page has a search box, and every item page lists the institution, accession number and source page.

- `scripts/collect_*.py`: collectors for open museum collection APIs (listed above).
- `scripts/` (other files): duplicate candidates, quality flags, attribution text, provenance, rectification, validation, dataset card and Hugging Face files.
- `cloud/`: segmentation (Grounding DINO + SAM 2.1 + BiRefNet) and geometry (MoGe-3) on a GPU, with their run scripts.
- `tests/`: a clean-environment loading check.

This is not a complete runnable pipeline: the classification and website modules and the review material are not included. No model was trained or fine-tuned.

Code licence: MIT (see `LICENSE`). The data files in Releases are not under MIT: image licences differ per image; see `LICENSE_GUIDE.md` in each ZIP or the licence notes on the dataset page. The code was written with the help of AI coding assistants.
