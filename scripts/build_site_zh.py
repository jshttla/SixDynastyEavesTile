"""Chinese-interface copy of the website for the domestic mirror (ModelScope Studio, Static SDK; author 2026-11-09, PROGRESS E-32).

Same pages, same data as docs/. Only the INTERFACE is made Chinese-only: English halves of bilingual labels, English
sentences of the guides and the English `.en` spans are removed. Museum verbatim text (titles, descriptions, credit lines,
notes copied from the source records) is NOT touched — it is data, not interface.
Output: dist/ms_space/ (images hard-linked). Usage: python -I scripts/build_site_zh.py
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, setup_console, wlog  # noqa: E402

SRC = ROOT / "docs"
OUT = ROOT / "dist" / "ms_space"
CJK = re.compile(r"[㐀-鿿가-힣぀-ヿ]")

# interface strings written by build_site.py: bilingual -> Chinese (exact, so museum text can never be hit)
PAIRS = [
    ("数据来源与可信度 Sources &amp; reliability", "数据来源与可信度"), ("数据来源与可信度 Sources and reliability", "数据来源与可信度"),
    ("许可说明 Licences in plain words", "许可说明"), ("一、图片从哪里来 Where the images come from", "一、图片从哪里来"),
    ("三、当前状态 Current status", "三、当前状态"), ("五、发现错误怎么办 Reporting errors", "五、发现错误怎么办"),
    ("四、已知偏差 Known biases", "四、已知偏差"), ("哪些是我们整理的 Which fields come from the museum", "哪些是我们整理的"),
    ("引用其中一张图 Citing a single image", "引用其中一张图"), ("引用整个数据集 Citing the whole dataset", "引用整个数据集"),
    ("单张原图 Single full-size image", "单张原图"), ("全部字段表 Full metadata", "全部字段表"), ("程序读取 Programmatic access", "程序读取"),
    ("下载 / 查看原图 Full-size image", "下载 / 查看原图"), ("署名怎么写 How to credit", "署名怎么写"), ("返回图库 Back to gallery", "返回图库"),
    ("收藏机构 Holding institution", "收藏机构"), ("物件名 / 分类 Object name", "物件名 / 分类"), ("文化 / 时代 Culture / period", "文化 / 时代"),
    ("出土地 / 地点 Place", "出土地 / 地点"), ("权利声明 Rights statement", "权利声明"), ("馆藏号 Accession no.", "馆藏号"),
    ("馆方条目页 Museum page", "馆方条目页"), ("馆方原文 Museum text", "馆方原文"), ("同尺度对照 Same-scale", "同尺度对照"),
    ("如何引用 How to cite", "如何引用"), ("许可说明 Licences", "许可说明"), ("数据来源与可信度 Sources", "数据来源与可信度"),
    ("推断 / 未定 Inferred", "推断 / 未定"), ("许可依据 Evidence", "许可依据"), ("自动测量 Measured", "自动测量"), ("规范化 Normalised", "规范化"),
    ("加载更多 Load more", "加载更多"), ("清空条件 Reset", "清空条件"), ("许可类型 Licence", "许可类型"), ("是否署名 Attribution", "是否署名"),
    ("论文插图 Paper", "论文插图"), ("网络传播 Web", "网络传播"), ("裁剪修改 Modify", "裁剪修改"), ("补充说明 Notes", "补充说明"),
    ("适合谁 For whom", "适合谁"), ("去哪里 Where", "去哪里"), ("内容 What", "内容"), ("出版 Publication", "出版"), ("讲义 Teaching", "讲义"),
    ("图库 Gallery", "图库"), ("下载 Download", "下载"), ("复制 Copy", "复制"), ("已复制 Copied", "已复制"), ("尺寸 Dimensions", "尺寸"),
    ("材质 Medium", "材质"), ("标题 Title", "标题"), ("说明 Description", "说明"), ("许可 Licence", "许可"), ("提醒 Notes", "提醒"),
    ("全部 all", "全部"), ("（空 empty）", "（空）"), ("有人脸 face", "有人脸"),
    # item pages (interface fragments around the museum text)
    ("）Full-size image</a>", "）</a>"), (" from museum title", ""), ("否 / 未写明 no / not stated", "否 / 未写明"),
    ("待人工复核 · pending human review", "待人工复核"), ("（馆方未提供 not given）", "（馆方未提供）"),
    ("未定 undetermined（待复核 pending）", "未定（待复核）"), ("馆方原词 museum wording：", "馆方原词："),
    ("由馆方标题关键词归类 · classified from a keyword in the museum title", "由馆方标题关键词归类"),
    ("推断 / 未定（馆方未写明） · inferred / undetermined (not stated by the museum)", "推断 / 未定（馆方未写明）"),
    ("由时代名推得 from period name", "由时代名推得"), ("原图 original：", "原图："), ("自动测量 automatic", "自动测量"),
    ("图像质量 image quality：", "图像质量："), (">是 yes<", ">是<"), ("器物 object ", "器物 "), ("馆方条目页 Museum page", "馆方条目页"),
]
LIC_NOTES: list[tuple[str, str]] = []  # licence plain-language notes (zh, en) read from build_site.py
SENTENCES = [  # English sentences of the page frame
    "Metadata CC0 1.0; each image keeps its own licence (see item page).",
    "All labels in this release are pending human review.",
]


def template_sentences() -> list[str]:
    """English sentences that follow a Chinese one inside build_site.py templates (interface text, never museum text)."""
    src = (ROOT / "scripts" / "build_site.py").read_text(encoding="utf-8")
    found = re.findall(r'[。）」]\s*([A-Z][A-Za-z0-9 ,;:\'’()/&.%×=+\-–]{15,}?[.?!])(?=["<\n]|\s*$)', src)
    out = []
    for x in SENTENCES + found:
        for v in (x, html.escape(x, quote=True), html.escape(x, quote=False), x.replace("'", "&#x27;"), x.replace("&", "&amp;")):
            if v not in out:
                out.append(v)
    return sorted(out, key=len, reverse=True)


def br_lines(t: str) -> str:
    """In a paragraph '中文…<br>English…': drop the <br>-separated parts that have no CJK and no link."""
    def para(m):
        toks = re.split(r"(<br\s*/?>|\n)", m.group(2))  # text, separator, text, ...
        out, sep = [], ""
        for i, tok in enumerate(toks):
            if i % 2:
                sep = tok
                continue
            if CJK.search(html.unescape(re.sub(r"<[^>]+>", "", tok))) or "http" in tok or not tok.strip():
                out.append((sep if out else "") + tok)
        return f"<p{m.group(1) or ''}>" + "".join(out).strip() + "</p>"
    return re.sub(r"<p(\s[^>]*)?>(.*?)</p>", para, t, flags=re.S)
SENTS: list[str] = []
UI_TAGS = r"label|th|h1|h2|h3|h4|button|option|dt|summary|figcaption"


def ui_label(m) -> str:
    """'中文 English' inside a label / heading / button: keep the Chinese part."""
    tag, attrs, inner = m.group(1), m.group(2) or "", m.group(3)
    if CJK.search(inner):
        inner = re.sub(r"(?<=[㐀-鿿）」》”])\s+[A-Za-z][A-Za-z0-9 &;'’/().,:-]*$", "", inner.strip())
    return f"<{tag}{attrs}>{inner}</{tag}>"


def zh_html(t: str, guide: bool, item: bool = False) -> str:
    t = re.sub(r"<span class=['\"]en['\"]>.*?</span>", "", t, flags=re.S)
    for a, b in PAIRS:
        t = t.replace(a, b)
    for snt in SENTS:
        t = t.replace(snt, "")
    t = re.sub(r"<small>[^<]*?(v\d+\.\d+\.\d+)</small>", r"<small>\1</small>", t)
    # '以下内容逐字引自 X 的记录，未作改动。Quoted verbatim from X.' / licence notes '中文。 English.' / badges '中文 · english'
    t = re.sub(r"(未作改动。)Quoted verbatim from [^<]*", r"\1", t)
    for zh_, en_ in LIC_NOTES:
        t = t.replace(f"{zh_} {en_}", zh_).replace(f"{zh_} {html.escape(en_)}", zh_)
    t = re.sub(r"(<span class=['\"]badge[^>]*>[^<]*?[㐀-鿿）])\s*·\s*[A-Za-z][^<]*(</span>)", r"\1\2", t)
    t = t.replace(" · </div>", "</div>")  # contact bar after its English half was removed
    t = re.sub(r"<p>\s*<strong>English summary\.</strong>.*?</p>", "", t, flags=re.S)  # disclaimer page: Chinese version only
    t = t.replace("声明与免责 Disclaimer", "声明与免责")
    t = re.sub(rf"<({UI_TAGS})(\s[^>]*)?>([^<]*)</\1>", ui_label, t)
    if not item:  # item pages carry museum verbatim text: only exact interface strings are removed there
        t = br_lines(t)
    if guide:  # guide pages: drop blocks without a single CJK character (English paraphrases)
        def drop(m):
            vis = html.unescape(re.sub(r"<[^>]+>", "", m.group(0)))
            return m.group(0) if (CJK.search(vis) or not vis.strip()) else ""
        t = re.sub(r"<(p|li|h[1-4])(\s[^>]*)?>.*?</\1>", drop, t, flags=re.S)
    t = re.sub(r"<br>\s*(?=</)", "", t)
    return t


def zh_js(name: str, t: str) -> str:
    for a, b in PAIRS:
        t = t.replace(a, b)
    if name == "app.js":
        t = t.replace("return v?v[0]+' '+v[1]:", "return v?v[0]:")
        t = t.replace("' 件器物 · '+res.length+' images / '+Object.keys(objs).length+' objects'", "' 件器物'")
    return t


def main():
    global SENTS
    SENTS = template_sentences()
    LIC_NOTES[:] = re.findall(r'note_zh="([^"]+)", note_en="([^"]+)"', (ROOT / "scripts" / "build_site.py").read_text(encoding="utf-8"))
    setup_console()
    if OUT.exists():
        shutil.rmtree(OUT)
    n_html = 0
    for p in SRC.rglob("*"):
        if not p.is_file():
            continue
        q = OUT / p.relative_to(SRC)
        q.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".html":
            q.write_text(zh_html(p.read_text(encoding="utf-8"), guide="guide" in p.parts, item="item" in p.parts), encoding="utf-8", newline="\n")
            n_html += 1
        elif p.suffix == ".js" and p.parent.name == "assets":
            q.write_text(zh_js(p.name, p.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
        elif p.suffix == ".css":
            q.write_text(p.read_text(encoding="utf-8") + "\n.en{display:none}\n", encoding="utf-8", newline="\n")
        elif p.name in ("sitemap.xml", "robots.txt"):
            continue  # the mirror's own address is not known before it exists
        else:
            try:
                os.link(p, q)
            except OSError:
                shutil.copy2(p, q)
    cfg = json.loads((ROOT / "release_config.json").read_text(encoding="utf-8"))
    recs = [json.loads(x) for x in (ROOT / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    (OUT / "README.md").write_text(f"""---
license: other
domain:
- cv
tags:
- 文化遗产
- 瓦当
---

**如有侵权或勘误，请随时联系：{cfg['contact_email']}**

本数据集为纯学术项目，遵守适用的法律法规，不含有、也不代表任何政治观点；馆方原文不代表维护者观点。详见网站“声明与免责”页。

# {cfg['dataset_title_zh']}（浏览网站 · 国内镜像）

{len(recs)} 张图 / {len({r['object_id'] for r in recs})} 件器物：各地区古代瓦当（中国瓦当、半瓦当、日本軒丸瓦、韩国수막새、地中海 antefix，含拓片与残片）的开放图像。
可按地区、时代、母题、收藏机构筛选；每件附馆方原文、许可说明与一键复制的出处注释。网站只放缩略图与预览图，原图链回收藏机构。

- 每张图沿用来源的许可（CC0、公有领域、CC BY、ColBase 利用规约、韩国 KOGL 第 1 类型），见各条目页与《许可说明》。
- 国际版（含英文）：{cfg.get('space_url', '')} ；数据集：{cfg.get('hf_url', '')}
""", encoding="utf-8", newline="\n")
    wlog("Stage7", "生成中文界面网站副本（国内镜像用，魔搭创空间 Static）：界面与说明只留中文，馆方原文不动",
         "python -I scripts/build_site_zh.py", n_html, f"dist/ms_space/，HTML {n_html} 页")
    print("zh site", OUT, "html", n_html)


if __name__ == "__main__":
    main()
