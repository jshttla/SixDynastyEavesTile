"""Attribution lines exactly in the form each licence's own text asks for, in that text's own language
(author, PROGRESS E-40, 2026-11-10: "官方咋署名你咋署名；是什么文就用什么文"). Two lines per image:
  original  — for the unmodified original image
  derived   — for the standard crop and the mask made from it (edited / adapted material)
Sources (archived in evidence/):
  ColBase 利用規約 (colbase_terms_ja_2026-11-08.json), 1. 出典の記載について:
    出典記載例「出典：国立文化財機構所蔵品統合検索システム（当該ページのURL）」「出典：ColBase（https://colbase.nich.go.jp/）」;
    編集・加工等して利用する場合の記載例「「ColBase」（当該ページのURL）をもとに○○作成」「「国立文化財機構所蔵品統合検索システム」（当該ページのURL）を加工して作成」.
    -> original 「出典：ColBase（当該ページのURL）」, derived 「「ColBase」（当該ページのURL）を加工して作成」 (Japanese only).
  공공누리 제1유형 (kogl_licenseType1_2026-11-07.html), 출처 표시 의무 1:
    "본 저작물은 'OOO(기관명)'에서 'OO년' 작성하여 공공누리 제O유형으로 개방한 '저작물명(작성자:OOO)'을 이용하였으며,
     해당 저작물은 'OOO(기관명), OOO(홈페이지 주소)'에서 무료로 다운받으실 수 있습니다." (Korean only; the same sentence for
    changed works — Type 1 allows 2차적 저작물, no separate wording is given; the year is left out because e뮤지엄 gives none).
  CC BY 4.0 s.3(a)(1) / CC BY 3.0 s.4(b): creator, title (3.0: if supplied), licence name + URI, URI of the material;
    for adapted material an indication of the modification (3.0 example: "French translation of the Work by Original
    Author"). The licences and the Commons / Europeana pages are English -> English only.
"""
from __future__ import annotations

import re

COLBASE = re.compile(r"^出典：ColBase（https://colbase\.nich\.go\.jp/）(?P<url>https://colbase\.nich\.go\.jp/collection_items/\S+?)"
                     r"(?:；撮影：(?P<ph>[^；]+))?$")
KOGL = re.compile(r"^본 저작물은 '(?P<org>[^']+)'에서 (?P<yr>'\d{4}년' 작성하여 )?공공누리 제1유형으로 개방한 '(?P<title>.+)\(작성자:(?P<au>[^)]+)\)'을 "
                  r"이용하였으며, 해당 저작물은 '(?P<org2>[^']+), e뮤지엄\((?P<url>https?://[^)]+)\)'에서 무료로 다운받으실 수 있습니다\.$")
CC_URL = {"CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/", "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/"}
DERIVED_EN = "Cropped, resized and automatically segmented (crop and mask) from: "
# E-55: an image that also has a rectified crop and mask says so (CC BY: indicate the modifications)
DERIVED_RECT_EN = ("Cropped, resized, automatically segmented and geometrically rectified "
                   "(crop, mask, rectified crop and rectified mask) from: ")


def official(r: dict, staged: str | None, has_rect: bool = False) -> tuple[str | None, str | None]:
    """(original, derived) attribution for one record; (None, None) if the licence asks for none.
    `staged` is the credit the collector built from the source page (provider names, title, photographer).
    The derived line is for every file made from the original: crop, mask and, where they exist, the rectified pair."""
    lic = r["license"]
    if lic == "ColBase-Terms":
        m = COLBASE.match(staged or "")
        assert m, (r.get("image_id"), staged)
        url, ph = m.group("url"), m.group("ph")
        tail = f"（撮影：{ph}）" if ph else ""  # the museum's photographer credit (E-29), kept in Japanese
        return f"出典：ColBase（{url}）{tail}", f"「ColBase」（{url}）を加工して作成{tail}"
    if lic == "KOGL-Type-1":
        m = KOGL.match(staged or "")
        assert m and m.group("org") == m.group("org2"), (r.get("image_id"), staged)
        yr = m.group("yr") or ""
        s = (f"본 저작물은 '{m.group('org')}'에서 {yr}공공누리 제1유형으로 개방한 '{m.group('title')}(작성자:{m.group('au')})'을 이용하였으며, "
             f"해당 저작물은 '{m.group('org')}, {m.group('url')}'에서 무료로 다운받으실 수 있습니다.")
        return s, s
    if lic in CC_URL:
        s = staged or ""
        assert s and not re.search(r"[；。]|std/", s), (r.get("image_id"), s)
        if r["source_url"] not in s:
            s = f"{s}, {r['source_url']}"
        if CC_URL[lic] not in s:
            s = f"{s}, {CC_URL[lic]}"
        return s, (DERIVED_RECT_EN if has_rect else DERIVED_EN) + s
    return None, None
