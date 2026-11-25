"""Stage 1 collector: Harvard-Yenching rubbings via Harvard LibraryCloud — ACCESS PROBE.

2026-11-08: every Harvard API / image host (api.lib.harvard.edu, ids.lib.harvard.edu, iiif.lib.harvard.edu,
nrs.harvard.edu, curiosity.lib.harvard.edu) answered HTTP 403 from the build machine (AWS ELB block page).
We do not work around access controls, so this source is skipped for v0.1. This script only re-probes and logs,
so the author can re-run it later from another network:

    python -I scripts/collect_harvard.py

If the probe succeeds, a full collector still has to be written (rights field per item: keep only
"Public Domain" / "No Known Copyright"; exclude Rübel and other permission-required collections; modality=rubbing).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Http, setup_console, wlog  # noqa: E402

PROBES = [
    "https://api.lib.harvard.edu/v2/items.json?q=%E7%93%A6%E7%95%B6&limit=1",
    "https://api.lib.harvard.edu/v2/items.json?q=wadang&limit=1",
    "https://api.lib.harvard.edu/v2/items.json?q=tile%20end%20rubbing&limit=1",
    "https://iiif.lib.harvard.edu/manifests/",
    "https://ids.lib.harvard.edu/ids/iiif/",
    "https://curiosity.lib.harvard.edu/",
]


def main():
    setup_console()
    h = Http(rate_per_sec=1)
    res = []
    for u in PROBES:
        try:
            r = h.get(u, tries=1)
            res.append((u, r.status_code, r.headers.get("server", "")))
        except Exception as e:
            res.append((u, f"ERR {type(e).__name__}", ""))
    ok = any(isinstance(c, int) and c == 200 for u, c, _ in res[:3])
    detail = "; ".join(f"{u} -> {c} {s}" for u, c, s in res)
    wlog("Stage1·harvard", "哈佛燕京拓片：探测 LibraryCloud API 与 IIIF/IDS 图像主机", "python -I scripts/collect_harvard.py",
         f"{sum(1 for _, c, _ in res if c == 200)}/{len(res)} 个端点返回 200",
         ("API 可达，需要补写完整采集器" if ok else
          "全部 403（AWS ELB 拦截页），不绕过访问控制，v0.1 跳过本来源；覆盖表中哈佛记 0，并在已知局限里说明"),
         extra=f"- 明细：{detail}")
    print(detail)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
