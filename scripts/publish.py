"""Morning publication helper — creates PRIVATE / DRAFT destinations only. It never makes anything public.

Tokens are read from environment variables (never stored in the repo):
  HF_TOKEN, ZENODO_TOKEN, ZENODO_SANDBOX_TOKEN, GITHUB_TOKEN

  python -I scripts/publish.py --checksums                      # dist/SHA256SUMS.txt
  python -I scripts/publish.py --hf  USER/wadang [--with-images] # private HF dataset repo + upload
  python -I scripts/publish.py --zenodo --sandbox               # rehearsal on sandbox.zenodo.org (draft)
  python -I scripts/publish.py --zenodo                         # real Zenodo DRAFT with reserved DOI (not published)
  python -I scripts/publish.py --space USER/wadang              # private static HF Space with the website (docs/)
  python -I scripts/publish.py --make-public dataset:USER/wadang # ONLY after the author confirms (also space:USER/wadang)
  (--github is legacy: the author decided not to use GitHub for this project, PROGRESS E-32)

Making things public (HF visibility, Zenodo "Publish", GitHub visibility/Pages) is the AUTHOR's manual step (方案 §1.9, §9.10).
evidence/ (full web-page snapshots of licence pages) is INTERNAL audit material and is never uploaded: the pages carry their own
text licences (e.g. Commons policy text is CC BY-SA 4.0; external copyright review round 2, PROGRESS E-30).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, setup_console, wlog  # noqa: E402

DIST = ROOT / "dist"
HF_FILES = ["README.md", "DATASHEET.md", "SCHEMA.md", "CHANGELOG.md", "CITATION.cff", "croissant.json", "LICENSE",
            "LICENSE-DATA", "ATTRIBUTION.md", "THIRD_PARTY.md", "metadata.jsonl", "metadata.csv", "catalog.csv",
            "如何引用.md", "许可说明.md", "数据来源与可信度.md"]


def need(var):
    v = os.environ.get(var)
    if not v:
        sys.exit(f"missing environment variable {var}")
    return v


def checksums():
    files = [p for p in [DIST / "wadang_v0.1.0_images.zip", ROOT / "metadata.jsonl", ROOT / "catalog.xlsx", ROOT / "catalog.csv"] if p.exists()]
    lines = []
    for p in files:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for ch in iter(lambda: f.read(1 << 20), b""):
                h.update(ch)
        lines.append(f"{h.hexdigest()}  {p.name}")
    (DIST / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def hf(repo: str, with_images: bool):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)  # else the token the author stored with `hf auth login`
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo, repo_type="dataset")
    assert info.private, "repo is not private — refusing to upload"
    # dist/hf/ is the complete, self-consistent upload folder (data/, card, linked docs, LICENSES/, vocab/) written by
    # build_hf.py; uploading it as one folder keeps the card and the files it links in step (recheck 2026-11-09)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(DIST / "hf"), delete_patterns=["*", "**/*"])  # drop stale files
    if with_images:
        # 2026-11-10: images/ also holds every excluded / deferred image (2738 files vs 1236 published); uploading it
        # would publish what the reviews removed. The originals are already in data/default (Parquet). Never do this.
        raise SystemExit("--with-images is disabled: images/ contains excluded images; originals ship in data/default")
    wlog("Stage7", f"上传到 Hugging Face 私有数据集仓库 {repo}（private=True 已核实）", f"python -I scripts/publish.py --hf {repo}", "",
         "仓库保持私有；公开由作者手动操作")


def zenodo(sandbox: bool):
    import requests
    base = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
    tok = need("ZENODO_SANDBOX_TOKEN" if sandbox else "ZENODO_TOKEN")
    meta = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    meta.pop("access_right", None)
    meta["prereserve_doi"] = True
    r = requests.post(f"{base}/deposit/depositions", params={"access_token": tok}, json={"metadata": meta}, timeout=60)
    r.raise_for_status()
    dep = r.json()
    bucket = dep["links"]["bucket"]
    for p in [DIST / "wadang_v0.1.0_images.zip", ROOT / "metadata.jsonl", ROOT / "catalog.xlsx", DIST / "SHA256SUMS.txt"]:
        if p.exists():
            with open(p, "rb") as f:
                requests.put(f"{bucket}/{p.name}", data=f, params={"access_token": tok}, timeout=None).raise_for_status()
    doi = dep["metadata"].get("prereserve_doi", {}).get("doi")
    print("DRAFT deposition", dep["id"], "reserved DOI", doi, "— NOT published")
    if not sandbox and doi:
        cfgp = ROOT / "release_config.json"
        cfg = json.loads(cfgp.read_text(encoding="utf-8"))
        cfg["zenodo_version_doi"] = doi
        cfg["zenodo_url"] = f"https://doi.org/{doi}"
        cfgp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    wlog("Stage7", f"Zenodo {'sandbox ' if sandbox else ''}草稿（预留 DOI，未发布）", f"python -I scripts/publish.py --zenodo{' --sandbox' if sandbox else ''}",
         "", f"deposition {dep['id']}，预留 DOI {doi}；发布按钮由作者手动点击")


def github(repo: str):
    import requests
    tok = need("GITHUB_TOKEN")
    owner, name = repo.split("/")
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}
    r = requests.post("https://api.github.com/user/repos", headers=h, json={"name": name, "private": True}, timeout=60)
    if r.status_code not in (201, 422):
        r.raise_for_status()
    info = requests.get(f"https://api.github.com/repos/{repo}", headers=h, timeout=60).json()
    assert info.get("private") is True, "GitHub repo is not private — refusing to push"
    url = f"https://x-access-token:{tok}@github.com/{repo}.git"
    subprocess.run(["git", "push", url, "main"], cwd=ROOT, check=True)
    wlog("Stage7", f"推送到 GitHub 私有仓库 {repo}", f"python -I scripts/publish.py --github {repo}", "", "仓库私有；GitHub Pages 需仓库公开（或付费账号）后再开启")


SPACE_README = """---
title: 瓦当开放图像数据集
emoji: 🏯
colorFrom: yellow
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: other
---

**如有侵权或勘误，请联系：{email}**

学术图像资料，不代表任何政治立场，不作现代主权与领土归属判断（见网站“声明”页）。

{title_zh}：{n_img} 张图、{n_obj} 件，按母题类别浏览（自动归类，可能有误）。数据集：{hf_url}
"""


def stage_space() -> Path:
    """dist/space/ = site/ (hard links; Chinese-only minimal site, PROGRESS E-35) + a Space card with `sdk: static`."""
    out = DIST / "space"
    if out.exists():
        shutil.rmtree(out)
    for p in (ROOT / "site").rglob("*"):
        if p.is_file():
            q = out / p.relative_to(ROOT / "site")
            q.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(p, q)
            except OSError:
                shutil.copy2(p, q)
    cfg = json.loads((ROOT / "release_config.json").read_text(encoding="utf-8"))
    recs = [json.loads(x) for x in (ROOT / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    (out / "README.md").write_text(SPACE_README.format(
        email=cfg["contact_email"], title_zh=cfg["dataset_title_zh"], title_en=cfg["dataset_title_en"], version=cfg["version"],
        n_img=len(recs), n_obj=len({r["object_id"] for r in recs}), hf_url=cfg["hf_url"]), encoding="utf-8", newline="\n")
    n = sum(1 for p in out.rglob("*") if p.is_file())
    print("staged", out, n, "files")
    return out


def space(repo: str):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    out = stage_space()
    api.create_repo(repo, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    assert api.repo_info(repo, repo_type="space").private, "space is not private — refusing to upload"
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(out), delete_patterns=["*", "**/*"])  # drop stale files
    wlog("Stage7", f"上传网站到 Hugging Face 私有静态 Space {repo}（private=True 已核实）", f"python -I scripts/publish.py --space {repo}", "",
         "Space 保持私有；公开由作者确认后进行")


def update_public(repo: str, repo_type: str):
    """Replace the content of an already public repository with the current build (author-approved updates only,
    e.g. PROGRESS E-55), then squash the history to ONE commit: the public repository never keeps earlier versions
    (author 2026-11-11: 不保留 commit 记录，一次就是最终版本). The folder is the same one the private upload uses."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    cfg = json.loads((ROOT / "release_config.json").read_text(encoding="utf-8"))
    folder = DIST / "hf" if repo_type == "dataset" else stage_space()
    api.upload_folder(repo_id=repo, repo_type=repo_type, folder_path=str(folder), delete_patterns=["*", "**/*"],
                      commit_message=f"{cfg['dataset_title_zh']} v{cfg['version']}")
    api.super_squash_history(repo, repo_type=repo_type, commit_message=f"{cfg['dataset_title_zh']} v{cfg['version']}")
    n = len(api.list_repo_commits(repo, repo_type=repo_type))
    assert n == 1, f"{repo_type} {repo}: {n} commits after squash"
    wlog("Stage7", f"更新已公开的 {repo_type} {repo} 并把历史压成一次提交（作者同意的更新）",
         f"python -I scripts/publish.py --update-public {repo_type}:{repo}", "", f"提交数 {n}")
    print(repo_type, repo, "updated; commits:", n)


def make_public(repo: str, repo_type: str):
    """Only after the author has explicitly confirmed (PROGRESS E-32)."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.update_repo_settings(repo, private=False, repo_type=repo_type)
    wlog("Stage7", f"作者确认后将 {repo_type} {repo} 设为公开", f"python -I scripts/publish.py --make-public {repo_type}:{repo}", "", "")


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--checksums", action="store_true")
    ap.add_argument("--hf")
    ap.add_argument("--with-images", action="store_true")
    ap.add_argument("--zenodo", action="store_true")
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--github")
    ap.add_argument("--space", help="USER/NAME: private static Space with the website (docs/)")
    ap.add_argument("--stage-space", action="store_true", help="only build dist/space/ (offline)")
    ap.add_argument("--make-public", help="dataset:USER/NAME or space:USER/NAME — only after the author confirms")
    ap.add_argument("--update-public", help="dataset:USER/NAME or space:USER/NAME — replace a public repo's content, then squash to one commit")
    a = ap.parse_args()
    if a.stage_space:
        stage_space()
    if a.space:
        space(a.space)
    if a.make_public:
        kind, repo = a.make_public.split(":", 1)
        make_public(repo, kind)
    if a.update_public:
        kind, repo = a.update_public.split(":", 1)
        update_public(repo, kind)
    if a.checksums:
        checksums()
    if a.hf:
        hf(a.hf, a.with_images)
    if a.zenodo:
        zenodo(a.sandbox)
    if a.github:
        github(a.github)


if __name__ == "__main__":
    main()
