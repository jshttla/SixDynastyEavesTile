"""reports/tilt_validation.md from the cloud outputs (tilt_validation.json + rim.csv). 方案 Stage 5.7 / 验收表."""
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPORTS, ROOT, setup_console, utcnow, wlog  # noqa: E402

setup_console()
d = json.load(open(REPORTS / "tilt_validation.json", encoding="utf-8"))
rim = list(csv.DictReader(open(ROOT / "rim.csv", encoding="utf-8")))
trials = d["trials"]
by = defaultdict(list)
for t in trials:
    by[t["phi"]].append(t)
L = ["# 倾角估计与拍正验证（v0.1.0：只有合成验证）", "",
     f"生成时间 {utcnow()}。数据：`reports/tilt_validation.json`、`rim.csv`。方法见 SCHEMA.md §5 与 PROGRESS E-14。", "",
     "## 表 1　合成仿射倾斜下的椭圆估计器（倾角误差与拍正后内部标记间距偏差）", "",
     f"取 {d['n_images']} 张完整、正面的公开图（rim_arc_deg = 360、b/a ≥ 0.98）。先用每张图实际所用的边（内缘或外缘）的拟合椭圆把它\"圆化\"，"
     "再按已知倾角 φ 和三个方向 ψ（0° 偏航、90° 俯仰、37° 斜向）做仿射压缩，在倾斜图上重新提取亚像素边缘并跑同一个估计器。"
     "间距偏差：在器物 0.8 半径内随机取 40 个标记点，拍正后两两距离与原距离之比（去掉全局尺度后）的 |比值 − 1|。", "",
     "| φ | 试验数 | 成功 | 平均 \\|误差\\| ° | 最大 \\|误差\\| ° | 间距偏差中位数 | 间距偏差 P95 | 判定（φ≤20°：误差≤2°、间距≤2%） |",
     "|---|---|---|---|---|---|---|---|"]
verdict_all = True
for phi in sorted(by):
    ok = [t for t in by[phi] if t.get("ok")]
    errs = [abs(t["err"]) for t in ok]
    sd = [t["spacing_dev_median"] for t in ok]
    sp = [t["spacing_dev_p95"] for t in ok]
    good = bool(ok) and max(errs) <= 2 and max(sp) <= 0.02
    if phi <= 20:
        verdict_all &= good
    L.append(f"| {phi}° | {len(by[phi])} | {len(ok)} | {st.mean(errs):.3f} | {max(errs):.3f} | {st.median(sd) * 100:.3f}% | "
             f"{max(sp) * 100:.3f}% | {'通过' if good else '未通过'}{'' if phi <= 20 else '（φ>20° 不在验收线内）'} |" if ok else
             f"| {phi}° | {len(by[phi])} | 0 | – | – | – | – | 未通过 |")
ba = d.get("frontal_complete_all_b_over_a", [])
L += ["", "按所用的边拆分（成功试验）：", "", "| 边 | φ | 试验数 | 最大 \\|误差\\| ° | 平均 \\|误差\\| ° |", "|---|---|---|---|---|"]
by_edge = defaultdict(list)
for t in trials:
    if t.get("ok"):
        by_edge[(t["edge"], t["phi"])].append(abs(t["err"]))
for (edge, phi), v in sorted(by_edge.items()):
    L.append(f"| {edge} | {phi}° | {len(v)} | {max(v):.3f} | {st.mean(v):.3f} |")
lost = [t for t in trials if not t.get("ok")]
L.append(f"\n失败（倾斜后找不到内缘）：{len(lost)} 次：" + "，".join(f"{t['image_id']} φ={t['phi']} ψ={t['psi']}" for t in lost))
r1 = REPORTS / "tilt_validation_run1.json"
if r1.exists():
    d1 = json.load(open(r1, encoding="utf-8"))
    def summ(tr, phi):
        ok = [t for t in tr if t["phi"] == phi and t.get("ok")]
        return (len([t for t in tr if t["phi"] == phi]), len(ok), max(abs(t["err"]) for t in ok) if ok else float("nan"))
    L += ["", "### 估计器 v1（无内缘自检）与 v2（有内缘自检）对比", "",
          "v1 在第一次验证中暴露出内缘检测会锁定装饰环（见 WORKLOG），v2 加了自检（PROGRESS E-19）。两次用的是同一套合成方案；"
          "v2 的图集由 v2 自己的正面分析选出，与 v1 不完全相同。自检参数没有为了通过验收再做调整。", "",
          "| φ | v1 试验 / 成功 / 最大误差 | v2 试验 / 成功 / 最大误差 |", "|---|---|---|"]
    for phi in sorted(by):
        a, b = summ(d1["trials"], phi), summ(trials, phi)
        L.append(f"| {phi}° | {a[0]} / {a[1]} / {a[2]:.2f}° | {b[0]} / {b[1]} / {b[2]:.2f}° |")
L += ["", f"**合成验证结论：φ ≤ 20° 的试验{'全部满足' if verdict_all else '未全部满足'}验收线（误差 ≤ 2°，间距偏差 ≤ 2%）。**", "",
      "这只说明估计器在**平面图形**上可靠。合成图既没有侧壁，也没有厚度，更没有透视，所以下面这些情况都没有被检验：", "",
      f"1. **器物本身不是正圆。** 本数据中完整正面件的外缘 b/a 分布：n = {len(ba)}，中位数 {st.median(ba) if ba else float('nan'):.4f}，"
      f"最小 {min(ba) if ba else float('nan'):.4f}。b/a = 0.99 对应 arccos = {math.degrees(math.acos(0.99)):.1f}°，所以在真实照片上，"
      "小倾角（< 10°）的估计会被器物自身的椭圆度淹没。这是精度下限，合成验证无法消除。",
      "2. **厚度和侧壁。** 倾斜时外轮廓会混入侧壁（直径 15 cm、厚 2.5 cm、倾斜 20° 时，外轮廓的 b/a 约为 0.997），所以优先用边轮内缘（rim_edge_used = inner）。内缘能否检出取决于纹样，厚度偏差今晚无法验证。",
      "3. **透视。** 照片没有 EXIF 焦距，只能做弱透视（仿射）近似。", "",
      "## 表 2　真实公开图上两个估计器的一致性", ""]
both = [r for r in rim if r.get("ok") == "1" and r.get("tilt_deg_ellipse") and r.get("tilt_deg_normal") and r.get("shape") == "round"]
agree = [r for r in both if abs(float(r["tilt_deg_ellipse"]) - float(r["tilt_deg_normal"])) <= 5]
from collections import Counter
src = Counter(r.get("tilt_source") or "none" for r in rim if r.get("ok") == "1")
shape = Counter(r.get("shape") or "?" for r in rim if r.get("ok") == "1")
L += ["| 项 | 数值 |", "|---|---|",
      f"| 几何分析成功的图 | {sum(1 for r in rim if r.get('ok') == '1')} / {len(rim)} |",
      f"| 形状判定 | {dict(shape)} |",
      f"| 两种估计都有的圆形件 | {len(both)} |",
      f"| 其中相差 ≤ 5° | {len(agree)}（{len(agree) / len(both) * 100 if both else 0:.0f}%） |",
      f"| 椭圆法 − 法线法 差值中位数 | {st.median([float(r['tilt_deg_ellipse']) - float(r['tilt_deg_normal']) for r in both]) if both else float('nan'):.1f}° |",
      f"| 发布的 tilt_source 分布 | {dict(src)} |", "",
      "差值的符号与大小说明：博物馆照片多数接近正面拍摄，法线法给出的倾角很小，而椭圆法常被残缺、背面照和器物自身的椭圆度推高。"
      "按方案规则，相差 > 5° 时以椭圆法为准，并列入 `reports/tilt_disagreements.csv` 交人工复核。"
      "两个估计器之间的一致率不等于准确率，物理验证留待后续版本。", "",
      "## 发布规则（本版）", "",
      "- `tilt_deg` 发布，同时 `tilt_validation = synthetic_only`；非圆形器物不发布（PROGRESS E-14）。",
      "- `rectified_file` 本版全部为 null，拍正图要等物理验证完成后的版本再发布。"]
(REPORTS / "tilt_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
dis = [r for r in both if abs(float(r["tilt_deg_ellipse"]) - float(r["tilt_deg_normal"])) > 5]
with open(REPORTS / "tilt_disagreements.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_id", "tilt_deg_ellipse", "tilt_deg_normal", "rim_edge_used", "rim_arc_deg", "overlay"])
    for r in dis:
        w.writerow([r["image_id"], r["tilt_deg_ellipse"], r["tilt_deg_normal"], r["rim_edge_used"], r["rim_arc_deg"], f"qa/masks_*.jpg 中的 {r['image_id']}"])
wlog("Stage5", "写 reports/tilt_validation.md（表 1 合成验证；表 2 真实图两估计器一致性）与 reports/tilt_disagreements.csv",
     "python -I scripts/tilt_report.py", len(trials),
     f"φ≤20° 验收{'通过' if verdict_all else '未通过'}；真实图两法都有 {len(both)}，一致 {len(agree)}；分歧 {len(dis)} 条进复核")
print("ok", verdict_all, len(both), len(agree))
