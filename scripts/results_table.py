"""Render the README results tables from results/*.json. With --write, replaces the text between
<!-- results:begin --> and <!-- results:end --> in README.md. Reads only committed JSON."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]


def j(name):
    p = Path("results") / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def ci(c):
    return f"[{pct(c[0])}, {pct(c[1])}]"


def gallery(n, method, kind, cause):
    return f"[{n}](docs/failures/{method}_{kind.replace(' ', '-')}_{cause.replace(' ', '-')}.jpg)"


def failures(f) -> list[str]:
    r, y = f["refdiff"], f["yolov8n"]
    out = [f"### Where the errors are (score ≥ 0.5; {r['n_gt']} labelled defects)", "",
           ("Every miss and false alarm, tagged with a cause by `scripts/failure_analysis.py`; "
           "its count links to a gallery (template | test | signed diff after registration; label green, detection red)."), ""]
    for kind, title in (("missed", "Missed defects"), ("false alarm", "False alarms")):
        causes = list(dict.fromkeys([*r[kind]["by_cause"], *y[kind]["by_cause"]]))
        out += [f"| {title} | refdiff | YOLOv8n |", "|---|---|---|"]
        for c in causes:
            cells = [(gallery(m[kind]["by_cause"][c], n, kind, c) if c in m[kind]["by_cause"] else "n/a")
                     for n, m in (("refdiff", r), ("yolov8n", y))]
            out.append(f"| {c} | " + " | ".join(cells) + " |")
        out += [f"| **total** | **{r[kind]['total']}** | **{y[kind]['total']}** |", ""]
    bb, yb = r["by_board"], y["by_board"]
    out += [("Errors by board. DeepPCB comes from a few boards, and the official split leaves some out of "
            "training entirely:"), "", ("| board | test images | train images | refdiff missed / false alarms | "
            "YOLOv8n missed / false alarms |"), "|---|---|---|---|---|"]
    for g in bb:
        out.append(f"| {g} | {bb[g]['images']} | {bb[g]['train_images']} | {bb[g]['missed']} / {bb[g]['false alarm']} | "
                   f"{yb[g]['missed']} / {yb[g]['false alarm']} |")
    out += ["", "| | seen boards mAP / F1 | unseen boards mAP / F1 |", "|---|---|---|"]
    for n, m in (("refdiff", r), ("YOLOv8n", y)):
        out.append(f"| {n} ({m['seen_boards']['images']} / {m['unseen_boards']['images']} images) | "
                   f"{pct(m['seen_boards']['mAP'])} / {pct(m['seen_boards']['f1'])} | "
                   f"{pct(m['unseen_boards']['mAP'])} / {pct(m['unseen_boards']['f1'])} |")
    if f.get("burned_in_label_images"):
        out += ["", f"Images (of all 1,500 pairs) with the annotations drawn into the pixels: {', '.join(f['burned_in_label_images'])}."]
    out.append("")
    return out


ABLATION_LABELS = {"full": "full model (as shipped)", "no_template": "no template channel",
                   "no_diff": "no signed-diff channel", "test_only": "test crop only (template-blind classifier)",
                   "no_side": "no side features", "no_warp_training": "no warped pairs in training"}


def ablations(a) -> list[str]:
    out = ["### What each part of the classifier buys (seed 0, as-shipped test pairs)", "",
           ("Same network and candidates, with one input or training source taken away "
           "(`scripts/ablations.py`)."), "",
           "| variant | mAP | F1 | full − variant, mAP points | 95% CI |", "|---|---|---|---|---|"]
    for k, v in a.items():
        d = v.get("full_minus")
        out.append(f"| {ABLATION_LABELS.get(k, k)} | {pct(v['mAP'])} | {pct(v['f1'])} | "
                   + (f"{pct(d['delta_mAP'])} | {ci(d['delta_mAP_ci95'])} |" if d else "n/a | n/a |"))
    return out + [""]


def robustness(r) -> list[str]:
    out = ["### Where it breaks (`scripts/robustness.py`)", "", "![robustness](docs/robustness.png)", "",
           "| degradation | level | mAP, registered | mAP, no registration |", "|---|---|---|---|"]
    units = {"rotation": "°", "blur": " px σ", "noise": " sd"}
    for axis, levels in r["levels"].items():
        for i, lv in enumerate(levels):
            g = r[axis]["guarded"][i]["mAP"]
            n = r[axis].get("none")
            out.append(f"| {axis} | {lv}{units[axis]} | {pct(g)} | {pct(n[i]['mAP']) if n else 'n/a'} |")
    return out + [""]


def pku_section(k) -> list[str]:
    s, reg = k["scenarios"], k["registration"]
    out = [(f"### A second dataset, no retraining: PKU synthetic PCB defects ({k['n_images']} images, "
           f"{k['n_defects']} defects)"), "",
           ("The DeepPCB-trained pipeline with every parameter unchanged, on binarized PKU colour photos scaled "
           f"by {k['base_scale']:.3f} (`scripts/pku_transfer.py`). *Any class* counts a defect box at IoU > 0.33 "
           "whatever its class."), "",
           "| images | registration | mAP | F1 | any-class mAP | any-class recall | any-class precision |",
           "|---|---|---|---|---|---|---|"]
    names = {"aligned": ("as shipped (aligned to ~1 px)", "guarded"), "aligned_noreg": ("as shipped (aligned to ~1 px)", "none"),
             "rotated": ("PKU's rotated copies", "guarded"), "rotated_noreg": ("PKU's rotated copies", "none")}
    for key, (a, b) in names.items():
        e = s[key]
        out.append(f"| {a} | {b} | {pct(e['mAP'])} | {pct(e['f1'])} | {pct(e['agnostic']['mAP'])} | "
                   f"{pct(e['agnostic']['recall'])} | {pct(e['agnostic']['precision'])} |")
    for key, label in (("yolov8n_template_blind_aligned", "YOLOv8n template-blind"),
                       ("yolov8n_ttd_aligned", "YOLOv8n on test + template + diff")):
        y = k.get(key)
        if y:
            out.append(f"| as shipped, **{label}** (DeepPCB-trained, 640 px tiles) | "
                       f"{'guarded' if 'ttd' in key else 'n/a'} | {pct(y['mAP'])} | {pct(y['f1'])} | "
                       f"{pct(y['agnostic']['mAP'])} | {pct(y['agnostic']['recall'])} | {pct(y['agnostic']['precision'])} |")
    by = s["aligned"]["by_pku_class"]
    out += ["", ("Per PKU class, as-shipped images, refdiff: recall of any-class boxes, and AP under the mapped "
            "DeepPCB class:"), "", "| PKU class | " + " | ".join(by) + " |", "|---|" + "---|" * len(by),
            "| any-class recall | " + " | ".join(pct(v["recall_any_class"]) for v in by.values()) + " |",
            "| AP as mapped class | " + " | ".join(pct(v["ap_as_mapped_class"]) for v in by.values()) + " |"]
    sw = k.get("scale_sweep", {})
    if sw:
        out += ["", "Scale sensitivity (as-shipped images): " + "; ".join(
            f"×{m}: mAP {pct(v['mAP'])}, any-class {pct(v['agnostic']['mAP'])}"
            for m, v in sorted(sw.items(), key=lambda kv: float(kv[0]))) + "."]
    out += ["", (f"Registration under known warps (±{reg['protocol']['rot_deg']:g}°, ±{reg['protocol']['shift_px']:g} px): "
                 f"median {reg['median_px']:.3f} px, p95 {reg['p95_px']:.3f}, max {reg['max_px']:.2f}; "
                 f"{round(reg['frac_over_2px'] * k['n_images'])} of {k['n_images']} pairs off by more than 2 px, "
                 f"{reg['flagged_for_review']} flagged for review."), ""]
    return out


def diff_sweep(d) -> list[str]:
    df = d["defaults"]
    out = ["### Diff-stage sensitivity (`scripts/diff_sweep.py`, shipped classifier, defaults in bold)", "",
           "| XOR tolerance (px) | min blob area (px) | candidate recall | candidates / image | mAP |",
           "|---|---|---|---|---|"]
    for g in d["grid"]:
        b = "**" if g["tol"] == df["tol"] and g["min_area"] == df["min_area"] else ""
        out.append(f"| {b}{g['tol']}{b} | {b}{g['min_area']}{b} | {pct(g['candidate_recall'])} | "
                   f"{g['candidates_per_image']:.1f} | {pct(g['mAP'])} |")
    return out + [""]


def render() -> str:
    det, cmp_, reg, d2d = j("detection.json"), j("compare.json"), j("registration.json"), j("die2db.json")
    fail = j("failures.json")
    base = (cmp_ or {}).get("baselines", {})
    out = []

    if det:
        s = det["scenarios"]
        out += ["### Detection, DeepPCB test split (500 images, IoU > 0.33, score ≥ 0.5)", "",
                "| method | sees template | mAP | 95% CI | F1 | precision | recall |",
                "|---|---|---|---|---|---|---|"]
        for b in base.values():
            out.append(f"| {b['label']} | {b['sees_template']} | {pct(b['mAP'])} | n/a | {pct(b['f1'])} | "
                       f"{pct(b['precision'])} | {pct(b['recall'])} |")
        n = s["native"]
        out.append(f"| **refdiff** (registration → diff → classifier) | yes | **{pct(n['mAP'])}** | {ci(n['ci95']['mAP'])} | "
                   f"**{pct(n['f1'])}** | {pct(n['precision'])} | {pct(n['recall'])} |")
        seeds = list(n["seed_mAP"].values())
        out += ["", (f"refdiff over {len(seeds)} classifier seeds: mAP {pct(min(seeds))}–{pct(max(seeds))}. "
                f"End to end {det['ms_per_image']['total']:.0f} ms/image on CPU "
                f"(registration {det['ms_per_image']['register']:.0f}, diff + crops "
                f"{det['ms_per_image']['candidates']:.0f}, ONNX classifier {det['ms_per_image']['classify']:.0f})."), ""]
        if base:
            out += ["Paired bootstrap over the same 500 images, refdiff minus each baseline:", "",
                    "| refdiff − | mAP points | 95% CI | F1 points | 95% CI |", "|---|---|---|---|---|"]
            for b in base.values():
                c = b["refdiff_minus"]
                out.append(f"| {b['label']} | {pct(c['delta_mAP'])} | {ci(c['delta_mAP_ci95'])} | "
                           f"{pct(c['delta_f1'])} | {ci(c['delta_f1_ci95'])} |")
            out.append("")
        sa = n.get("sign_agreement")
        if sa:
            out += [(f"Diff sign vs class family on matched test candidates: short / spur / copper came out as extra "
                    f"copper {sa['extra_classes'][0]} of {sa['extra_classes'][1]} times; open / mousebite / pin-hole "
                    f"as missing copper {sa['missing_classes'][0]} of {sa['missing_classes'][1]}."), ""]
        out += ["Per class AP:", "", "| class | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]
        for b in base.values():
            out.append(f"| {b['label']} | " + " | ".join(pct(c["ap"]) for c in b["per_class"]) + " |")
        out.append("| refdiff | " + " | ".join(pct(c["ap"]) for c in n["per_class"]) + " |")
        out += ["", "### Does registration matter? (same classifier, same scorer)", "",
                "| scenario | registration | candidate recall | mAP | 95% CI | F1 |", "|---|---|---|---|---|---|"]
        names = {"native": ("as-shipped pairs", "guarded"), "native_noreg": ("as-shipped pairs", "none"),
                 "warp": ("test warped ±3°, ±20 px, blur + noise", "guarded"),
                 "warp_noreg": ("test warped ±3°, ±20 px, blur + noise", "none")}
        for k, (a, b) in names.items():
            e = s[k]
            out.append(f"| {a} | {b} | {pct(e['candidate_recall'])} | {pct(e['mAP'])} | {ci(e['ci95']['mAP'])} | {pct(e['f1'])} |")
        if cmp_ and "registration_on_vs_off_native" in cmp_:
            c = cmp_["registration_on_vs_off_native"]
            out += ["", (f"On the as-shipped pairs, registration on − off: mAP {pct(c['delta_mAP'])} points, "
                    f"paired 95% CI {ci(c['delta_mAP_ci95'])}.")]
        out.append("")

    abl, rob, pk = j("ablations.json"), j("robustness.json"), j("pku.json")
    if abl:
        out += ablations(abl)
    if rob:
        out += robustness(rob)
    if (ds := j("diff_sweep.json")):
        out += diff_sweep(ds)
    if fail:
        out += failures(fail)

    if reg:
        out += [f"### Registration accuracy ({reg['n_pairs']} test pairs, known warp, corner error in px)", "",
                "| method | warp only: median | p95 | max | warp + optics: median | p95 | max | ms |",
                "|---|---|---|---|---|---|---|---|"]
        for m, a in reg["synthetic"]["warp_only"].items():
            b = reg["synthetic"]["warp+optics"][m]
            out.append(f"| {m} | {a['median_px']:.3f} | {a['p95_px']:.2f} | {a['max_px']:.1f} | {b['median_px']:.3f} | "
                       f"{b['p95_px']:.2f} | {b['max_px']:.1f} | {b['ms_per_pair']:.0f} |")
        g = reg["synthetic"]["warp+optics"]["guarded"]
        nat = reg["native"]
        out += ["", (f"Guarded, warp + optics: {g['flagged_for_review']} of {reg['n_pairs']} pairs flagged for review "
                f"(NCC < 0.5); worst error among the rest {g['max_px_confident']:.2f} px. "
                f"On the as-shipped pairs, {nat['pairs_moved_over_1px']} of {reg['n_pairs']} move by more than 1 px when "
                f"registered (median {nat['residual_px_p50_p95_max'][0]:.2f}, p95 {nat['residual_px_p50_p95_max'][1]:.1f} px), and "
                f"the off-defect pixel mismatch drops from {pct(nat['offdefect_mismatch_identity_mean'], 2)}% to "
                f"{pct(nat['offdefect_mismatch_registered_mean'], 2)}%."), ""]

    if pk:
        out += pku_section(pk)

    if d2d:
        r, n = d2d["registration_guarded"], d2d["registration_none"]
        a, b = d2d["defects_registered"], d2d["defects_unregistered"]
        out += [f"### Die-to-database stretch ({d2d['n_layouts']} random GDSII layouts, held-out seeds)", "",
                "| | registration error median / p95 / max (px) | defect recall | precision | candidates / die |",
                "|---|---|---|---|---|",
                (f"| no registration | {n['median_px']:.2f} / {n['p95_px']:.2f} / {n['max_px']:.2f} | {pct(b['recall'])} | "
                f"{pct(b['precision'])} | {b['candidates_per_die']:.1f} |"),
                (f"| guarded registration | {r['median_px']:.3f} / {r['p95_px']:.3f} / {r['max_px']:.3f} | {pct(a['recall'])} | "
                f"{pct(a['precision'])} | {a['candidates_per_die']:.1f} |"), ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    md = render()
    if not args.write:
        print(md)
        return
    readme = Path("README.md")
    text = readme.read_text()
    new = re.sub(r"(<!-- results:begin -->\n).*?(<!-- results:end -->)", lambda m: m.group(1) + md + "\n" + m.group(2),
                 text, flags=re.DOTALL)
    readme.write_text(new)
    print("README.md tables updated")


if __name__ == "__main__":
    main()
