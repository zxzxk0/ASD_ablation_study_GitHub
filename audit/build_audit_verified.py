# -*- coding: utf-8 -*-
"""
build_audit_verified.py
=======================
Final, paper-level coding for the targeted audit of reported evaluation
procedures in studies classifying ASD vs TD on the public scanpath-image set
(Carette et al.; Figshare 7073087; 547 images, 59 children).

Inputs
  literature_audit_pdf_verification.csv   82 evidence rows (16 papers), verbatim
                                          quotes + PDF page, checked against PDFs
Outputs
  literature_audit_final.csv              one row per paper, final codes + page refs
  literature_audit_final_counts.csv       counts (each paper counted once per item)
  literature_audit_final_table.tex        LaTeX table for the Supplement
  literature_audit_evidence_used.csv      the exact evidence row behind every code

Rules (fixed before recounting)
  * Eligibility for counts: full text read, and the paper's own analytic sample is
    the 547-image release (preprints eligible, as in the original rule).
  * A code is assigned only from an explicit statement; otherwise "not reported".
    Split descriptions given only in image counts/batches are "not reported",
    with the description kept in `split_note` (never merged with explicit codes).
  * Codes describe the paper's HEADLINE (primary) analysis. Secondary analyses are
    recorded in separate columns (e.g., participant-disjoint subset analyses,
    sensitivity analyses) and never overwrite the primary code.
  * Another paper's statements about a study are not evidence for that study.
"""
import csv
import collections
import pandas as pd

V = pd.read_csv("literature_audit_pdf_verification.csv", dtype=str).fillna("")
NR = "not reported"


def ev(pid, field, contains=None):
    """Return the verification row for (paper, field); `contains` disambiguates
    papers with several evidence rows for the same field."""
    d = V[(V.id == pid) & (V.field == field)]
    if contains is not None:
        d = d[d.section_or_table.str.contains(contains, regex=False)]
    if len(d) != 1:
        raise SystemExit(f"[FATAL] evidence lookup {pid}/{field}/{contains!r} -> {len(d)} rows")
    r = d.iloc[0]
    return dict(page=r.page, section=r.section_or_table, verdict=r.verdict,
                quote=" ".join(r.verbatim_quote.split()))


P = []  # final paper-level records


def paper(pid, citation, doi, include, split_primary, split_note, pd_analysis, augmentation,
          aug_note, selection, author_leakage, headline, headline_level, evidence, notes=""):
    P.append(dict(id=pid, citation=citation, doi=doi, in_counts=include,
                  split_primary=split_primary, split_note=split_note,
                  participant_disjoint_analysis=pd_analysis, augmentation=augmentation,
                  aug_note=aug_note, selection=selection, author_discusses_leakage=author_leakage,
                  headline=headline, headline_level=headline_level, notes=notes,
                  **{f"{k}_page": v["page"] for k, v in evidence.items()},
                  **{f"{k}_section": v["section"] for k, v in evidence.items()},
                  _evidence=evidence))


# ---------------------------------------------------------------- 15 included
paper("Carette2019", "Carette et al., HEALTHINF 2019", "10.5220/0007402601030112", "yes",
      NR, "", "none", "applied; timing not reported", "5 synthetic per image (2,735)",
      "fixed epochs; selection not described", "no",
      "ROC-AUC ~0.92 (narrative: '~92.0%' performance)", "image",
      dict(dataset=ev("Carette2019", "dataset"), split=ev("Carette2019", "split_unit"),
           aug=ev("Carette2019", "augmentation"), sel=ev("Carette2019", "selection"),
           headline=ev("Carette2019", "headline")),
      "metric label ambiguous (accuracy vs AUC)")
paper("Cilia2021", "Cilia et al., JMIR Hum Factors 2021", "10.2196/27706", "yes",
      NR, "headline (~90%) experiment: split unit not stated",
      "full image set, participant-level (secondary experiment; ~71%)",
      "applied; timing not reported", "2,735 synthetic", "20% validation; fixed epochs; selection not described", "no",
      "~90% (headline) vs ~71% (participant split); metric label ambiguous", "not stated",
      dict(dataset=ev("Cilia2021", "dataset"), split=ev("Cilia2021", "split_unit"),
           aug=ev("Cilia2021", "augmentation"), sel=ev("Cilia2021", "selection"),
           headline=ev("Cilia2021", "headline")))
paper("Elbattah2021VAE", "Elbattah et al., J Imaging 2021", "10.3390/jimaging7050083", "yes",
      NR, "", "none", "generated images added to training only; generator-fitting scope not reported",
      "class-specific VAEs (30% validation)", "fixed epochs; selection not described", "no",
      "ROC-AUC 0.71 -> 0.76; accuracy 67% -> 70% (VAE augmentation)", "image",
      dict(dataset=ev("Elbattah2021VAE", "dataset"), split=ev("Elbattah2021VAE", "split_unit"),
           aug=ev("Elbattah2021VAE", "augmentation"), sel=ev("Elbattah2021VAE", "selection"),
           headline=ev("Elbattah2021VAE", "headline")))
paper("Elbattah2022TL", "Elbattah et al., BIOSTEC 2022", "10.5220/0010975500003123", "yes",
      "participant-level (explicit)", "", "full image set (primary)", "train-only after split (explicit)", "",
      NR, "no", "ROC-AUC ~0.78 (VGG-16)", "not stated",
      dict(dataset=ev("Elbattah2022TL", "dataset"), split=ev("Elbattah2022TL", "split_unit"),
           aug=ev("Elbattah2022TL", "augmentation"), sel=ev("Elbattah2022TL", "selection"),
           headline=ev("Elbattah2022TL", "headline")))
paper("Kanhirakadavath2022", "Kanhirakadavath & Chandran, Diagnostics 2022", "10.3390/diagnostics12020518", "yes",
      NR, "split described only in image counts (382/165 images)",
      "subset only: one image per participant (59 images, LOOCV; 72.88%)",
      "applied; timing not reported", "rotations; CV performed on augmented stack",
      "architectures varied 'to obtain the best results'; selection data not reported",
      "yes (possible data contamination, PDF p.15)",
      "ROC-AUC 0.970 (augmented) vs 0.786 (original)", "image",
      dict(dataset=ev("Kanhirakadavath2022", "dataset"), split=ev("Kanhirakadavath2022", "split_unit", "Step 2"),
           pd_subset=ev("Kanhirakadavath2022", "split_unit", "LOOCV"),
           aug=ev("Kanhirakadavath2022", "augmentation"), sel=ev("Kanhirakadavath2022", "selection"),
           headline=ev("Kanhirakadavath2022", "headline")),
      "Table 6 labels the 72.88 subset value as AUC while the text calls it accuracy")
paper("AhmedIA2022", "Ahmed et al., Electronics 2022", "10.3390/electronics11040530", "yes",
      NR, "80% train+validation / 20% test (image counts)", "none",
      "after test hold-out, applied to pooled train+validation", "internal validation contamination not excluded",
      "validation subset; fixed max epochs; selection not described", "no",
      "99.8% accuracy (FFNN/ANN)", "image",
      dict(dataset=ev("AhmedIA2022", "dataset"), split=ev("AhmedIA2022", "split_unit"),
           aug=ev("AhmedIA2022", "augmentation"), sel=ev("AhmedIA2022", "selection"),
           headline=ev("AhmedIA2022", "headline")),
      "text (ASD x7, TD x10) and Table 5 (ASD x10, TD x7) disagree")
paper("Thanarajan2023", "Thanarajan et al., CMC 2023", "10.32604/cmc.2023.039644", "yes",
      NR, "80:20 and 70:30 hold-out", "none", NR, "",
      "optimiser fitness based on accuracy; data used not reported", "no",
      "99.29% (balanced accuracy, 80:20)", "image",
      dict(dataset=ev("Thanarajan2023", "dataset"), split=ev("Thanarajan2023", "split_unit"),
           aug=ev("Thanarajan2023", "augmentation"), sel=ev("Thanarajan2023", "selection"),
           headline=ev("Thanarajan2023", "headline")))
paper("Islam2024", "Islam et al., arXiv 2401.03575 (2024; preprint)", "arXiv:2401.03575", "yes",
      NR, "random 80:10:10", "none", "applied; timing not reported", "x4; results reported on augmented data",
      "epoch count described as 'optimal'; data used not reported", "no",
      "99.43% accuracy (Dataset 1, augmented)", "image",
      dict(dataset=ev("Islam2024", "dataset"), split=ev("Islam2024", "split_unit"),
           aug=ev("Islam2024", "augmentation"), sel=ev("Islam2024", "selection"),
           headline=ev("Islam2024", "headline")), "preprint")
paper("Alsaidi2024", "Alsaidi et al., Information 2024", "10.3390/info15030133", "yes",
      NR, "75:25 hold-out (image counts)", "none", "none", "",
      "configurations compared on the reported evaluation results; separate validation not reported", "no",
      "95.59% accuracy", "image",
      dict(dataset=ev("Alsaidi2024", "dataset"), split=ev("Alsaidi2024", "split_unit"),
           aug=ev("Alsaidi2024", "augmentation"), sel=ev("Alsaidi2024", "selection"),
           headline=ev("Alsaidi2024", "headline")),
      "reported sensitivity/specificity (<80%) inconsistent with 95.59% accuracy")
paper("Alsharif2024", "Alsharif et al., Front Med 2024", "10.3389/fmed.2024.1436646", "yes",
      NR, "train/validation/test", "none", "train-only after split (explicit)", "",
      "validation set; fixed epochs; selection not described", "no", "100% accuracy (MobileNet)", "image",
      dict(dataset=ev("Alsharif2024", "dataset"), split=ev("Alsharif2024", "split_unit"),
           aug=ev("Alsharif2024", "augmentation"), sel=ev("Alsharif2024", "selection"),
           headline=ev("Alsharif2024", "headline")))
paper("Gawish2025", "Gawish et al., IJCESEN 2025", "10.22399/ijcesen.3900", "yes",
      NR, "split described only in image batches (70-20-10)", "none", "train-only after split (explicit)", "",
      "variant chosen as 'best performing model'; criterion/data not reported", "no",
      "98.4% accuracy", "image",
      dict(dataset=ev("Gawish2025", "dataset"), split=ev("Gawish2025", "split_unit"),
           aug=ev("Gawish2025", "augmentation"), sel=ev("Gawish2025", "selection"),
           headline=ev("Gawish2025", "headline")))
paper("Kaloforidis2025", "Kaloforidis et al., Eng Proc 2025", "10.3390/engproc2025107012", "yes",
      NR, "5-fold CV", "none", NR, "",
      "grid search with 5-fold CV; nested outer evaluation not reported", "no", "ROC-AUC 0.88 (ConvRF)", "image",
      dict(dataset=ev("Kaloforidis2025", "dataset"), split=ev("Kaloforidis2025", "split_unit"),
           aug=ev("Kaloforidis2025", "augmentation"), sel=ev("Kaloforidis2025", "selection"),
           headline=ev("Kaloforidis2025", "headline")))
paper("Mousli2025", "Mousli et al., Multimedia Systems 2025", "10.1007/s00530-025-01885-4", "yes",
      NR, "70/20/10", "none", "applied; timing not reported", "used in contrastive pre-training",
      "architecture chosen from reported results (Table 1); validation role not reported", "no",
      "F1 0.621 / accuracy 0.697 (full data); F1 0.725 (200-shot)", "image",
      dict(dataset=ev("Mousli2025", "dataset"), split=ev("Mousli2025", "split_unit"),
           aug=ev("Mousli2025", "augmentation"), sel=ev("Mousli2025", "selection"),
           headline=ev("Mousli2025", "headline")))
paper("AhmedM2026", "Ahmed et al., MAKE 2026", "10.3390/make8070176", "yes",
      "image-level (explicit)", "", "none", "train-only after split (explicit)", "",
      "validation-based (ensemble weights); fixed epochs", "yes (image-level split may leak, PDF p.19)",
      "98.18% accuracy; ROC-AUC 0.9959", "image",
      dict(dataset=ev("AhmedM2026", "dataset"), split=ev("AhmedM2026", "split_unit"),
           aug=ev("AhmedM2026", "augmentation"), sel=ev("AhmedM2026", "selection"),
           headline=ev("AhmedM2026", "headline")))
paper("Zhang2026", "Zhang et al., J Eye Mov Res 2026", "10.3390/jemr19040085", "yes",
      "participant-level (explicit)", "", "full image set (primary)", "none",
      "primary pipeline without augmentation; train-fold rotation only in a sensitivity analysis",
      "fixed configuration and final epoch; outer-test selection explicitly excluded", "yes (critiques prior studies)",
      "participant-level accuracy 0.870; ROC-AUC 0.937 (pooled)", "participant",
      dict(dataset=ev("Zhang2026", "dataset"), split=ev("Zhang2026", "split_unit"),
           aug=ev("Zhang2026", "augmentation", "primary"), sel=ev("Zhang2026", "selection"),
           headline=ev("Zhang2026", "headline")),
      "no comparison of alternative evaluation protocols in main text or supplement")
# ---------------------------------------------------------------- excluded
paper("Jaradat2025", "Jaradat et al., Diagnostics 2025", "10.3390/diagnostics15010066", "no",
      NR, "80/20; 5-fold", "none", "applied; timing not reported", "",
      "k reported as best on test metrics", "no", "~98.0% accuracy", "image",
      dict(dataset=ev("Jaradat2025", "dataset"), split=ev("Jaradat2025", "split_unit"),
           aug=ev("Jaradat2025", "augmentation"), sel=ev("Jaradat2025", "selection"),
           headline=ev("Jaradat2025", "headline")),
      "EXCLUDED: own analytic sample reported as 534 images (215 ASD / 319 non-ASD), not the 547-image release")

HV = {  # headline value used for the >=0.95 count (accuracy or AUC as reported for the headline analysis)
    "Carette2019": 0.92, "Cilia2021": 0.90, "Elbattah2021VAE": 0.76, "Elbattah2022TL": 0.78,
    "Kanhirakadavath2022": 0.970, "AhmedIA2022": 0.998, "Thanarajan2023": 0.9929, "Islam2024": 0.9943,
    "Alsaidi2024": 0.9559, "Alsharif2024": 1.00, "Gawish2025": 0.984, "Kaloforidis2025": 0.88,
    "Mousli2025": 0.697, "AhmedM2026": 0.9818, "Zhang2026": 0.937, "Jaradat2025": 0.98}
SHORT = {"Carette2019": "AUC ~0.92", "Cilia2021": "~90% / ~71% (part.)", "Elbattah2021VAE": "AUC 0.71 to 0.76",
         "Elbattah2022TL": "AUC ~0.78", "Kanhirakadavath2022": "AUC 0.970 (aug.) / 0.786",
         "AhmedIA2022": "Acc 99.8%", "Thanarajan2023": "Bal. acc 99.29%", "Islam2024": "Acc 99.43%",
         "Alsaidi2024": "Acc 95.59%", "Alsharif2024": "Acc 100%", "Gawish2025": "Acc 98.4%",
         "Kaloforidis2025": "AUC 0.88", "Mousli2025": "F1 0.62 / Acc 0.70", "AhmedM2026": "Acc 98.18%, AUC 0.996",
         "Zhang2026": "Acc 0.870, AUC 0.937", "Jaradat2025": "Acc ~98%"}
import re as _re
for p in P:
    p["headline_value"] = HV[p["id"]]
    p["year"] = int(_re.search(r"(19|20)\d\d", p["citation"]).group(0))

# ------------------------------------------------------------------ checks
ids = [p["id"] for p in P]
assert len(ids) == len(set(ids)), "duplicate paper"
assert set(ids) == set(V.id), f"papers without final codes: {set(V.id) ^ set(ids)}"
for p in P:
    for k, e in p["_evidence"].items():
        assert e["verdict"] in ("verified", "corrected", "not found"), (p["id"], k)
inc = [p for p in P if p["in_counts"] == "yes"]
print(f"[CHECK] {len(P)} papers coded, {len(inc)} in counts, every code linked to one evidence row")

# ------------------------------------------------------------------ outputs
cols = [c for c in P[0] if c != "_evidence"]
allcols = sorted({c for p in P for c in p if c != "_evidence"}, key=lambda c: (c not in cols, c))
with open("literature_audit_final.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=allcols)
    w.writeheader()
    for p in P:
        w.writerow({k: v for k, v in p.items() if k != "_evidence"})
with open("literature_audit_evidence_used.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "item", "page", "section_or_table", "verdict", "verbatim_quote"])
    for p in P:
        for k, e in p["_evidence"].items():
            w.writerow([p["id"], k, e["page"], e["section"], e["verdict"], e["quote"]])

C = collections.OrderedDict()
C["studies included (full text; 547-image release)"] = len(inc)
for label, key in [("headline split unit", "split_primary"), ("augmentation (headline analysis)", "augmentation"),
                   ("participant-disjoint analysis reported", "participant_disjoint_analysis")]:
    cnt = collections.Counter(
        ("full image set" if (key == "participant_disjoint_analysis" and p[key].startswith("full"))
         else "one-image-per-participant subset only" if (key == "participant_disjoint_analysis" and p[key].startswith("subset"))
         else p[key]) for p in inc)
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        C[f"{label}: {k}"] = v
C["headline split unit not reported, split described only in image counts/batches"] = sum(
    p["split_primary"] == NR and ("image counts" in p["split_note"] or "image batches" in p["split_note"]) for p in inc)
sel_cat = collections.Counter(
    "explicitly independent of the test set" if ("excluded" in p["selection"] or p["selection"].startswith("validation-based"))
    else "not described" if (p["selection"] == NR or "selection not described" in p["selection"])
    else "selection performed; data source not reported or ambiguous" for p in inc)
for k, v in sel_cat.items():
    C[f"model selection: {k}"] = v
C["authors discuss leakage/contamination risk in their own protocol"] = sum(
    p["author_discusses_leakage"].startswith("yes") and "prior" not in p["author_discusses_leakage"] for p in inc)
C["authors critique leakage in prior studies"] = sum("prior" in p["author_discusses_leakage"] for p in inc)
hi = [p for p in inc if p["headline_value"] >= 0.95]
C["headline value >=0.95 (accuracy or AUC)"] = len(hi)
C["  ...of which with a participant-disjoint evaluation of the full image set"] = sum(
    p["participant_disjoint_analysis"].startswith("full") for p in hi)
full = [p for p in inc if p["participant_disjoint_analysis"].startswith("full")]
C["participant-disjoint full-set results (range as reported)"] = "; ".join(f"{p['id']}: {SHORT[p['id']]}" for p in full)
with open("literature_audit_final_counts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["item", "n"]); w.writerows(C.items())
for k, v in C.items():
    print(f"{v:>3}  {k}")

# ------------------------------------------------------------------ LaTeX
def esc(s):
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
             .replace("_", "\\_").replace("~", "$\\sim$").replace("->", "$\\rightarrow$").replace(">=", "$\\geq$"))


def short_split(p):
    return {"participant-level (explicit)": "participant", "image-level (explicit)": "image"}.get(p["split_primary"], "NR")


def short_aug(p):
    a = p["augmentation"]
    return ("train-only" if a.startswith("train-only") else "train+val pool" if a.startswith("after test")
            else "train (gen. NR)" if a.startswith("generated") else "none" if a == "none"
            else "NR" if a == NR else "timing NR")


def short_pd(p):
    a = p["participant_disjoint_analysis"]
    return "full set" if a.startswith("full") else "subset" if a.startswith("subset") else "--"


L = [r"\begin{table}[htbp]", r"\centering\scriptsize",
     r"\caption{Reported evaluation procedures in studies classifying ASD vs.\ TD on the public scanpath-image release "
     r"(targeted audit of $n=15$ studies; full text verified against the source PDFs). Codes describe the headline analysis "
     r"and use explicit statements only; NR = not reported. ``Part.-disjoint'' marks any participant-disjoint analysis reported "
     r"(full image set or a one-image-per-participant subset).}",
     r"\label{tab:lit-audit}", r"\begin{tabular}{@{}llllll@{}}", r"\toprule",
     r"Study & Split unit & Augmentation & Part.-disjoint & Headline (level) & PDF p. \\", r"\midrule"]
for p in sorted(inc, key=lambda p: (p["year"], p["id"])):
    pages = "/".join(dict.fromkeys(str(p.get(f"{k}_page", "")) for k in ("split", "aug", "headline")))
    L.append(f"{esc(p['citation'])} & {short_split(p)} & {short_aug(p)} & {short_pd(p)} & "
             f"{esc(SHORT[p['id']])} ({p['headline_level']}) & {esc(pages)} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open("literature_audit_final_table.tex", "w", encoding="utf-8").write("\n".join(L))
print("[DONE] literature_audit_final.csv, _counts.csv, _table.tex, literature_audit_evidence_used.csv")
