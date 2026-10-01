# PDF verification of the scanpath literature audit

Verified against the supplied PDFs, 30 September 2026. This is a source-verification pass, not a new systematic literature search.

## Files and interpretation

- `literature_audit_pdf_verification.csv` uses `current_code` from the supplied `literature_audit.csv`.
- `literature_audit_pdf_verification_latest.csv` uses `current_code` from `literature_audit(1).csv`. All evidence and judgments are identical.
- Scope: the 14 rows marked `in_primary_counts=yes`, plus Islam2024 and Jaradat2025. Five fields per paper, with two additional rows for a separately evaluated subset/protocol: 82 rows for 16 papers.
- `page` means the 1-based page position in the attached PDF, not its printed journal page. The Carette attachment has alternating blank pages: PDF pages 9 and 13 correspond to printed pages 107 and 109.
- Quotes come directly from each paper, not from another paper's description. PDF text-layer quotes retain original line breaks, ligatures, and line-end hyphens. Carette's scanned pages were visually read and manually transcribed. Table and figure excerpts are identified as such.
- `not found` means that the requested information was not located in the inspected paper. Nearby methodological text is included when useful; it does not establish an unreported split unit or establish that leakage occurred.
- For `selection`, `headline`, and `dataset`, the request did not supply enumerated corrected-code categories; corrected values therefore use explanatory text.
- The requested final rule says not to infer a split unit without an explicit statement. Under that rule, Kanhirakadavath and Gawish's principal analyses become `not reported`; their image-count/batch descriptions remain in the evidence. If a separate inferred category is retained, those descriptions can support it, but it must not be merged with explicitly image-level studies.
- Multiple rows for the same paper/field are evidence records, not additional papers. Do not count them independently or apply them as successive dictionary overwrites.

## Important qualifications before rebuilding counts

### Kanhirakadavath2022

PDF p. 13 describes creating the augmented dataset, then running experiments on the augmented stack; p. 14 reports five-fold CV on that dataset. PDF p. 15, Section 4, directly discusses **possible** contamination:

> The discrepancy between the results using original and augmented datasets may be
> due to possible data contamination as similar replicates are entered into the training set,
> as seen in the testing dataset.

The original paper therefore provides a basis for discussing contamination risk. The previous statement that this concern had no basis in the original paper was incorrect. Preserve the author's uncertainty: the paper does not supply source-image grouping assignments that establish the exact overlap.

The provided augmentation vocabulary has no category for “augmented stack cross-validation; author acknowledges possible contamination.” The CSV preserves the evidence and the author's discussion without inventing a category or treating the exact assignment as verified. Do not interpret `verified` on the existing timing-unspecified entry as proof that augmentation happened safely after splitting.

A separate one-trial-per-participant, 59-image LOOCV analysis appears on pp. 14–15. This qualifies as participant-disjoint **subset** evaluation. It must not be counted as participant-disjoint evaluation of the full augmented dataset. The narrative calls 72.88% accuracy; Table 6 puts 72.88 under an AUC column, so retain that inconsistency. The original DNN AUC is 0.786, rather than exactly 0.78.

### Gawish2025 and Alsharif2024

Gawish PDF p. 8, Section 5.1, describes the batch split and explicitly discusses expanding the **training batches**. The previous timing-unspecified code omitted this passage. Alsharif PDF p. 4, Section 3.2, lists splitting first and augmentation of the training dataset next. The corrected category reflects the reported training augmentation procedure. These descriptions are not a forensic verification of implementation, and neither should be cited as a code-level guarantee that validation/test augmentation never occurred.

### Elbattah2021VAE

PDF p. 10 explicitly restricts adding generated images to the downstream classifier's training set and keeps its test images original. PDF p. 9 describes class-specific VAE training with 30% validation, but does not establish that the VAE was fitted afresh without downstream test-fold images. Keep the generator-fitting uncertainty separately; a downstream train-only augmentation label alone loses this distinction. Table 2, p. 11, gives accuracy 67% → 70% and AUC 0.71 → 0.76.

### Cilia2021 and Carette2019

Both papers use “accuracy” language alongside ROC/AUC plots. Keep the metric-label ambiguity when quoting their headline values rather than silently treating the plotted AUC as a separately verified accuracy. Cilia's participant split is explicit, but its first experiment's split unit remains unspecified. The participant split entry does not describe both experiments.

### AhmedIA2022

PDF p. 14 describes the test hold-out; p. 19, Table 5 and nearby text, pools training and validation when reporting augmentation. This is not equivalent to augmentation exclusively inside each training fold. Internal validation contamination is not excluded, but is also not established from these passages alone. Preserve its distinct category.

### Kaloforidis2025 and Mousli2025

Kaloforidis p. 5 reports grid search with five-fold CV. It does not document an independently nested outer evaluation; the current wording “same CV used” goes beyond the explicit description. Mousli p. 7 explicitly chooses ResNet-50 over ResNet-101 from Table 1 results. Fixed epochs alone do not describe all selection. The validation role and independence of architecture selection remain unclear.

## Zhang2026: requested protocol-comparison check

Read the supplied 28-page main PDF (`jemr-19-00085.pdf`) and 7-page supplement (`jemr-4445244-supplementary.pdf`). The checks below concern controlled alternative **evaluation-protocol** comparisons, not architectural ablations.

| Requested comparison | Finding in supplied main article | Supplement |
|---|---|---|
| Image-level versus participant-level splits | No such comparison found. Sections 2.2.1 and 2.4.3 describe participant-independent evaluation. | No such comparison found. Table S4 reports image-level metrics under participant-disjoint folds. |
| Augment-then-split versus split-then-augment | No order comparison found. Sections 2.2.1, 2.4.3 and 3.8 describe training augmentation / rotation sensitivity. | No order comparison found. Methods S2 specifies rotation within the training fold. |
| Test-based epoch/hyperparameter selection versus validation or inner-CV selection | No such comparison found. Section 2.4.3 uses prespecified settings and the fixed final epoch. | No such comparison found; Methods S2 explicitly excludes outer-test selection. |
| Participant-label permutation nulls | None found in the main article. | None found in the supplement. |

The main text contains two evaluation descriptions. The secondary original fixed-split description (p. 12, Section 2.4.1) says that validation was used for model selection and hyperparameter tuning. The primary repeated participant-level evaluation (pp. 13–14, Section 2.4.3) says settings were fixed before CV and evaluates epoch 20 without early stopping. For the primary pooled result AUC 0.937, use the latter. Likewise, the primary repeated evaluation has **no augmentation** (p. 14); its separate rotation sensitivity analysis is not the default primary pipeline. Do not let the additional augmentation evidence row overwrite the primary `none` entry when rebuilding.

The supplement confirms the distinction on PDF p. 3, Methods S2:

> Primary training uses deterministic resizing and normalization only. The rotation sensitivity
> analysis samples an angle uniformly from −10◦to +10◦with probability 0.5 within the training fold. Every
> model is evaluated at its fixed final epoch. No outer-test prediction is used for model, epoch, threshold, or
> hyperparameter selection.

Supported manuscript positioning:

> Zhang et al. used participant-independent evaluation on the same image set and discussed leakage risks in earlier studies. We extend this line of work through controlled comparisons of split unit, augmentation order, and test-set selection, with participant-label permutation analyses for the frozen-feature experiments.

This states a verified difference from this paper. It does not establish a literature-wide “first.” E4 currently lacks permutation experiments, so the permutation clause is restricted accordingly.

## Dataset checks requested separately

**Islam2024:** PDF p. 6, Section 3.1, explicitly identifies Dataset 1 as 547 images, 328 TD and 219 ASD, credited to Elbattah. PDF p. 12, Table 3, assigns 99.43% to Dataset 1; Table 4 assigns 96.78% to the separate Duan dataset. Section 4.4 says those comparisons use augmented versions. The dataset attribution is therefore confirmed. Inclusion in primary counts still depends on whether preprints are eligible under the audit's stated rules; do not change that rule after seeing the results.

**Jaradat2025:** PDF p. 4, Section 3.1, explicitly reports ETSDS from Figshare with **215 ASD and 319 non-ASD images = 534**, rather than 547. The related-work table's descriptions of other papers using 547 images are not evidence that Jaradat's own analytic sample used 547. Keep this as a related dataset/subset until exact provenance is reconciled. The paper also classifies mild/moderate/severe severity; this is not continuous, participant-level CARS regression. Table 2 on p. 8 lists 98 + 87 + 34 severity images = 219, which is inconsistent with the separately reported 215 ASD images; do not silently reconcile those counts.

## Reproducibility and limits

CSV generation checks that every non-scanned excerpt is an exact substring of its cited PDF text layer; it does not reconstruct quotes from the audit's old evidence columns. Current codes are copied unchanged from their respective baseline CSVs. The output has the ten requested columns and one paper/field/evidence record per CSV row (quoted cells may contain line breaks).

No existing `build_audit.py` was modified and no final aggregate counts were produced in this pass. Rebuilding requires protocol-aware handling of Cilia, Kanhirakadavath, Zhang and VAE generator fitting, rather than blindly treating every evidence row as a paper.

No retrospective search log was fabricated: the present task verifies supplied PDFs; it does not provide the original literature-search queries, dates or exclusion trail needed to reconstruct that earlier search.
