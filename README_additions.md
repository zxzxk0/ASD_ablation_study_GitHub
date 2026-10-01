## Validation-protocol analysis (MetaCA-MIL, nested vs non-nested early stopping)

Script: `scripts/image/run_metacamil_nested_es.py`
Reproduces main-text Table 3 and Supplementary Table S1:
- four stopping rules: held-out epoch, fixed final epoch, nested refit, nested inner ensemble
- 5 seeds
- 50 participant-label permutations
- automated consistency checks

```bat
python scripts\image\run_metacamil_nested_es.py --data-path data\all_scanpath_absolute.jsonl --cache outputs\cache_v2\all_data_EVA02-B-16_merged2b_s8b_b131k.npz --out-dir outputs\metacamil_nested_es --configs best_cand --n-perm 50 --strict
```

The MetaCA-MIL command is a documented invocation with explicit 50 permutations, not a recovered execution log. The cache filename follows the supplied script; verify that this cache is available and adjust the data path. Other options retain script defaults; confirm them against the saved run configuration. The default `--n-perm 0` does not run permutations.

Outputs:
- `summary.csv`
- `permutation_summary.csv`
- `participant_predictions.csv`

Supplementary leakage-contrast figure: `scripts/figures/make_fig_leakage_contrast.py --pred outputs\metacamil_nested_es\participant_predictions.csv`

## ASD-versus-TD protocol replay (E1-E4)

Script: `scripts/replay/run_audit_replay_E1_E4.py` (main-text Section 3.5, Supplementary Section S8, Table S14)

| Experiment | Comparison |
|---|---|
| E1 | split unit |
| E2 | augmentation order |
| E3 | test-fold epoch/configuration selection |
| E4 | fine-tuned ResNet-50 epoch selection |

The replay compares image-level and participant-level splitting; participant-disjoint conditions use deterministic stratified group splitting. A synthetic self-test checks that leaks are detected.

```bat
python scripts\replay\run_audit_replay_E1_E4.py --experiments E1 E2 E3 E4 --e4-seeds 5 --strict
```

## Literature audit (Supplementary Section S9, Table S15)

Folder: `audit/`
- `literature_audit_pdf_verification.csv`: quote and page evidence checked against each PDF.
- `build_audit_verified.py`: derives the paper-level codes, counts and LaTeX table from that file.
- `search_log.md`: search queries.

```bat
cd audit && python build_audit_verified.py
```
