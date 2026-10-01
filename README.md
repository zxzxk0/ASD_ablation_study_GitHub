# ASD Ablation Study

Reproducible code for the study:

**Subject-Level Evaluation of Autism Severity Prediction from Eye Tracking: Temporal Gaze Dynamics, Scanpath Images, and Recording-Quality Sensitivity**

## Repository structure

```text
ASD_ablation_study/
├── README.md
├── requirements.txt
├── config.py
├── data_loader.py
├── evaluator.py
├── run_ablation.py
├── visualize_results.py
├── models/
├── results/
└── scripts/
    ├── gaze/
    ├── audits/
    └── image/
```

## Data

The original eye-tracking dataset is not redistributed in this repository.

The canonical gaze pipeline expects the original raw SMI exports and participant metadata. A typical local layout is:

```text
Eye-Tracking Dataset/
├── Eye-tracking Output/
│   ├── 1.csv
│   ├── 2.csv
│   └── ...
└── Metadata_Participants.csv
```

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Core dependencies include NumPy, pandas, SciPy, scikit-learn, matplotlib, PyTorch, torchvision, open_clip, Pillow, and tqdm.

## Canonical gaze-dynamics analysis

Primary script:

```text
scripts/gaze/run_gaze_canonical_from_raw.py
```

Recommended final run:

```bat
python scripts\gaze\run_gaze_canonical_from_raw.py ^
  --bootstrap 10000 ^
  --fixed-permutations 2000 ^
  --nested-family-permutations 2000
```

This script reconstructs the 18 base and 11 enhanced gaze features, aggregates them at participant level, runs outer LOPO evaluation with fold-specific preprocessing, performs ridge/SVR/GBR comparisons, and runs the fully nested family-selection analysis used as the primary inferential result.

## Follow-up audits

```bat
python scripts\gaze\run_canonical_followup_audits.py
```

This reproduces severity-stratified metrics, Bland-Altman agreement, clinical tolerance rates, empirical leave-one-out residual-based prediction intervals, and complete-case recording-quality sensitivity analyses.

## Table 4 audit from retained participant-level features

```text
scripts/gaze/run_table4_gaze_reproduction.py
```

This is a retained-feature audit pipeline. The raw-data canonical reconstruction above should be treated as the preferred provenance source for the final manuscript values.

## Marker and recording-quality analyses

```text
scripts/audits/run_table7_and_section44_reconciliation.py
scripts/audits/run_marker_confound_correlations.py
scripts/audits/run_quality_only_model.py
scripts/audits/check_missing_features.py
```

These scripts support the univariate marker scan, partial-correlation reconciliation, recording-quality confound checks, and missingness audit.

## Rendered-image arm

```text
scripts/image/run_shallow_probe_all_backbones.py
scripts/image/train_mlp_v6_final.py
scripts/image/verify_v6_checkpoints_match_table1.py
```

The shallow-probe script supports the current image arm. `train_mlp_v6_final.py` is retained for embedding-cache generation and historical PEA-MLP experiments; its PEA-MLP results and `verify_v6_checkpoints_match_table1.py` refer to earlier manuscript tables, not v5 Table 1. The current MetaCA-MIL evaluation is documented below.

Example image-arm commands:

```bat
python scripts\image\run_shallow_probe_all_backbones.py --cache-dir outputs\cache_v3_unambiguous --metadata-csv Metadata\Metadata\Metadata_Participants.csv

python scripts\image\train_mlp_v6_final.py --data_csv data\all_scanpath_absolute.jsonl --image-root data\Images\TSImages --cache-dir outputs\cache_v2 --output-dir outputs\checkpoints_v6

python scripts\image\verify_v6_checkpoints_match_table1.py --ckpt-dir outputs\checkpoints_v6 --inspect-only
```

## LLM / scanpath-image ablation

The repository root contains the LLM ablation framework:

```text
config.py
data_loader.py
evaluator.py
run_ablation.py
visualize_results.py
models/
```

API credentials are not included. Supply required credentials through environment variables or a local `.env` file ignored by Git.

For example on Windows CMD:

```bat
set OPENAI_API_KEY=YOUR_KEY_HERE
set GOOGLE_API_KEY=YOUR_KEY_HERE
```

## Reproducibility notes

- The primary analyses use participant-disjoint evaluation; protocol stress tests deliberately compare alternative split and selection rules.
- The primary gaze-dynamics inferential result uses fully nested model-family selection.
- For gaze regression, imputation and standardization are re-fitted on the training participants within each inner and outer fold. Classification and protocol-replay preprocessing follow the separately documented procedures; they should not be described collectively as fully nested preprocessing.
- Individual ridge/SVR/GBR results are descriptive; family-specific permutation tests are secondary.
- The recording-quality ablation uses a fixed complete-case cohort.
- Large model checkpoints, embedding caches, and the original dataset are excluded from Git.
- Some scripts retain the local paths used during development. Override input, cache, and output paths with their command-line arguments.

## Generative AI disclosure

Generative AI (ChatGPT, OpenAI; Claude, Anthropic) was used to assist in developing and refactoring portions of the analysis code. All code was reviewed, tested, and executed by the authors, who take full responsibility for the analyses and reported results.

## Citation

Citation information will be added after publication.

## License

Add the appropriate software license before public release.

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

## Remaining reproducibility materials

This package merges the supplied repository and additions; it is not yet a complete archive of every v5 analysis. Still to add: Claude Opus 4.5 implementation and saved results; final MetaCA-MIL and E1-E4 run outputs; the canonical 2,000-permutation gaze output; and the package versions used for the published runs. The original data and large caches are not bundled. No new experiment results were generated during packaging. A software license still needs to be selected by the authors.

## Participant-overlap stress tests (Supplementary S5)

Keep these two files in the same directory: the identity/metadata script imports the ridge stress-test module.

```bat
python scripts\image\run_participant_leakage_stress_test.py --cache-dir outputs\cache_v3_unambiguous --metadata-csv Metadata\Metadata\Metadata_Participants.csv --n-repeats 10 --n-label-permutations 100
python scripts\image\run_identity_and_metadata_leakage.py --cache-dir outputs\cache_v3_unambiguous --metadata-csv Metadata\Metadata\Metadata_Participants.csv --n-repeats 10 --n-label-permutations 100 --n-identity-permutations 10000
```

These commands make the manuscript repeat/permutation counts explicit; they are not recovered execution logs. Compare all settings and saved outputs with the actual runs before claiming numerical reproduction. Identity permutations reassign participant IDs across images while preserving image counts; CARS-label permutations shuffle labels at participant level.

Quick checks without the original data:

```bat
python scripts\image\run_participant_leakage_stress_test.py --self-test-only
python scripts\image\run_identity_and_metadata_leakage.py --self-test-only
```

Both self-tests passed during packaging. Python syntax and ZIP integrity were checked; full dataset-dependent experiments and API calls were not rerun.

## Historical figure code

`legacy/generate_plots.py` targets historical v2 checkpoints and is not the v5 figure-generation pipeline. The current leakage-contrast figure uses `scripts/figures/make_fig_leakage_contrast.py` and saved MetaCA-MIL participant predictions.

## Submission snapshot

See `RELEASE_STATUS.md` for the remaining materials. After the final files are added, record the Git commit or release tag used for submission so the manuscript points to a fixed code version.
