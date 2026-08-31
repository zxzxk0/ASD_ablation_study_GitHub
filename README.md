# ASD Ablation Study

Reproducible code for the study:

**Temporal Gaze Dynamics as Interpretable Candidate Markers of Autism Symptom Severity: A Subject-Level Comparison with Rendered Scanpath-Image Representations**

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
  --nested-family-permutations 200
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

These scripts support the rendered scanpath-image experiments, shallow-probe analysis, model training, and checkpoint provenance audit.

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

- Evaluation is performed at the participant level.
- The primary gaze-dynamics inferential result uses fully nested model-family selection.
- Data-dependent preprocessing is fit only on outer-training participants.
- Individual ridge/SVR/GBR results are descriptive; family-specific permutation tests are secondary.
- The recording-quality ablation uses a fixed complete-case cohort.
- Large model checkpoints, embedding caches, and the original dataset are excluded from Git.
- Image-arm paths are repository-relative by default and can be overridden with command-line arguments.

## Generative AI disclosure

Generative AI (ChatGPT, OpenAI) was used to assist in developing and refactoring portions of the analysis code. All code was reviewed, tested, and executed by the authors, who take full responsibility for the analyses and reported results.

## Citation

Citation information will be added after publication.

## License

Add the appropriate software license before public release.
