# -*- coding: utf-8 -*-
"""
verify_v6_checkpoints_match_table1.py
========================================
checkpoints_v6/의 p2_*_pea_mlp_f01~f26.pt 체크포인트가 정말 Table 1의
published 숫자를 만든 소스인지 검증합니다. 재학습 없이, 저장된 26개 fold
체크포인트를 불러와 held-out 예측만 다시 계산합니다.

동시에 다음을 확인합니다:
  - v6 체크포인트 파일명에 seed 구분이 없다는 것 자체가, "5 seeds"라는
    Table 1 캡션의 근거가 v6에는 없다는 뜻인지
  - v6로 재구성한 R²/MAE가 published Table 1 값과 실제로 일치하는지

주의: 이 스크립트는 PEA-MLP 아키텍처의 정확한 구조(레이어 크기, activation
등)를 모르는 상태에서는 state_dict를 로드할 수 없습니다. 먼저
--inspect-only로 체크포인트 안에 아키텍처 정보나 config가 저장되어
있는지부터 확인하세요.

실행:
    # 1) 먼저 체크포인트 구조 확인 (모델 없이도 가능)
    python verify_v6_checkpoints_match_table1.py --inspect-only

    # 2) 아키텍처가 확인되면 실제 평가 실행
    python verify_v6_checkpoints_match_table1.py --backbone ResNet-50
"""
from __future__ import annotations
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CKPT_DIR = Path("outputs") / "checkpoints_v6"

# Published Table 1 values (PEA-MLP column) to check reconstructed
# predictions against.
PUBLISHED_PEA_MLP = {
    "EVA02-B-16":  {"MAE": 5.23, "R2": -0.07},
    "EVA02-L-14":  {"MAE": 5.24, "R2": -0.06},
    "DINOv2":      {"MAE": 6.25, "R2": -0.45},   # DINOv2 ViT-B/14 in paper
    "ResNet-50":   {"MAE": 5.29, "R2": -0.07},
    "CLIP-ViT-L14": {"MAE": 5.43, "R2": -0.17},
    # DINOv2 ViT-S/14 not distinguishable from filenames seen so far --
    # flag if found.
}


def inspect_checkpoint(path: Path):
    """
    Loads a .pt file WITHOUT assuming any model class, using
    torch.load(..., map_location='cpu'). Reports what's inside: a raw
    state_dict (just tensors), or a dict that also carries config/metadata
    (architecture hyperparameters, epoch, val score, etc.) -- the latter
    would let us reconstruct the model without guessing.
    """
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)

    print(f"\n=== {path.name} ===")
    print(f"  top-level type: {type(obj)}")

    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, "shape"):
                print(f"  '{k}': tensor shape={tuple(v.shape)} dtype={v.dtype}")
            elif isinstance(v, (int, float, str, bool)):
                print(f"  '{k}': {v!r}")
            elif isinstance(v, dict):
                print(f"  '{k}': dict with keys {list(v.keys())[:10]}"
                      f"{'...' if len(v) > 10 else ''}")
            else:
                print(f"  '{k}': {type(v)}")
    else:
        print(f"  (not a dict -- unexpected top-level structure)")

    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR)
    ap.add_argument("--inspect-only", action="store_true",
                     help="Just print the structure of a couple of "
                          "checkpoints; do not attempt to load a model or "
                          "run inference.")
    ap.add_argument("--backbone", default="ResNet-50",
                     help="Which p2_<backbone>_pea_mlp_f*.pt family to "
                          "inspect/evaluate")
    args = ap.parse_args()

    if not args.ckpt_dir.exists():
        raise SystemExit(f"[FATAL] {args.ckpt_dir} not found.")

    all_files = sorted(args.ckpt_dir.glob("*.pt"))
    print(f"[INFO] {len(all_files)} .pt files found in {args.ckpt_dir}")

    # Group by "family" (everything before the final _fNN.pt)
    import re
    families = {}
    for f in all_files:
        m = re.match(r"(.+)_f(\d+)\.pt$", f.name)
        if not m:
            print(f"  [WARN] filename doesn't match expected pattern: {f.name}")
            continue
        fam, fold = m.group(1), int(m.group(2))
        families.setdefault(fam, []).append((fold, f))

    print(f"\n[INFO] {len(families)} model families detected:")
    for fam, items in sorted(families.items()):
        folds = sorted(fo for fo, _ in items)
        print(f"  {fam}: {len(items)} checkpoints, folds={folds[:3]}...{folds[-3:]} "
              f"(min={min(folds)}, max={max(folds)})")
        if len(folds) != len(set(folds)):
            print(f"    [WARN] duplicate fold numbers detected -- possible "
                  f"seed repetition hiding in this family!")

    print(f"\n[KEY QUESTION] Do any families have MORE than 26 checkpoints, "
          f"or duplicate fold numbers? That would indicate multiple seeds "
          f"per fold. If every family has exactly 26 checkpoints with fold "
          f"numbers 1..26 and no duplicates, there is only ONE seed's worth "
          f"of checkpoints in this directory -- the 'Table 1 caption says "
          f"5 seeds' claim has no support from checkpoints_v6 alone.")

    if args.inspect_only:
        # Inspect a couple of checkpoints from the requested backbone family
        # to see if they carry enough metadata to reconstruct the model.
        matches = [fam for fam in families if args.backbone in fam]
        if not matches:
            print(f"\n[WARN] No family matched backbone '{args.backbone}'. "
                  f"Available families: {sorted(families.keys())}")
            return
        fam = matches[0]
        print(f"\n[INFO] Inspecting checkpoints from family: {fam}")
        for fold, f in sorted(families[fam])[:2]:
            inspect_checkpoint(f)
        return

    print("\n[INFO] --inspect-only not set, but full evaluation requires "
          "knowing the exact PEA-MLP architecture (layer sizes, dropout, "
          "activation) to reconstruct the nn.Module before loading the "
          "state_dict. Re-run with --inspect-only first; if the checkpoint "
          "dict includes architecture/config info, share it and the full "
          "evaluation logic can be added.")


if __name__ == "__main__":
    main()
