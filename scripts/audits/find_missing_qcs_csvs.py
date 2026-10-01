# -*- coding: utf-8 -*-
"""
find_missing_qcs_csvs.py
---------------------------
Recursively searches BASE_DIR for every CSV whose filename suggests it's a
per-image QCS feature file (contains "qcs" and "result", case-insensitive),
and reports its full path plus which CONFIG_FOLDERS entry it most likely
belongs to (by checking whether the CSV's own Filename column overlaps
with images actually present in that config's folder).

This exists because apply_independent_pool_to_configs.py's simple
folder-local glob missed 11 of 18 configurations -- their CSVs exist
somewhere (you uploaded several of them earlier: QCS_results_RAUNE___
Rule_based.csv etc.), just not in the exact folder-local location the
current finder function checks. Run this FIRST to see where they actually
are, then we'll fix the path-finding logic in apply_independent_pool_to_
configs.py to match reality instead of guessing.

Usage:
    python find_missing_qcs_csvs.py
"""

import os
import re
import pandas as pd

BASE_DIR = r"C:\Users\zxzxk\OneDrive - University of Georgia\Coral_Project_Results"

CONFIG_FOLDERS = [
    "Raw_Input",
    "Retinex_+_LLaVA",
    "Retinex_+_Rule_based",
    "Retinex_w_o_post",
    "Retinex_plain_w_phys=0",
    "Retinex_+_Default_MHF-UIE",
    "RAUNE_+_LLaVA",
    "RAUNE_+_Rule_based",
    "RAUNE_w_o_post",
    "RAUNE_plain_w_phys=0",
    "RAUNE_+_Default_MHF-UIE",
    "RAUNE_unsupervised",
    "RAUNE_lsup=0.0",
    "RAUNE_lsup=0.1",
    "RAUNE_lsup=0.5",
    "RAUNE_lsup=1.0",
    "CCL_NET",
    "NUCE",
]

ALREADY_FOUND = {
    "Raw_Input", "Retinex_+_LLaVA", "Retinex_w_o_post",
    "Retinex_+_Default_MHF-UIE", "RAUNE_+_LLaVA", "RAUNE_w_o_post",
    "RAUNE_+_Default_MHF-UIE",
}

EXCLUDE_KEYWORDS = ["summary", "bootstrap", "wilcoxon", "component", "post_metrics"]


def normalize(name):
    """Loose normalization for fuzzy matching folder names to filenames:
    lowercase, strip non-alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def main():
    print("=" * 90)
    print("  Recursive search for qcs_results CSVs under BASE_DIR")
    print("=" * 90)
    print(f"  BASE_DIR = {BASE_DIR}\n")

    all_csvs = []
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            name_lower = f.lower()
            if "qcs" not in name_lower or "result" not in name_lower:
                continue
            if any(kw in name_lower for kw in EXCLUDE_KEYWORDS):
                continue
            full_path = os.path.join(root, f)
            all_csvs.append(full_path)

    print(f"Found {len(all_csvs)} candidate CSVs total (all configs, not just missing ones):\n")
    for p in sorted(all_csvs):
        rel = os.path.relpath(p, BASE_DIR)
        print(f"  {rel}")

    missing = [c for c in CONFIG_FOLDERS if c not in ALREADY_FOUND]
    print("\n" + "=" * 90)
    print(f"  Matching against the {len(missing)} MISSING configs")
    print("=" * 90)

    for cfg in missing:
        cfg_norm = normalize(cfg)
        print(f"\n[{cfg}]")

        # Strategy 1: CSV physically located anywhere inside that config's folder
        folder_path = os.path.join(BASE_DIR, cfg)
        inside = [p for p in all_csvs if os.path.commonpath([p, folder_path]) == os.path.normpath(folder_path)] \
            if os.path.isdir(folder_path) else []
        if inside:
            print(f"  [inside folder] {inside}")
            continue

        # Strategy 2: filename contains a normalized substring match to the config name
        candidates = []
        for p in all_csvs:
            fname_norm = normalize(os.path.splitext(os.path.basename(p))[0])
            if cfg_norm in fname_norm or fname_norm in cfg_norm:
                candidates.append(p)
        if candidates:
            print(f"  [name-matched candidates] {candidates}")
            continue

        # Strategy 3: check Filename-column overlap against images actually
        # present in that config's folder (most reliable but slower --
        # only try this for remaining unmatched configs)
        if os.path.isdir(folder_path):
            folder_stems = set(
                os.path.splitext(f)[0].lower()
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            best_match, best_overlap = None, 0
            for p in all_csvs:
                try:
                    df = pd.read_csv(p, usecols=["Filename"], nrows=200)
                except Exception:
                    continue
                stems = set(os.path.splitext(str(x))[0].lower() for x in df["Filename"])
                overlap = len(stems & folder_stems)
                if overlap > best_overlap:
                    best_overlap, best_match = overlap, p
            if best_match and best_overlap > 5:
                print(f"  [content-matched, {best_overlap}/200 sampled filenames overlap] {best_match}")
                continue

        print("  [NOT FOUND] -- no CSV located by any strategy. This config likely has no")
        print("  per-image QCS features computed yet, or its folder doesn't exist under this name.")
        if not os.path.isdir(folder_path):
            print(f"  (folder does not exist: {folder_path})")


if __name__ == "__main__":
    main()
