"""
run_ablation.py — LLM Ablation Study 메인 실행 스크립트
=========================================================
사용법:
    # 모든 모델 실행
    python run_ablation.py --models gpt4v gemini

    # 특정 모델만 실행
    python run_ablation.py --models gemini

    # 테스트 모드 (처음 5개 샘플만)
    python run_ablation.py --models gpt4v --max_samples 5

    # 이미 실행한 모델 결과로 시각화만
    python run_ablation.py --visualize_only
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TEST_JSONL, TRAIN_JSONL, RESULTS_DIR,
    REQUEST_DELAY_SEC, MODELS
)
from data_loader import load_jsonl, format_meta_text
from evaluator import compute_metrics, print_metrics_table, save_results, save_summary, save_csv
from visualize_results import load_results, make_summary_figure


# ─────────────────────────────────────────
# Few-shot 샘플 선택
# ─────────────────────────────────────────
def select_few_shot_examples(
    train_samples: List[dict],
    n_shots: int = 5,
    seed: int = 42,
) -> List[dict]:
    """
    Train 셋에서 CARS 점수 분포를 고르게 대표하는 n_shots개 샘플 선택.
    이미지가 없는 샘플은 제외.
    """
    valid = [s for s in train_samples if s["image_b64"] is not None]

    # CARS 점수 기준 정렬 후 균등 간격으로 선택 (stratified)
    valid_sorted = sorted(valid, key=lambda x: x["cars"])
    indices = np.linspace(0, len(valid_sorted) - 1, n_shots, dtype=int)
    selected = [valid_sorted[i] for i in indices]

    print(f"  Few-shot examples selected ({n_shots}샘플):")
    for i, s in enumerate(selected, 1):
        pid = s["meta"].get("participant_id", "?")
        print(f"    [{i}] PID={pid} CARS={s['cars']:.1f}")

    return [
        {
            "image_b64": s["image_b64"],
            "meta_text": format_meta_text(s["meta"]),
            "cars": s["cars"],
        }
        for s in selected
    ]


# ─────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────
def build_predictor(model_key: str):
    """model_key → 해당 Predictor 인스턴스 반환."""
    if model_key == "gpt4v":
        from models.gpt4v import GPT4VPredictor
        return GPT4VPredictor()
    elif model_key == "gemini":
        from models.gemini import GeminiPredictor
        return GeminiPredictor()
    else:
        raise ValueError(f"Unknown model key: '{model_key}'. Choose from: {list(MODELS.keys())}")


# ─────────────────────────────────────────
# Single model evaluation
# ─────────────────────────────────────────
def run_single_model(
    model_key: str,
    samples: List[dict],
    few_shot_examples: Optional[List[dict]] = None,
    max_samples: Optional[int] = None,
    n_shots: int = 0,
) -> dict:
    """
    한 모델로 전체 테스트셋을 추론하고 평가 지표를 반환.
    """
    display_name = MODELS[model_key]
    shot_info = f"{n_shots}-shot"
    print(f"\n{'='*60}")
    print(f"  Running: {display_name} [{shot_info}]")
    print(f"{'='*60}")

    predictor = build_predictor(model_key)

    if max_samples:
        samples = samples[:max_samples]
        print(f"  [DEBUG] Using first {max_samples} samples only")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # n_shots별로 별도 로그 파일 저장
    pred_log_path = os.path.join(RESULTS_DIR, f"{model_key}_{n_shots}shot_predictions.jsonl")

    y_true, y_pred = [], []

    with open(pred_log_path, "w", encoding="utf-8") as log_f:
        for i, sample in enumerate(samples):
            image_b64 = sample["image_b64"]
            cars_gt   = sample["cars"]
            meta      = sample["meta"]
            meta_text = format_meta_text(meta)

            print(f"  [{i+1:>3}/{len(samples)}] PID={meta.get('participant_id','?'):>4} | GT={cars_gt:.1f}", end="", flush=True)

            if image_b64 is None:
                print(" | SKIP (no image)")
                y_true.append(cars_gt)
                y_pred.append(None)
                continue

            t0   = time.time()
            pred = predictor.predict(
                image_b64=image_b64,
                meta_text=meta_text,
                few_shot_examples=few_shot_examples,
            )
            elapsed = time.time() - t0

            if pred is not None:
                err = abs(cars_gt - pred)
                print(f" | Pred={pred:.2f} | AE={err:.2f} | {elapsed:.1f}s")
            else:
                print(f" | Pred=FAILED | {elapsed:.1f}s")

            y_true.append(cars_gt)
            y_pred.append(pred)

            import json
            log_f.write(json.dumps({
                "idx":            i,
                "participant_id": meta.get("participant_id"),
                "cars_gt":        cars_gt,
                "cars_pred":      pred,
            }) + "\n")
            log_f.flush()

            if i < len(samples) - 1:
                time.sleep(REQUEST_DELAY_SEC)

    metrics = compute_metrics(y_true, y_pred, model_name=f"{display_name} ({shot_info})")
    save_results(metrics, model_key=f"{model_key}_{n_shots}shot")

    print(f"\n  → MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}"
          f"  R²={metrics['r2']:.3f}  Spearman={metrics['spearman']:.3f}"
          f"  (Failed={metrics['n_failed']}/{metrics['n_total']})")

    return metrics


# ─────────────────────────────────────────
# Resume from partial log
# ─────────────────────────────────────────
def load_partial_predictions(model_key: str, n_total: int):
    """
    이미 실행된 모델의 중간 저장 결과 로드.
    완료된 경우 (y_true, y_pred) 튜플, 미완성이면 None.
    """
    import json
    log_path = os.path.join(RESULTS_DIR, f"{model_key}_predictions.jsonl")
    if not os.path.exists(log_path):
        return None

    y_true, y_pred = [], []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            y_true.append(obj["cars_gt"])
            y_pred.append(obj["cars_pred"])

    if len(y_true) < n_total:
        print(f"  [INFO] Partial log found: {len(y_true)}/{n_total}. Continuing from checkpoint...")
        return None  # 미완성 → 재실행

    print(f"  [INFO] Found complete log for '{model_key}' ({len(y_true)} samples). Loading cached results.")
    return y_true, y_pred


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LLM Ablation Study for CARS Regression"
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="실행할 모델 목록 (기본: 전체)"
    )
    parser.add_argument(
        "--split", choices=["test", "train"], default="test",
        help="평가할 데이터 스플릿 (기본: test)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="디버그용: 최대 샘플 수 제한"
    )
    parser.add_argument(
        "--n_shots", type=int, nargs="+", default=[5],
        help="Few-shot 예시 수 (여러 개 가능, 예: --n_shots 0 3 5 10)"
    )
    parser.add_argument(
        "--visualize_only", action="store_true",
        help="이미 저장된 결과로 시각화만 실행"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="이미 완료된 모델은 건너뛰고 결과 로드"
    )
    args = parser.parse_args()

    # ── 시각화만 실행 ──
    if args.visualize_only:
        all_results = load_results(RESULTS_DIR)
        if not all_results:
            print(f"No results found in '{RESULTS_DIR}'. Run models first.")
            return
        print_metrics_table(all_results)
        fig_path = os.path.join(RESULTS_DIR, "ablation_figure.png")
        make_summary_figure(all_results, fig_path)
        return

    # ── 데이터 로드 ──
    jsonl_path = TEST_JSONL if args.split == "test" else TRAIN_JSONL
    print(f"\nLoading {args.split} data from: {jsonl_path}")
    samples = load_jsonl(jsonl_path, encode_images=True)

    if not samples:
        print("No samples loaded. Check your JSONL path and IMAGE_ROOT_REMAP in config.py")
        sys.exit(1)

    # ── Train 데이터 로드 (few-shot용, 0-shot만 있으면 스킵) ──
    train_samples = None
    if any(n > 0 for n in args.n_shots):
        print(f"\nLoading train data for few-shot examples...")
        train_samples = load_jsonl(TRAIN_JSONL, encode_images=True)

    # ── 멀티 shot × 멀티 모델 실행 ──
    all_results = []

    for n_shots in args.n_shots:
        print(f"\n{'#'*60}")
        print(f"  n_shots = {n_shots}")
        print(f"{'#'*60}")

        # few-shot 예시 준비
        if n_shots > 0 and train_samples:
            few_shot_examples = select_few_shot_examples(train_samples, n_shots=n_shots)
        else:
            few_shot_examples = None
            print("  Zero-shot mode")

        for model_key in args.models:
            # 기존 완료 결과 확인 (--skip_existing)
            if args.skip_existing:
                log_key = f"{model_key}_{n_shots}shot"
                cached = load_partial_predictions(log_key, len(samples))
                if cached is not None:
                    y_true, y_pred = cached
                    display_name = f"{MODELS[model_key]} ({n_shots}-shot)"
                    metrics = compute_metrics(y_true, y_pred, model_name=display_name)
                    metrics["n_shots"] = n_shots
                    metrics["model_key"] = model_key
                    all_results.append(metrics)
                    continue

            metrics = run_single_model(
                model_key=model_key,
                samples=samples,
                few_shot_examples=few_shot_examples,
                max_samples=args.max_samples,
                n_shots=n_shots,
            )
            metrics["n_shots"] = n_shots
            metrics["model_key"] = model_key
            all_results.append(metrics)

    # ── 최종 결과 출력 및 저장 ──
    if all_results:
        print_metrics_table(all_results)
        save_summary(all_results)
        save_csv(all_results)

        fig_path = os.path.join(RESULTS_DIR, "ablation_figure.png")
        make_summary_figure(all_results, fig_path)
        print(f"\nDone. Results saved to '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()