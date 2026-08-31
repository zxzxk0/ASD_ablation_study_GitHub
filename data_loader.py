"""
data_loader.py — JSONL 데이터 로딩 및 이미지 Base64 인코딩
"""

import json
import base64
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import IMAGE_ROOT_REMAP, CARS_MIN, CARS_MAX


def remap_image_path(raw_path: str) -> str:
    """
    JSONL 내 Windows 절대경로 → 로컬 실제 경로로 변환.
    config.IMAGE_ROOT_REMAP 설정 기반.
    """
    if IMAGE_ROOT_REMAP is None:
        return raw_path

    src = IMAGE_ROOT_REMAP["from"].replace("\\", "/")
    dst = IMAGE_ROOT_REMAP["to"]

    normalized = raw_path.replace("\\", "/")
    if normalized.startswith(src):
        relative = normalized[len(src):].lstrip("/")
        return os.path.join(dst, relative)
    return normalized


def encode_image_base64(image_path: str) -> Optional[str]:
    """
    이미지 파일을 Base64 문자열로 인코딩.
    파일이 없거나 읽기 실패 시 None 반환.
    """
    resolved = remap_image_path(image_path)
    try:
        with open(resolved, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"  [WARNING] Image not found: {resolved}")
        return None
    except Exception as e:
        print(f"  [WARNING] Failed to read image {resolved}: {e}")
        return None


def load_jsonl(path: str, encode_images: bool = True) -> List[Dict[str, Any]]:
    """
    JSONL 파일 로드.

    각 샘플:
        {
            "image_path": str,           # 원본 경로 (참고용)
            "image_b64": str | None,     # Base64 인코딩된 이미지
            "cars": float,               # Ground truth CARS 점수
            "meta": {                    # 참가자 메타데이터
                "participant_id": str,
                "age": float,
                "gender": str,           # "M" | "F"
            }
        }
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARNING] Line {line_no} JSON parse error: {e}")
                continue

            image_path = obj.get("image", "")
            cars       = float(obj.get("cars", 0.0))
            meta       = obj.get("meta", {})

            # CARS 범위 클리핑
            cars = max(CARS_MIN, min(CARS_MAX, cars))

            image_b64 = None
            if encode_images and image_path:
                image_b64 = encode_image_base64(image_path)

            samples.append({
                "image_path": image_path,
                "image_b64":  image_b64,
                "cars":       cars,
                "meta":       meta,
            })

    print(f"  Loaded {len(samples)} samples from '{path}'")
    missing = sum(1 for s in samples if s["image_b64"] is None)
    if missing:
        print(f"  [WARNING] {missing} samples have missing images")

    return samples


def format_meta_text(meta: Dict[str, Any]) -> str:
    """메타데이터를 텍스트 설명으로 변환 (프롬프트 삽입용)."""
    age    = meta.get("age", "unknown")
    gender = "Male" if meta.get("gender", "").upper() == "M" else "Female"
    pid    = meta.get("participant_id", "N/A")
    return f"Participant ID: {pid}, Age: {age} years, Gender: {gender}"
