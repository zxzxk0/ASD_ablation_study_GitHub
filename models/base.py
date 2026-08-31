"""
models/base.py — 모든 VLM 모델의 공통 추상 베이스 클래스
"""

import re
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from config import CARS_MIN, CARS_MAX, MAX_RETRIES, RETRY_DELAY_SEC


# ─────────────────────────────────────────────────────────
# Shared CARS Regression Prompt
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert clinical AI assistant specializing in \
Autism Spectrum Disorder (ASD) severity assessment.

You will be given eye-tracking scanpath images of children with ASD and \
their demographic information. The scanpath image visualizes the child's gaze \
trajectory while viewing social stimuli.

Your task is to predict the Childhood Autism Rating Scale (CARS) score for \
a child based on the gaze pattern and metadata.

CARS Score Guidelines:
- Range: 15 to 60
- 15–29: Non-autistic / minimal symptoms
- 30–36: Mild to Moderate ASD
- 37–60: Severe ASD

Analyze the scanpath image carefully:
- Fixation density and distribution across the image
- Gaze trajectory structure and coherence
- Attention to social vs. non-social regions
- Scanpath complexity and regularity

You will first see several EXAMPLE cases with known CARS scores to calibrate \
your predictions. Then you will predict the score for a NEW case.

IMPORTANT: You MUST respond with ONLY a JSON object in the following format \
(no additional text, no markdown, no explanation):
{"cars_score": <float between 15.0 and 60.0>}"""

FEW_SHOT_EXAMPLE_INTRO = """Below are {n} reference examples from the training set. \
Study the relationship between the scanpath patterns and their CARS scores carefully."""

FEW_SHOT_EXAMPLE_TEMPLATE = """--- Example {idx} ---
Clinical Information: {meta_text}
Known CARS Score: {cars:.1f}
Scanpath Image: [see image above]"""

USER_PROMPT_TEMPLATE = """--- NEW CASE (predict this) ---
Clinical Information:
{meta_text}

Based on the reference examples above, analyze this new scanpath image and \
predict the CARS score.

Respond with ONLY the JSON object: {{"cars_score": <value>}}"""

USER_PROMPT_TEMPLATE_ZEROSHOT = """Clinical Information:
{meta_text}

Please analyze the eye-tracking scanpath image above and predict the CARS \
score for this participant.

Respond with ONLY the JSON object: {{"cars_score": <value>}}"""


# ─────────────────────────────────────────────────────────
# Base Class
# ─────────────────────────────────────────────────────────
class BaseVLMPredictor(ABC):
    """
    모든 VLM 기반 CARS 예측기의 추상 베이스.
    서브클래스는 _call_api() 만 구현하면 됩니다.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def _call_api(
        self,
        image_b64: str,
        user_prompt: str,
        system_prompt: str,
        few_shot_examples: Optional[list] = None,
    ) -> str:
        """
        실제 API 호출. 모델의 원시 텍스트 응답을 반환.
        few_shot_examples: [{"image_b64": str, "meta_text": str, "cars": float}, ...]
        """
        ...

    def predict(
        self,
        image_b64: str,
        meta_text: str,
        few_shot_examples: Optional[list] = None,
    ) -> Optional[float]:
        """
        단일 샘플에 대한 CARS 점수 예측.
        few_shot_examples: [{"image_b64": str, "meta_text": str, "cars": float}, ...]
        실패 시 None 반환.
        """
        if few_shot_examples:
            user_prompt = USER_PROMPT_TEMPLATE.format(meta_text=meta_text)
        else:
            user_prompt = USER_PROMPT_TEMPLATE_ZEROSHOT.format(meta_text=meta_text)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw_response = self._call_api(
                    image_b64=image_b64,
                    user_prompt=user_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    few_shot_examples=few_shot_examples,
                )
                score = self._parse_score(raw_response)
                if score is not None:
                    return score
                print(f"    [WARN] Attempt {attempt}: parse failed. Raw: {raw_response[:200]!r}")
            except Exception as e:
                print(f"    [ERROR] Attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SEC * attempt)

        return None

    @staticmethod
    def _parse_score(text: str) -> Optional[float]:
        """
        모델 응답에서 CARS 점수를 추출.
        JSON 파싱 우선, 실패 시 정규식 fallback.
        """
        import json

        # 1차: JSON 파싱 시도
        try:
            # 마크다운 코드 블록 제거
            cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
            obj = json.loads(cleaned)
            if "cars_score" in obj:
                score = float(obj["cars_score"])
                return max(CARS_MIN, min(CARS_MAX, score))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 2차: "cars_score": 숫자 패턴 매칭
        match = re.search(r'"cars_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        if match:
            score = float(match.group(1))
            return max(CARS_MIN, min(CARS_MAX, score))

        # 3차: 범위 내 첫 번째 숫자 추출
        numbers = re.findall(r'\b([1-5][0-9](?:\.[0-9]+)?|60(?:\.0+)?)\b', text)
        if numbers:
            score = float(numbers[0])
            if CARS_MIN <= score <= CARS_MAX:
                return score

        return None