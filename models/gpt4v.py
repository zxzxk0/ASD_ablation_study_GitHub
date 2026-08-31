"""
models/gpt4v.py — OpenAI GPT-4o (Vision) CARS 예측기 (Few-shot 지원)
"""

from typing import Optional
from openai import OpenAI
from config import OPENAI_API_KEY, MODELS, MAX_TOKENS, TEMPERATURE
from models.base import (
    BaseVLMPredictor, SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLE_INTRO, FEW_SHOT_EXAMPLE_TEMPLATE,
)


class GPT4VPredictor(BaseVLMPredictor):
    """
    OpenAI GPT-4o Vision 기반 CARS 회귀 예측기 (Few-shot 지원).
    """

    def __init__(self):
        super().__init__(model_name=MODELS["gpt4v"])
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _call_api(
        self,
        image_b64: str,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        few_shot_examples: Optional[list] = None,
    ) -> str:
        """
        few_shot_examples: [{"image_b64": str, "meta_text": str, "cars": float}, ...]
        OpenAI multi-turn 방식으로 few-shot 구성:
          user: 예시 이미지 + 메타 → assistant: {"cars_score": xx} 반복
          user: 새 이미지 + 예측 요청
        """
        messages = [{"role": "system", "content": system_prompt}]

        # ── Few-shot 예시 삽입 ──
        if few_shot_examples:
            intro = FEW_SHOT_EXAMPLE_INTRO.format(n=len(few_shot_examples))
            messages.append({"role": "user", "content": intro})
            messages.append({
                "role": "assistant",
                "content": "Understood. I will study these examples carefully to calibrate my predictions.",
            })

            for idx, ex in enumerate(few_shot_examples, 1):
                ex_text = FEW_SHOT_EXAMPLE_TEMPLATE.format(
                    idx=idx,
                    meta_text=ex["meta_text"],
                    cars=ex["cars"],
                )
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{ex['image_b64']}",
                                "detail": "low",   # 예시는 low로 토큰 절약
                            },
                        },
                        {"type": "text", "text": ex_text},
                    ],
                })
                messages.append({
                    "role": "assistant",
                    "content": f'{{"cars_score": {ex["cars"]:.1f}}}',
                })

        # ── 새 샘플 예측 요청 ──
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}",
                        "detail": "high",
                    },
                },
                {"type": "text", "text": user_prompt},
            ],
        })

        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=messages,
        )
        return response.choices[0].message.content or ""