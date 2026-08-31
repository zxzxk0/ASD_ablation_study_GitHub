"""
models/gemini.py — Google Gemini Vision CARS 예측기 (Few-shot 지원)
google-genai 패키지 사용 (신버전)
"""

import base64
from typing import Optional
from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, MODELS, MAX_TOKENS, TEMPERATURE
from models.base import (
    BaseVLMPredictor, SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLE_INTRO, FEW_SHOT_EXAMPLE_TEMPLATE,
)


class GeminiPredictor(BaseVLMPredictor):
    """
    Google Gemini Vision 기반 CARS 회귀 예측기 (Few-shot 지원).
    """

    def __init__(self):
        super().__init__(model_name=MODELS["gemini"])
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def _call_api(
        self,
        image_b64: str,
        user_prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        few_shot_examples: Optional[list] = None,
    ) -> str:
        """
        few_shot_examples: [{"image_b64": str, "meta_text": str, "cars": float}, ...]
        Gemini multi-turn contents 방식으로 few-shot 구성.
        """
        contents = []

        # ── Few-shot 예시 삽입 ──
        if few_shot_examples:
            intro = FEW_SHOT_EXAMPLE_INTRO.format(n=len(few_shot_examples))

            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=intro)],
            ))
            contents.append(types.Content(
                role="model",
                parts=[types.Part(text="Understood. I will study these examples carefully to calibrate my predictions.")],
            ))

            for idx, ex in enumerate(few_shot_examples, 1):
                ex_text = FEW_SHOT_EXAMPLE_TEMPLATE.format(
                    idx=idx,
                    meta_text=ex["meta_text"],
                    cars=ex["cars"],
                )
                img_bytes = base64.b64decode(ex["image_b64"])
                contents.append(types.Content(
                    role="user",
                    parts=[
                        types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)),
                        types.Part(text=ex_text),
                    ],
                ))
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=f'{{"cars_score": {ex["cars"]:.1f}}}')],
                ))

        # ── 새 샘플 예측 요청 ──
        img_bytes = base64.b64decode(image_b64)
        contents.append(types.Content(
            role="user",
            parts=[
                types.Part(inline_data=types.Blob(mime_type="image/png", data=img_bytes)),
                types.Part(text=user_prompt),
            ],
        ))

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            ),
        )

        try:
            return response.text or ""
        except Exception:
            return ""