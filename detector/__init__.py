"""
Qwen2-VL Video Scene Detector Package
"""

from .scene_detector import VideoSceneDetector, SceneMatch
from .qwen_model import QwenVLModel, get_cached_qwen_model
from .text_llm import TextLLMClassifier, get_cached_text_llm, DEFAULT_TEXT_MODEL_ID
from .video_utils import VideoUtils
from .prompt_analyzer import PromptAnalyzer, PromptAnalysis
from .batch_processor import BatchProcessor, BatchProgress
from .prompts import (
    get_captioner_prompt,
    get_matcher_prompt,
    save_captioner_prompt,
    save_matcher_prompt,
    DEFAULT_CAPTIONER_PROMPT,
    DEFAULT_MATCHER_PROMPT,
)

__all__ = [
    "VideoSceneDetector",
    "QwenVLModel",
    "get_cached_qwen_model",
    "TextLLMClassifier",
    "get_cached_text_llm",
    "DEFAULT_TEXT_MODEL_ID",
    "VideoUtils",
    "SceneMatch",
    "PromptAnalyzer",
    "PromptAnalysis",
    "BatchProcessor",
    "BatchProgress",
    "get_captioner_prompt",
    "get_matcher_prompt",
    "save_captioner_prompt",
    "save_matcher_prompt",
    "DEFAULT_CAPTIONER_PROMPT",
    "DEFAULT_MATCHER_PROMPT",
]


