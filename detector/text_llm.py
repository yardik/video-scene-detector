"""Text-only LLM Classifier for Dual-Model Scene Detection.

Decouples vision (Qwen2-VL describing the scene) from reasoning (text LLM
evaluating if the description matches the target scene prompt).
"""

import sys
import os
import time
import gc
import torch
from typing import Optional, Tuple, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SUGGESTED_TEXT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# No hardcoded default: if not specified, get_cached_text_llm() picks the first text
# model already installed in models_cache/hub/ (see detector.video_utils.scan_installed_models).
DEFAULT_TEXT_MODEL_ID = None

_cached_text_model_instance: Optional["TextLLMClassifier"] = None
_cached_text_model_key: Optional[Tuple[str, Optional[str], bool]] = None


class TextLLMClassifier:
    """Loads and manages a text-only LLM for evaluating scene descriptions."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = DEFAULT_TEXT_MODEL_ID,
        quantization: Optional[str] = "4bit",
        device: str = "auto",
        use_sage_attention: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.quantization = quantization
        self.device = device
        self.use_sage_attention = use_sage_attention
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        import threading
        self._load_lock = threading.Lock()

    def unload(self):
        """
        Frees GPU VRAM by dropping the model and tokenizer. Any subsequent classify_*
        call auto-reloads via its existing `if not self.is_loaded: self.load_model()` guard —
        callers don't need to call load_model() explicitly afterward.
        """
        with self._load_lock:
            if not self.is_loaded:
                return
            print(f"💤 Unloading Text LLM from VRAM...", file=sys.stderr)
            if self.model is not None:
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            self.is_loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self):
        """Loads the text model and tokenizer into GPU memory."""
        with self._load_lock:
            if self.is_loaded:
                return

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from .video_utils import resolve_model_path
        self.model_name_or_path = resolve_model_path(self.model_name_or_path)

        print(f"📦 Loading Text LLM: {self.model_name_or_path} (Quant: {self.quantization})...", file=sys.stderr)

        attn_impl = "sdpa"
        if self.use_sage_attention:
            from .sage_patcher import enable_sage_attention
            enable_sage_attention()
        else:
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                print("⚡ [Text LLM] FlashAttention-2 active!", file=sys.stderr)
            except ImportError:
                attn_impl = "sdpa"
                print("⚡ [Text LLM] PyTorch Native SDPA (Scaled Dot-Product Attention) active on Tensor Cores.", file=sys.stderr)

        kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "attn_implementation": attn_impl,
        }

        if torch.cuda.is_available():
            if self.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["quantization_config"] = bnb_config
                kwargs["device_map"] = {"": 0}
            else:
                kwargs["device_map"] = {"": 0}
        else:
            kwargs["device_map"] = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            **kwargs,
        )

        self.model.eval()
        self.is_loaded = True
        print(f"✅ Text LLM loaded successfully into memory.", file=sys.stderr)

    def classify_description(
        self,
        description: str,
        target_prompt: str,
        threshold: float = 50.0,
    ) -> Tuple[bool, str]:
        """Evaluates whether a video scene description matches the target prompt by calculating
        a similarity probability percentage. If probability >= threshold (default 80%), returns True.

        Returns:
            Tuple[bool, str]: (is_match, reasoning_explanation)
        """
        if not self.is_loaded:
            self.load_model()

        import re

        from detector.prompts import get_matcher_prompt
        system_msg = get_matcher_prompt()

        user_content = (
            f"--- VIDEO SCENE DESCRIPTION ---\n"
            f"{description}\n\n"
            f"--- TARGET SCENE PROMPT ---\n"
            f"{target_prompt}\n\n"
            f"Evaluate phrasing similarities and visual alignment between the video description and target prompt, then estimate the match probability percentage (0-100%)."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)

        t0 = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1,
            )
        inference_time = time.time() - t0

        output_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        response_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Parse match probability percentage
        prob_val = 0.0
        prob_match = re.search(r"PROBABILITY:\s*(\d{1,3})\s*%?", response_text, re.IGNORECASE)
        if prob_match:
            prob_val = float(prob_match.group(1))
        else:
            # Fallback: search any number followed by %
            pct_matches = re.findall(r"(\d{1,3})\s*%", response_text)
            if pct_matches:
                prob_val = float(pct_matches[0])
            else:
                # Fallback to YES/NO search
                if "YES" in response_text.upper():
                    prob_val = 85.0
                else:
                    prob_val = 0.0

        prob_val = max(0.0, min(100.0, prob_val))
        is_match = prob_val >= threshold

        summary_header = f"📊 Match Probability: {prob_val:.0f}% (Threshold: {threshold:.0f}% → {'MATCH ✓' if is_match else 'NO MATCH ✗'}) (Text LLM Time: {inference_time:.2f}s)"
        full_reasoning = f"{summary_header}\n\n{response_text}"

        print(f"[Text LLM Classifier] {summary_header}", file=sys.stderr)
        return is_match, full_reasoning

    def classify_descriptions_batch(
        self,
        descriptions: List[str],
        target_prompts: List[str],
        threshold: float = 80.0,
        batch_size: int = 4,
    ) -> List[Tuple[bool, str]]:
        """Batch evaluates multiple (description, target_prompt) pairs in micro-batches (default 4)
        to maintain minimal VRAM usage while achieving maximum GPU parallel speedup.

        Returns:
            List[Tuple[bool, str]]: List of (is_match, reasoning_explanation) tuples.
        """
        if not descriptions:
            return []

        if not self.is_loaded:
            self.load_model()

        import re

        # Ensure tokenizer padding side is set to left for batch generation
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        from detector.prompts import get_matcher_prompt
        system_msg = get_matcher_prompt()

        all_results = []
        total_items = len(descriptions)

        # Micro-batch chunking (process in chunks of batch_size)
        for chunk_idx in range(0, total_items, batch_size):
            chunk_descs = descriptions[chunk_idx : chunk_idx + batch_size]
            chunk_prompts = target_prompts[chunk_idx : chunk_idx + batch_size]

            formatted_prompts = []
            for desc, target_prompt in zip(chunk_descs, chunk_prompts):
                user_content = (
                    f"--- VIDEO SCENE DESCRIPTION ---\n"
                    f"{desc}\n\n"
                    f"--- TARGET SCENE PROMPT ---\n"
                    f"{target_prompt}\n\n"
                    f"Evaluate phrasing similarities and visual alignment between the video description and target prompt, then estimate the match probability percentage (0-100%)."
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_content},
                ]
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted_prompts.append(prompt_text)

            inputs = self.tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(self.model.device)

            t0 = time.time()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=False,
                    repetition_penalty=1.1,
                )
            total_inference_time = time.time() - t0
            per_item_time = total_inference_time / max(len(chunk_descs), 1)

            print(f"[Text LLM Batch {chunk_idx+1}-{min(chunk_idx+batch_size, total_items)}/{total_items}] Processed in {total_inference_time:.2f}s ({per_item_time:.2f}s / item)", file=sys.stderr)

            output_ids = [
                out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
            ]
            decoded_responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            del inputs, generated_ids, output_ids, formatted_prompts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for response_text in decoded_responses:
                response_text = response_text.strip()
                prob_val = 0.0
                prob_match = re.search(r"PROBABILITY:\s*(\d{1,3})\s*%?", response_text, re.IGNORECASE)
                if prob_match:
                    prob_val = float(prob_match.group(1))
                else:
                    pct_matches = re.findall(r"(\d{1,3})\s*%", response_text)
                    if pct_matches:
                        prob_val = float(pct_matches[0])
                    else:
                        if "YES" in response_text.upper():
                            prob_val = 85.0
                        else:
                            prob_val = 0.0

                prob_val = max(0.0, min(100.0, prob_val))
                is_match = prob_val >= threshold

                summary_header = f"📊 Match Probability: {prob_val:.0f}% (Threshold: {threshold:.0f}% → {'MATCH ✓' if is_match else 'NO MATCH ✗'}) (Batch Avg: {per_item_time:.2f}s)"
                full_reasoning = f"{summary_header}\n\n{response_text}"
                all_results.append((is_match, full_reasoning))

        return all_results



def get_cached_text_llm(
    model_name_or_path: Optional[str] = DEFAULT_TEXT_MODEL_ID,
    quantization: Optional[str] = "4bit",
    use_sage_attention: bool = True,
) -> TextLLMClassifier:
    """
    Returns a cached TextLLMClassifier instance, re-loading only if settings change.

    If model_name_or_path is not given, uses the first text model already installed in
    models_cache/hub/ (see detector.video_utils.scan_installed_models). Raises ValueError if
    none is installed yet.
    """
    global _cached_text_model_instance, _cached_text_model_key

    if not model_name_or_path:
        from .video_utils import pick_default_installed_model
        model_name_or_path = pick_default_installed_model("text")
        if not model_name_or_path:
            raise ValueError(
                "No text model specified and none installed in models_cache/hub/. "
                f"Pass --text-model / model_name_or_path, or download one first (e.g. '{SUGGESTED_TEXT_MODEL_ID}')."
            )
        print(f"⚡ No text model specified — using first installed model: '{model_name_or_path}'", file=sys.stderr)

    current_key = (model_name_or_path, quantization, use_sage_attention)

    if (
        _cached_text_model_instance is not None
        and _cached_text_model_key == current_key
        and _cached_text_model_instance.is_loaded
    ):
        print(f"⚡ Reusing cached GPU Text LLM ({model_name_or_path})", file=sys.stderr)
        return _cached_text_model_instance

    if _cached_text_model_instance is not None:
        print(f"🔄 Text LLM settings changed. Unloading old model from VRAM...", file=sys.stderr)
        try:
            del _cached_text_model_instance.model
            del _cached_text_model_instance.tokenizer
            _cached_text_model_instance.is_loaded = False
        except Exception as e:
            print(f"Warning during text model cleanup: {e}", file=sys.stderr)
        _cached_text_model_instance = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    instance = TextLLMClassifier(
        model_name_or_path=model_name_or_path,
        quantization=quantization,
        use_sage_attention=use_sage_attention,
    )
    # Lazy loading: model weights will be loaded when classify_descriptions_batch() is called.

    _cached_text_model_instance = instance
    _cached_text_model_key = current_key

    return instance
