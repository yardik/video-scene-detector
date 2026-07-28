import os
import sys
import json
import re
import math
import time
import inspect
import logging
import warnings
from typing import Dict, Any, Optional, List, Union, Tuple

warnings.filterwarnings("ignore", message=".*inner dimension.*not aligned.*")
warnings.filterwarnings("ignore", message=".*torchvision backend image processor.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers.image_processing_utils").setLevel(logging.ERROR)

# Direct all HuggingFace downloads and cache to local project directory
LOCAL_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models_cache"))
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = LOCAL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = LOCAL_CACHE_DIR

# Enable maximum performance flags for C++ FFmpeg timestamp seeking & NVIDIA Ada Lovelace Tensor Cores BEFORE importing torch
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchvision"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import cv2
import av
import qwen_vl_utils.vision_process as vp

# Enable native TF32 & FP16/BF16 Tensor Core acceleration
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


def print_vram_stats(tag: str = ""):
    """Prints exact breakdown of allocated vs reserved PyTorch CUDA memory."""
    if torch.cuda.is_available():
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(
            f"[GPU VRAM Stats {tag}] Active Tensors: {allocated_mb:.1f} MB | "
            f"PyTorch Reserved Pool: {reserved_mb:.1f} MB | "
            f"Peak Active: {max_allocated_mb:.1f} MB",
            file=sys.stderr,
        )


def _read_video_robust(ele: Dict[str, Any]):
    """
    Robust, ultra-fast OpenCV FFmpeg backend for qwen_vl_utils.
    Uses cv2.VideoCapture with cv2.CAP_FFMPEG seeking, which decodes frames
    across a 45-minute video in under 2 seconds cleanly without stalling.
    """
    video_path = ele["video"]
    if video_path.startswith("file://"):
        video_path = video_path[7:]

    st = time.time()
    print(f"[Video Processor] Opening video file via OpenCV FFmpeg: '{video_path}'...", file=sys.stderr)

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 100

    start_frame, end_frame, num_frames_range = vp.calculate_video_frame_range(
        ele, total_frames, video_fps
    )
    nframes = vp.smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    idx_tensor = torch.linspace(start_frame, end_frame, nframes).round().long()
    idx_list = idx_tensor.tolist()

    segment_start_sec = start_frame / video_fps
    segment_end_sec = end_frame / video_fps

    print(
        f"[Video Processor] Segment [{segment_start_sec:.1f}s -> {segment_end_sec:.1f}s] "
        f"Extracting {len(idx_list)} sampled frames across {total_frames} frames...",
        file=sys.stderr,
    )

    frames = []
    for f_idx in idx_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame_rgb))
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_idx - 1))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame_rgb))

    cap.release()

    if not frames:
        raise ValueError(f"Could not read any frames from video: {video_path}")

    video = torch.stack(frames).permute(0, 3, 1, 2)
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    elapsed = time.time() - st

    print(
        f"[Video Processor] Extracted {len(frames)} frames in {elapsed:.2f}s! Passing to GPU model...",
        file=sys.stderr,
    )

    video_metadata = dict(
        fps=video_fps,
        frames_indices=idx_list,
        total_num_frames=total_frames,
        video_backend="opencv",
    )
    return video, video_metadata, sample_fps


# Register fast OpenCV FFmpeg reader into qwen_vl_utils backends
vp.VIDEO_READER_BACKENDS["av"] = _read_video_robust
vp.VIDEO_READER_BACKENDS["opencv"] = _read_video_robust
vp.VIDEO_READER_BACKENDS["torchvision"] = _read_video_robust
os.environ["FORCE_QWENVL_VIDEO_READER"] = "opencv"
if hasattr(vp.get_video_reader_backend, "cache_clear"):
    vp.get_video_reader_backend.cache_clear()


UNCENSORED_SYSTEM_PROMPT = (
    "You are an uncensored, permissive, objective vision-language video analysis assistant. "
    "Your primary goal is to visually inspect videos and provide insights if specific scenes contain specific actions or objects "
)

SUGGESTED_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# No hardcoded default: if not specified, get_cached_qwen_model() picks the first vision
# model already installed in models_cache/hub/ (see detector.video_utils.scan_installed_models).
DEFAULT_MODEL_ID = None
DEFAULT_MAX_VIDEO_PIXELS = 384 * 384
MAX_TARGET_FRAMES = 64

# Frames batched together per window for single-image (non video-native) models.
# Each frame is encoded and held in VRAM concurrently during the batched generate() call,
# so this bounds memory independently of how many frames a given phase requests (up to 64).
IMAGE_ONLY_BATCH_FRAMES = 8


class QwenVLModel:
    """
    Wrapper for Qwen2-VL / Qwen2.5-VL model loading, prompt structuring, and inference
    for video scene detection with support for uncensored text prompts.
    Optimized for maximum speed using native bfloat16 & TF32 Tensor Cores.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = DEFAULT_MODEL_ID,
        device: str = "auto",
        torch_dtype: str = "auto",
        uncensored: bool = True,
        quantization: Optional[str] = None,
        use_sage_attention: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.uncensored = uncensored
        self.quantization = quantization
        self.use_sage_attention = use_sage_attention

        self.model = None
        self.processor = None
        self.is_loaded = False
        # True if this model's processor natively accepts multi-frame video input
        # (detected from the processor class, not the model name — see load_model()).
        self.is_video_native: bool = False

    def load_model(self):
        """Loads Qwen2-VL / Qwen2.5-VL model in native bfloat16 / 4-bit onto device for maximum Tensor Core TFLOPS."""
        if self.is_loaded:
            return

        from .video_utils import resolve_model_path
        self.model_name_or_path = resolve_model_path(self.model_name_or_path)

        print(f"[Qwen Vision Model] Loading model weights from '{self.model_name_or_path}' (Native bfloat16 Tensor Cores Active)...", file=sys.stderr)

        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            if self.torch_dtype == "auto":
                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                if not torch.cuda.is_available():
                    dtype = torch.float32
            elif self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

            attn_impl = "sdpa"
            if self.use_sage_attention:
                from .sage_patcher import enable_sage_attention
                enable_sage_attention()
            else:
                try:
                    import flash_attn
                    attn_impl = "flash_attention_2"
                    print("⚡ [Vision Model] FlashAttention-2 active!", file=sys.stderr)
                except ImportError:
                    attn_impl = "sdpa"
                    print("⚡ [Vision Model] PyTorch Native SDPA (Scaled Dot-Product Attention) active on Tensor Cores.", file=sys.stderr)

            kwargs = {
                "torch_dtype": dtype,
                "attn_implementation": attn_impl,
            }

            if self.quantization in ["4bit", "bitsandbytes", "nf4"]:
                from transformers import BitsAndBytesConfig
                print(f"[Qwen Vision Model] Activating bitsandbytes 4-bit NF4 quantization for 7B model...", file=sys.stderr)
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=["multi_modal_projector", "mm_projector"],
                )

            if self.device == "auto":
                if torch.cuda.is_available():
                    kwargs["device_map"] = {"": 0}
                else:
                    kwargs["device_map"] = "auto"
            elif self.device in ["cuda", "cpu"]:
                kwargs["device_map"] = self.device

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            local_weights = None
            target_repo = self.model_name_or_path

            if os.path.isfile(self.model_name_or_path) and self.model_name_or_path.endswith(".safetensors"):
                local_weights = self.model_name_or_path
                target_repo = "Qwen/Qwen2.5-VL-7B-Instruct"
                print(f"[Qwen Vision Model] Detected local safetensors file: '{local_weights}'. Loading config from '{target_repo}'...", file=sys.stderr)

            if local_weights:
                import safetensors.torch
                print(f"[Qwen Vision Model] Instantiating base model architecture from '{target_repo}'...", file=sys.stderr)
                self.model = AutoModelForImageTextToText.from_pretrained(target_repo, **kwargs)
                print(f"[Qwen Vision Model] Loading custom weights from '{local_weights}'...", file=sys.stderr)
                state_dict = safetensors.torch.load_file(local_weights)
                self.model.load_state_dict(state_dict, strict=False)
                del state_dict
                self.processor = AutoProcessor.from_pretrained(target_repo)
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(target_repo, **kwargs)
                self.processor = AutoProcessor.from_pretrained(target_repo)

            if hasattr(self.processor, "tokenizer") and getattr(self.processor.tokenizer, "pad_token", None) is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

            # Detect video-input capability from the processor class itself (not the model name):
            # processors for video-native architectures (Qwen2-VL/2.5-VL/3-VL, VideoLlava,
            # LLaVA-NeXT-Video, InternVL, ...) accept a `video_processor` constructor argument;
            # single-image architectures (plain LLaVA, LLaVA-NeXT, ...) do not.
            self.is_video_native = "video_processor" in inspect.signature(type(self.processor).__init__).parameters

            if not self.is_video_native and hasattr(self.processor, "tokenizer"):
                # Left-padding is required for correct batched multi-frame generation.
                self.processor.tokenizer.padding_side = "left"

            self.is_loaded = True

            target_device = getattr(self.model, "device", self.device)
            quant_tag = " (4-bit NF4 Quantized)" if self.quantization in ["4bit", "bitsandbytes", "nf4"] else " (Native bfloat16 & SDPA Active)"
            print(f"[Qwen Vision Model] Successfully loaded '{self.model_name_or_path}' onto device: {target_device}{quant_tag}", file=sys.stderr)
            print_vram_stats("After Load")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def build_prompt(self, target_prompt: str, duration_sec: float, video_start: Optional[float] = None, video_end: Optional[float] = None) -> str:
        """Constructs a structured prompt asking Qwen2-VL for scene detection timestamps."""
        segment_info = ""
        if video_start is not None and video_end is not None:
            segment_info = f" (Focusing specifically on video segment from {video_start:.1f}s to {video_end:.1f}s)"

        instruction = (
            f"Watch the provided video{segment_info} (total duration: {duration_sec:.1f} seconds). "
            f"Detect any scene or segment matching the following target prompt:\n"
            f"Target Prompt: \"{target_prompt}\"\n\n"
            f"Instructions:\n"
            f"1. Identify the exact start and end timestamps where this target scene occurs in the video.\n"
            f"2. Return your answer as a JSON object with key fields:\n"
            f"   - \"found\": true if scene exists, false otherwise.\n"
            f"   - \"start_time\": timestamp in HH:MM:SS or MM:SS format (e.g., '00:01:15').\n"
            f"   - \"end_time\": timestamp in HH:MM:SS or MM:SS format (e.g., '00:01:30').\n"
            f"   - \"start_seconds\": start time in float seconds.\n"
            f"   - \"end_seconds\": end time in float seconds.\n"
            f"   - \"explanation\": brief objective visual description of the detected scene.\n"
            f"Respond ONLY with the valid JSON block."
        )
        return instruction

    def parse_json_response(self, response_text: str, target_prompt: str) -> Dict[str, Any]:
        """Parses model text response into structured JSON timestamp dict."""
        clean_text = response_text.strip()

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(1)
        else:
            braces_match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
            if braces_match:
                clean_text = braces_match.group(1)

        try:
            parsed = json.loads(clean_text)
            found_bool = parsed.get("found", True)
            start_sec = float(parsed.get("start_seconds", 0.0))
            end_sec = float(parsed.get("end_seconds", 0.0))
            explanation_str = str(parsed.get("explanation", response_text))

            # Strictly validate 'found' boolean logic against model explanation text
            if "does not exist" in explanation_str.lower() or "not found" in explanation_str.lower():
                found_bool = False
            elif start_sec == 0.0 and end_sec == 0.0 and ("not present" in explanation_str.lower() or "no scene" in explanation_str.lower()):
                found_bool = False

            return {
                "success": True,
                "raw_response": response_text,
                "found": found_bool,
                "start_time": parsed.get("start_time", "00:00:00"),
                "end_time": parsed.get("end_time", "00:00:00"),
                "start_seconds": start_sec,
                "end_seconds": end_sec,
                "explanation": explanation_str,
                "prompt": target_prompt,
            }
        except Exception:
            start_sec = 0.0
            end_sec = 0.0

            timestamps = re.findall(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", response_text)
            if len(timestamps) >= 2:
                from .video_utils import VideoUtils
                start_sec = VideoUtils.parse_timestamp(timestamps[0])
                end_sec = VideoUtils.parse_timestamp(timestamps[1])
            elif len(timestamps) == 1:
                from .video_utils import VideoUtils
                start_sec = VideoUtils.parse_timestamp(timestamps[0])
                end_sec = start_sec + 5.0

            found_bool = True if timestamps else False
            if "does not exist" in response_text.lower() or "not found" in response_text.lower():
                found_bool = False

            return {
                "success": True,
                "raw_response": response_text,
                "found": found_bool,
                "start_time": timestamps[0] if timestamps else "00:00:00",
                "end_time": timestamps[1] if len(timestamps) > 1 else (timestamps[0] if timestamps else "00:00:00"),
                "start_seconds": start_sec,
                "end_seconds": end_sec,
                "explanation": response_text,
                "prompt": target_prompt,
            }

    @staticmethod
    def _extract_pil_frames(
        video_path: str,
        video_start: Optional[float],
        video_end: Optional[float],
        num_frames: int,
    ) -> Tuple[List[Any], List[float]]:
        """
        Samples up to `num_frames` frames evenly across [video_start, video_end] for
        single-image (non video-native) models. Returns (pil_images, timestamps_sec)
        in temporal order.
        """
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_f <= 0:
            total_f = 100

        s_f = int((video_start or 0.0) * fps_val)
        e_f = int((video_end if video_end is not None else total_f / fps_val) * fps_val)
        s_f = max(0, min(s_f, total_f - 1))
        e_f = max(s_f + 1, min(e_f, total_f))

        n = max(1, min(num_frames, e_f - s_f))
        frame_indices = torch.linspace(s_f, e_f - 1, n).long().tolist()

        pil_images = []
        timestamps = []
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))
                timestamps.append(f_idx / fps_val)
        cap.release()

        if not pil_images:
            pil_images = [Image.new("RGB", (384, 384))]
            timestamps = [video_start or 0.0]

        return pil_images, timestamps

    def detect_presence_with_reason(
        self,
        video_path: str,
        target_prompt: str,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
        max_frames: int = MAX_TARGET_FRAMES,
        max_pixels: int = DEFAULT_MAX_VIDEO_PIXELS,
    ) -> Tuple[bool, str]:
        """
        Binary scene presence detection returning both boolean verdict (YES/NO)
        and the model's full text reasoning response.
        """
        if not self.is_loaded:
            self.load_model()

        if not self.is_video_native:
            pil_images, timestamps = self._extract_pil_frames(
                video_path, video_start, video_end, IMAGE_ONLY_BATCH_FRAMES
            )

            prompt_text = (
                f"<image>\nWatch this video frame carefully. Question: Does this show the following: "
                f"\"{target_prompt}\"? Answer ONLY YES or NO, followed by a brief reason."
            )
            msgs = [{"role": "user", "content": prompt_text}]
            try:
                chat_prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
            except Exception:
                image_token = getattr(self.processor, "image_token", "<image>")
                chat_prompt = f"USER: {image_token}\n{prompt_text}\nASSISTANT:"

            inputs = self.processor(
                text=[chat_prompt] * len(pil_images),
                images=pil_images,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            t0 = time.time()
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=128, do_sample=False,
                )
            inference_time = time.time() - t0

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            answers = [
                a.strip() for a in self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            ]
            del inputs, generated_ids

            hits = [(ts, ans) for ts, ans in zip(timestamps, answers) if ans.upper().startswith("YES")]
            is_yes = len(hits) > 0
            reported = hits if is_yes else list(zip(timestamps, answers))
            combined_reason = " | ".join(f"[t={ts:.1f}s] {ans}" for ts, ans in reported)

            print(
                f"[Image-Only Model] {len(hits)}/{len(pil_images)} frame(s) matched "
                f"(Inference: {inference_time:.2f}s)",
                file=sys.stderr,
            )
            return is_yes, combined_reason

        from qwen_vl_utils import process_vision_info

        safe_max_pixels = max(min(max_pixels, 350000), 100352)

        prompt = (
            f"Watch this video clip carefully. Focus on the VISUAL ACTIONS and BODY POSITIONS of the people, not just the setting or who is present.\n"
            f"Question: Does this clip visually show the following: \"{target_prompt}\"?\n"
            f"Answer ONLY with the word YES or NO. Followed by a brief description of why it matches or why it does not match."
        )

        messages = []
        if self.uncensored:
            messages.append({"role": "system", "content": UNCENSORED_SYSTEM_PROMPT})

        video_ele = {
            "type": "video",
            "video": video_path,
            "nframes": max_frames,
            "max_pixels": safe_max_pixels,
        }
        if video_start is not None:
            video_ele["video_start"] = video_start
        if video_end is not None:
            video_ele["video_end"] = video_end

        messages.append({"role": "user", "content": [video_ele, {"type": "text", "text": prompt}]})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        t0 = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
            )
        inference_time = time.time() - t0

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        answer = output_text.strip()

        # Clean up GPU tensor references immediately
        del inputs, generated_ids, image_inputs, video_inputs
        print(
            f"[Qwen Vision Model] Response: {answer} (Inference: {inference_time:.2f}s)",
            file=sys.stderr,
        )
        is_yes = answer.upper().startswith("YES")
        return is_yes, answer

    def prepare_describe_inputs(
        self,
        video_path: str,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
        max_frames: int = MAX_TARGET_FRAMES,
        max_pixels: int = DEFAULT_MAX_VIDEO_PIXELS,
        smart_motion_sampling: bool = False,
    ) -> Dict[str, Any]:
        """Pre-extracts and decodes video frames on CPU asynchronously on background threads."""
        if not self.is_loaded:
            self.load_model()

        safe_max_pixels = max(min(max_pixels, 350000), 100352)

        from detector.prompts import get_captioner_prompt
        prompt = get_captioner_prompt()

        if not self.is_video_native:
            pil_images, timestamps = self._extract_pil_frames(
                video_path, video_start, video_end, IMAGE_ONLY_BATCH_FRAMES
            )

            prompt_text = f"<image>\n{prompt}"
            msgs = [{"role": "user", "content": prompt_text}]
            try:
                chat_prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
            except Exception:
                image_token = getattr(self.processor, "image_token", "<image>")
                chat_prompt = f"USER: {image_token}\n{prompt}\nASSISTANT:"

            inputs = self.processor(
                text=[chat_prompt] * len(pil_images),
                images=pil_images,
                padding=True,
                return_tensors="pt",
            )
            return {
                "inputs": inputs,
                "image_inputs": pil_images,
                "video_inputs": None,
                "frame_timestamps": timestamps,
            }

        from qwen_vl_utils import process_vision_info

        messages = []
        if self.uncensored:
            messages.append({"role": "system", "content": UNCENSORED_SYSTEM_PROMPT})

        video_ele = {
            "type": "video",
            "video": video_path,
            "nframes": max_frames,
            "max_pixels": safe_max_pixels,
        }
        if video_start is not None:
            video_ele["video_start"] = video_start
        if video_end is not None:
            video_ele["video_end"] = video_end

        messages.append({"role": "user", "content": [video_ele, {"type": "text", "text": prompt}]})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        return {
            "inputs": inputs,
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
        }

    def describe_scene_from_inputs(self, prepared_data: Dict[str, Any]) -> str:
        """Runs GPU Vision Model inference on pre-extracted video frame tensors.

        For single-image (non video-native) models, `prepared_data` holds a batch of
        independently-encoded frames (batch dim = frame count); each is captioned separately
        and the results are fused into one temporally-ordered description so the matcher's
        one-description-per-window contract is unchanged regardless of model type.
        """
        inputs = prepared_data["inputs"].to(self.model.device)
        frame_timestamps = prepared_data.get("frame_timestamps")

        # Batched multi-frame captions can be shorter individually; video-native single-call
        # descriptions need the full budget.
        max_new_tokens = 200 if frame_timestamps else 600

        t0 = time.time()
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        inference_time = time.time() - t0

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        if frame_timestamps:
            description = "Sequential frame observations:\n" + "\n".join(
                f"[t={ts:.1f}s] {text.strip()}" for ts, text in zip(frame_timestamps, output_texts)
            )
        else:
            description = output_texts[0].strip()

        del inputs, generated_ids
        return description

    def describe_scene(
        self,
        video_path: str,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
        max_frames: int = MAX_TARGET_FRAMES,
        max_pixels: int = DEFAULT_MAX_VIDEO_PIXELS,
        smart_motion_sampling: bool = False,
    ) -> str:
        """Asks Qwen2-VL to describe what is happening in the video clip in detail."""
        prepared_data = self.prepare_describe_inputs(
            video_path=video_path,
            video_start=video_start,
            video_end=video_end,
            max_frames=max_frames,
            max_pixels=max_pixels,
            smart_motion_sampling=smart_motion_sampling,
        )
        return self.describe_scene_from_inputs(prepared_data)


    def detect_presence(
        self,
        video_path: str,
        target_prompt: str,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
        max_frames: int = MAX_TARGET_FRAMES,
        max_pixels: int = DEFAULT_MAX_VIDEO_PIXELS,
    ) -> bool:
        """Fast scene presence detection returning bool."""
        is_yes, _ = self.detect_presence_with_reason(
            video_path, target_prompt,
            video_start=video_start, video_end=video_end,
            max_frames=max_frames, max_pixels=max_pixels,
        )
        return is_yes

    def analyze_frame_at_timestamp(
        self,
        video_path: str,
        target_prompt: str,
        timestamp_sec: float,
        window_duration: float = 2.0,
        max_frames: int = 16,
        max_pixels: int = DEFAULT_MAX_VIDEO_PIXELS,
    ) -> Dict[str, Any]:
        """
        Analyzes a specific video timestamp/frame window against the target prompt for model debugging.
        Returns match status (bool), explanation reason, and raw output text.
        """
        half_win = window_duration / 2.0
        v_start = max(0.0, timestamp_sec - half_win)
        v_end = timestamp_sec + half_win

        is_yes, reason = self.detect_presence_with_reason(
            video_path,
            target_prompt,
            video_start=v_start,
            video_end=v_end,
            max_frames=max_frames,
            max_pixels=max_pixels,
        )

        return {
            "found": is_yes,
            "timestamp": timestamp_sec,
            "prompt": target_prompt,
            "reason": reason,
            "raw_response": reason,
        }


# Global GPU memory model instance cache across API calls & batch runs
_cached_model_instance: Optional[QwenVLModel] = None
_cached_model_key: Optional[Tuple[str, Optional[str], bool, str, str, bool]] = None


def get_cached_qwen_model(
    model_name_or_path: Optional[str] = DEFAULT_MODEL_ID,
    quantization: Optional[str] = "4bit",
    uncensored: bool = True,
    device: str = "auto",
    torch_dtype: str = "auto",
    use_sage_attention: bool = False,
) -> QwenVLModel:
    """
    Returns a cached global QwenVLModel GPU instance across runs.
    Reuses in-memory GPU weights if model_name, quantization, uncensored, device, and torch_dtype settings match.
    Unloads previous model and frees GPU VRAM if settings change.

    If model_name_or_path is not given, uses the first vision model already installed in
    models_cache/hub/ (see detector.video_utils.scan_installed_models). Raises ValueError if
    none is installed yet.
    """
    global _cached_model_instance, _cached_model_key

    if not model_name_or_path:
        from .video_utils import pick_default_installed_model
        model_name_or_path = pick_default_installed_model("vision")
        if not model_name_or_path:
            raise ValueError(
                "No vision model specified and none installed in models_cache/hub/. "
                f"Pass --model / model_name_or_path, or download one first (e.g. '{SUGGESTED_MODEL_ID}')."
            )
        print(f"[Model Cache] No vision model specified — using first installed model: '{model_name_or_path}'", file=sys.stderr)

    current_key = (model_name_or_path, quantization, uncensored, device, torch_dtype, use_sage_attention)

    if _cached_model_instance is not None and _cached_model_key == current_key and _cached_model_instance.is_loaded:
        print(f"[Model Cache] Reusing existing loaded GPU model instance for '{model_name_or_path}' (Quantization: {quantization})", file=sys.stderr)
        return _cached_model_instance

    # Clear previous model instance if settings changed
    if _cached_model_instance is not None:
        print(f"[Model Cache] Settings changed. Unloading previous GPU model instance...", file=sys.stderr)
        try:
            del _cached_model_instance.model
            del _cached_model_instance.processor
        except Exception:
            pass
        _cached_model_instance.is_loaded = False
        _cached_model_instance = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[Model Cache] Instantiating & loading GPU model: '{model_name_or_path}' (Quantization: {quantization})...", file=sys.stderr)
    model_obj = QwenVLModel(
        model_name_or_path=model_name_or_path,
        quantization=quantization,
        uncensored=uncensored,
        device=device,
        torch_dtype=torch_dtype,
        use_sage_attention=use_sage_attention,
    )
    model_obj.load_model()

    _cached_model_instance = model_obj
    _cached_model_key = current_key

    return _cached_model_instance
