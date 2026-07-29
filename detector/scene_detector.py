import sys
import os
import json
import time
import gc
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging

import torch

from .video_utils import VideoUtils
from .qwen_model import (
    DEFAULT_MODEL_ID,
    DEFAULT_MAX_VIDEO_PIXELS,
    MAX_TARGET_FRAMES,
    IMAGE_ONLY_MICRO_BATCH_SIZE,
    get_cached_qwen_model,
)
from .text_llm import TextLLMClassifier, DEFAULT_TEXT_MODEL_ID, get_cached_text_llm

logger = logging.getLogger(__name__)


@dataclass
class SceneMatch:
    video_path: str
    prompt: str
    found: bool
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    explanation: str
    clip_path: Optional[str] = None
    scene_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionTracker:
    def __init__(self):
        self.log_lines: List[str] = []
        self.pipeline_start: float = time.time()
        self.vision_inference_sec: float = 0.0
        self.llm_inference_sec: float = 0.0
        self.active_frame_extract_sec: float = 0.0

    def log(self, msg: str = ""):
        if not str(msg).strip():
            self.log_lines.append("")
            print("", file=sys.stderr)
            return
        now = time.time()
        elapsed = now - self.pipeline_start
        timestamp = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"
        msg_str = f"[{timestamp} | +{elapsed:5.1f}s] {msg}"
        self.log_lines.append(msg_str)
        print(msg_str, file=sys.stderr)


class VideoSceneDetector:
    """
    Hybrid Coarse-to-Fine scene detector using 3-phase binary detection:

    Phase 1 (Coarse): 3-min sliding windows scan the whole video for candidate regions.
    Phase 2 (Medium): 30-sec sub-windows pinpoint which candidate regions actually match.
    Phase 3 (Fine):   Binary search within boundary sub-windows.
                      Finds exact start/end timestamps to ~4 second precision.

    Finds ALL matching scenes in the video.
    Optimized for NVIDIA RTX 4080 SUPER (14GB VRAM budget).
    """

    # ── Phase 1: Coarse Scan Parameters (3-Minute Windows with 64 High-Density Frames) ──
    COARSE_WINDOW_SEC = 180.0     # 3-minute windows
    COARSE_STRIDE_SEC = 150.0     # 30s overlap between adjacent windows
    COARSE_FRAMES = 64            # 64 frames across 3 minutes (1 frame sampled every 2.81 seconds)
    COARSE_MAX_PIXELS = 100_352   # ~316x316 — sharp & ultra-fast

    # ── Phase 2: Medium Scan Parameters (Pass 2) ──
    MEDIUM_WINDOW_SEC = 30.0      # 30-second sub-windows
    MEDIUM_FRAMES = 16            # 16 frames across 30s (~0.8s per sub-window)
    MEDIUM_MAX_PIXELS = 100_352   # ~316x316 — sharp facial detail

    FINE_PRECISION_SEC = 4.0      # Binary search stops at 4s precision
    FINE_FRAMES = 12              # 12 frames for binary search steps
    FINE_MAX_PIXELS = 80_000      # ~282x282
    FINE_MIN_SEGMENT_SEC = 3.0    # Minimum segment for model to analyze

    def __init__(
        self,
        model_name_or_path: Optional[str] = DEFAULT_MODEL_ID,
        device: str = "auto",
        torch_dtype: str = "auto",
        uncensored: bool = True,
        quantization: Optional[str] = None,
        dual_model: bool = True,
        text_model_name: Optional[str] = DEFAULT_TEXT_MODEL_ID,
        similarity_threshold: float = 50.0,
        coarse_window_minutes: float = 2.0,
        coarse_frames: int = 16,
        smart_motion_sampling: bool = False,
        use_sage_attention: bool = True,
        image_only_micro_batch_size: int = IMAGE_ONLY_MICRO_BATCH_SIZE,
        text_llm_batch_size: int = 10,
    ):
        self.dual_model = dual_model
        self.text_model_name = text_model_name
        self.similarity_threshold = similarity_threshold
        self.smart_motion_sampling = smart_motion_sampling
        self.use_sage_attention = use_sage_attention
        self.text_llm_batch_size = max(1, text_llm_batch_size)

        # Bounded Coarse Scan Parameters (Min 1.0m, Max 5.0m; Min 8 frames, Max 64 frames)
        bounded_min = max(1.0, min(5.0, coarse_window_minutes))
        bounded_frames = max(8, min(64, coarse_frames))

        self.COARSE_WINDOW_SEC = bounded_min * 60.0
        self.COARSE_STRIDE_SEC = max(15.0, self.COARSE_WINDOW_SEC - 30.0)
        self.COARSE_FRAMES = bounded_frames

        self.model = get_cached_qwen_model(
            model_name_or_path=model_name_or_path,
            device=device,
            torch_dtype=torch_dtype,
            uncensored=uncensored,
            quantization=quantization,
            use_sage_attention=use_sage_attention,
            image_only_micro_batch_size=image_only_micro_batch_size,
        )

        self.text_llm = None
        if self.dual_model:
            self.text_llm = get_cached_text_llm(
                model_name_or_path=text_model_name,
                quantization=quantization,
                use_sage_attention=use_sage_attention,
            )

    # ══════════════════════════════════════════════════════════════════════
    # Condition Verification Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _eval_single_condition(
        self,
        video_path: str,
        condition: str,
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
        max_frames: int = MAX_TARGET_FRAMES,
        max_pixels: int = DEFAULT_MAX_VIDEO_PIXELS,
        tracker: Optional[ExecutionTracker] = None,
    ) -> Tuple[bool, str]:
        """Evaluates a single condition using either single Qwen2-VL or Dual-Model pipeline."""
        if self.dual_model and self.text_llm:
            if self.text_llm.is_loaded:
                self.text_llm.unload()

            t0 = time.time()
            description = self.model.describe_scene(
                video_path,
                video_start=video_start,
                video_end=video_end,
                max_frames=max_frames,
                max_pixels=max_pixels,
                smart_motion_sampling=self.smart_motion_sampling,
            )
            dt_vision = time.time() - t0
            if tracker:
                tracker.vision_inference_sec += dt_vision

            self.model.unload()

            t0 = time.time()
            is_match, reason = self.text_llm.classify_description(
                description, condition, threshold=self.similarity_threshold
            )
            dt_llm = time.time() - t0
            if tracker:
                tracker.llm_inference_sec += dt_llm

            full_reason = f"Qwen Description: {description} || Text LLM Reasoning: {reason}"
            return is_match, full_reason
        else:
            t0 = time.time()
            res = self.model.detect_presence_with_reason(
                video_path,
                condition,
                video_start=video_start,
                video_end=video_end,
                max_frames=max_frames,
                max_pixels=max_pixels,
            )
            dt_vision = time.time() - t0
            if tracker:
                tracker.vision_inference_sec += dt_vision
            return res

    def _eval_conditions(
        self,
        video_path: str,
        conditions: List[str],
        video_start: Optional[float] = None,
        video_end: Optional[float] = None,
        max_frames: int = MAX_TARGET_FRAMES,
        max_pixels: int = DEFAULT_MAX_VIDEO_PIXELS,
        tracker: Optional[ExecutionTracker] = None,
    ) -> Tuple[bool, str]:
        def log_msg(msg: str):
            if tracker:
                tracker.log(msg)
            else:
                print(msg, file=sys.stderr)

        cond_logs = []
        reasons = []
        for i, cond in enumerate(conditions, 1):
            detected, reason = self._eval_single_condition(
                video_path, cond,
                video_start=video_start, video_end=video_end,
                max_frames=max_frames,
                max_pixels=max_pixels,
                tracker=tracker,
            )
            tag = "YES" if detected else "NO"
            if len(conditions) > 1:
                cond_logs.append(f"C{i}:{tag}")
            else:
                cond_logs.append(tag)

            clean_reason = reason.replace("\n", " ")
            reasons.append(f"C{i}: {clean_reason}")

            if not detected:
                log_str = " | ".join(cond_logs)
                log_msg(f"      [Reason C{i}]: {clean_reason}")
                return False, f"NO ({log_str}) | Reason: {clean_reason}" if len(conditions) > 1 else f"NO | Reason: {clean_reason}"

        log_str = " | ".join(cond_logs)
        full_reason_str = " || ".join(reasons)
        log_msg(f"      [Match Reasons]: {full_reason_str}")
        return True, f"YES ✓ [{log_str}] | Reason: {full_reason_str}" if len(conditions) > 1 else f"YES ✓ | Reason: {full_reason_str}"



    def _save_summary_log(
        self,
        video_path: str,
        tracker: ExecutionTracker,
        total_time: float,
        all_scenes: List[SceneMatch],
        clip_output_dir: Optional[str] = None,
    ) -> str:
        def _fmt(s: float) -> str:
            if s >= 60.0:
                mins = int(s // 60)
                secs = s % 60
                return f"{mins}m {secs:.1f}s ({s:.1f}s)"
            return f"{s:.1f}s"

        total_gpu = tracker.vision_inference_sec + tracker.llm_inference_sec
        extracted_clips = [os.path.basename(s.clip_path) for s in all_scenes if s.clip_path]
        clip_names_str = ", ".join(extracted_clips) if extracted_clips else "None"

        summary_header = (
            f"Filename: {os.path.basename(video_path)}\n"
            f"Total Time: {_fmt(total_time)}\n"
            f"Total GPU Time: {_fmt(total_gpu)}\n"
            f"Active Frame Extraction Time: {_fmt(tracker.active_frame_extract_sec)}\n"
            f"Vision Inference Time: {_fmt(tracker.vision_inference_sec)}\n"
            f"LLM Inference Time: {_fmt(tracker.llm_inference_sec)}\n"
            f"Clips Extracted: {len(extracted_clips)}\n"
            f"Clip Names: {clip_names_str}\n\n"
            f"{'='*70}\n"
            f"DETAILED INFERENCE & EXTRACTION LOG\n"
            f"{'='*70}\n"
        )

        full_content = summary_header + "\n".join(tracker.log_lines) + "\n"

        target_dir = clip_output_dir or os.path.dirname(video_path) or "."
        os.makedirs(target_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        summary_path = os.path.join(target_dir, f"{base_name}_summary.log")

        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(full_content)
            tracker.log(f"📄 Saved per-file execution summary log to: {summary_path}")
        except Exception as e:
            logger.warning(f"Could not write summary log file: {e}")

        return summary_path

    @staticmethod
    def _release_cuda_cache(tracker: Optional[ExecutionTracker] = None):
        """
        Returns unused cached VRAM to the driver once a video finishes processing.
        Model weights stay resident (get_cached_qwen_model/get_cached_text_llm keep them
        loaded for fast reuse on the next video) — this only clears the allocator's
        cached-but-unused blocks, which otherwise linger at whatever the run's peak
        batch size was and show up as "still using memory" in nvidia-smi/Task Manager.
        """
        gc.collect()
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        msg = f"🧹 Released cached VRAM (Active: {allocated_mb:.0f} MB | Reserved: {reserved_mb:.0f} MB)"
        if tracker:
            tracker.log(msg)
        else:
            print(msg, file=sys.stderr)

    # ══════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════

    def detect_scenes(
        self,
        video_path: str,
        prompt: str,
        cut_clips: bool = False,
        clip_output_dir: Optional[str] = None,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[SceneMatch]:
        """
        Finds ALL scenes in the video matching the prompt.
        Returns a list of SceneMatch objects with precise boundaries.

        Guarantees cached-but-unused VRAM is released (see _release_cuda_cache) whether
        this completes normally or raises partway through — a crash mid-run (bad video file,
        an OOM, any bug) is exactly when leftover reserved memory would otherwise compound
        across subsequent runs in the same process.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        try:
            return self._detect_scenes_impl(
                video_path, prompt,
                cut_clips=cut_clips,
                clip_output_dir=clip_output_dir,
                start_time=start_time,
                end_time=end_time,
                is_cancelled=is_cancelled,
            )
        finally:
            self._release_cuda_cache()
            from .qwen_model import clear_video_capture_cache
            clear_video_capture_cache()

    def _detect_scenes_impl(
        self,
        video_path: str,
        prompt: str,
        cut_clips: bool = False,
        clip_output_dir: Optional[str] = None,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[SceneMatch]:
        tracker = ExecutionTracker()
        log = tracker.log

        conditions = [prompt]

        video_info = VideoUtils.get_video_info(video_path)
        duration_sec = video_info["duration"]
        pipeline_start = time.time()

        start_sec = VideoUtils.parse_timestamp(start_time) if start_time is not None else 0.0
        end_sec = VideoUtils.parse_timestamp(end_time) if end_time is not None else duration_sec

        log(f"{'='*70}")
        log(f"  Coarse-to-Fine Scene Detection Pipeline")
        log(f"  Video:        {os.path.basename(video_path)}")
        log(f"  Duration:     {duration_sec:.1f}s ({VideoUtils.format_timestamp(duration_sec)})")
        if start_time is not None or end_time is not None:
            log(f"  Bounded Range:{VideoUtils.format_timestamp(start_sec)} → {VideoUtils.format_timestamp(end_sec)} ({end_sec - start_sec:.1f}s)")
        log(f"  Prompt:       \"{prompt}\"")
        log(f"{'='*70}\n")

        # ── Phase 1: Coarse Scan (entire video or bounded range) ──
        p1_start = time.time()
        coarse_results = self._coarse_scan(
            video_path, conditions, duration_sec, tracker=tracker,
            start_sec=start_sec, end_sec=end_sec, is_cancelled=is_cancelled,
        )
        coarse_regions = self._merge_hits(coarse_results)
        p1_time = time.time() - p1_start

        log(f"\n{'='*70}")
        log(f"📊 PHASE 1 STATS SUMMARY (Coarse Recall Scan)")
        log(f"   • Time Elapsed:        {p1_time:.1f}s")
        log(f"   • Windows Evaluated:   {len(coarse_results)} (3-min windows @ 30s stride)")
        log(f"   • Candidate Regions:   {len(coarse_regions)} region(s) matched Primary Action")
        if coarse_regions:
            log(f"   • Candidate Timestamps:")
            for i, (rs, re) in enumerate(coarse_regions, 1):
                log(f"       Region {i}: {VideoUtils.format_timestamp(rs)} → {VideoUtils.format_timestamp(re)} ({re - rs:.0f}s)")
        log(f"{'='*70}\n")

        # Phase 1 can run large batches; release before Phase 2 starts so its peak
        # doesn't compound on top of Phase 1's leftover reserved memory.
        self._release_cuda_cache(tracker)

        if not coarse_regions:
            elapsed = time.time() - pipeline_start
            log(f"\n❌ No matching scenes found satisfying ALL conditions. (Total: {elapsed:.1f}s)")
            self._save_summary_log(video_path, tracker, elapsed, [], clip_output_dir=clip_output_dir)
            return []

        # ── Phase 2: Global Medium Scan ──
        p2_start = time.time()
        all_region_medium_results = self._medium_scan_all_regions(video_path, conditions, coarse_regions, tracker=tracker, is_cancelled=is_cancelled)
        p2_time = time.time() - p2_start

        all_medium_spans: List[Tuple[float, float]] = []
        region_spans_map: List[List[Tuple[float, float]]] = []

        for region_idx, (region_start, region_end) in enumerate(coarse_regions):
            medium_results = all_region_medium_results[region_idx]
            scene_spans = self._merge_hits(medium_results)
            region_spans_map.append(scene_spans)
            if scene_spans:
                all_medium_spans.extend(scene_spans)

        log(f"\n{'='*70}")
        log(f"📊 PHASE 2 STATS SUMMARY (Sub-Window Scan)")
        log(f"   • Time Elapsed:        {p2_time:.1f}s")
        log(f"   • Confirmed Spans:     {len(all_medium_spans)} sub-window span(s) matched")
        if all_medium_spans:
            log(f"   • Verified Timestamps:")
            for i, (ss, se) in enumerate(all_medium_spans, 1):
                log(f"       Span {i}: {VideoUtils.format_timestamp(ss)} → {VideoUtils.format_timestamp(se)} ({se - ss:.0f}s)")
        log(f"{'='*70}\n")

        # Release before Phase 3's many small binary-search steps begin.
        self._release_cuda_cache(tracker)

        if not all_medium_spans:
            elapsed = time.time() - pipeline_start
            log(f"\n❌ No matching sub-windows passed Phase 2 verification. (Total: {elapsed:.1f}s)")
            self._save_summary_log(video_path, tracker, elapsed, [], clip_output_dir=clip_output_dir)
            return []

        # ── Phase 3: Fine Boundary Search (Binary Search) ──
        p3_start = time.time()
        raw_scene_intervals: List[Tuple[float, float]] = []

        for region_idx, scene_spans in enumerate(region_spans_map):
            if not scene_spans:
                continue
            r_start, r_end = coarse_regions[region_idx]
            log(f"\n{'─'*60}")
            log(f"  Region {region_idx+1}/{len(coarse_regions)} Boundary Refinement: {VideoUtils.format_timestamp(r_start)} → {VideoUtils.format_timestamp(r_end)}")

            for span_start, span_end in scene_spans:
                log(f"\n  🎯 Confirmed span: {VideoUtils.format_timestamp(span_start)} → {VideoUtils.format_timestamp(span_end)} ({span_end - span_start:.0f}s)")

                # ── Phase 3: Binary search for exact boundaries ──
                exact_start = self._find_start_boundary(
                    video_path, conditions, span_start, span_end, duration_sec, tracker=tracker, is_cancelled=is_cancelled,
                )
                exact_end = self._find_end_boundary(
                    video_path, conditions, span_start, span_end, duration_sec, tracker=tracker, is_cancelled=is_cancelled,
                )

                if exact_end <= exact_start:
                    exact_end = min(duration_sec, exact_start + 10.0)

                raw_scene_intervals.append((exact_start, exact_end))

        p3_time = time.time() - p3_start

        log(f"\n{'='*70}")
        log(f"📊 PHASE 3 STATS SUMMARY (Fine Boundary Refinement)")
        log(f"   • Time Elapsed:        {p3_time:.1f}s")
        log(f"   • Refined Boundaries:  {len(raw_scene_intervals)} scene boundary interval(s)")
        log(f"{'='*70}\n")

        # ── Smart Scene Merging: Merge non-consecutive scenes separated by <= 10.0s ──
        merged_scene_intervals = self._merge_adjacent_scenes(raw_scene_intervals, max_gap_sec=10.0, tracker=tracker)

        all_scenes: List[SceneMatch] = []

        for scene_idx, (exact_start, exact_end) in enumerate(merged_scene_intervals, 1):
            duration = exact_end - exact_start
            log(f"\n  ✅ Scene {scene_idx}: {VideoUtils.format_timestamp(exact_start)} → {VideoUtils.format_timestamp(exact_end)} ({duration:.1f}s)")

            # Cut clip for consolidated consecutive scene
            clip_path = None
            if cut_clips:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                clip_filename = f"{base_name}_scene{scene_idx}_{int(exact_start)}s-{int(exact_end)}s.mp4"
                if clip_output_dir:
                    os.makedirs(clip_output_dir, exist_ok=True)
                    clip_filename = os.path.join(clip_output_dir, clip_filename)
                try:
                    clip_path = VideoUtils.cut_video_clip(
                        video_path, clip_filename, exact_start, exact_end
                    )
                    log(f"  📎 Clip saved: {clip_path}")
                except Exception as e:
                    logger.warning(f"Could not cut clip: {e}")

            all_scenes.append(SceneMatch(
                video_path=video_path,
                prompt=prompt,
                found=True,
                start_time=VideoUtils.format_timestamp(exact_start),
                end_time=VideoUtils.format_timestamp(exact_end),
                start_seconds=round(exact_start, 2),
                end_seconds=round(exact_end, 2),
                explanation=f"Matched \"{prompt}\" ({duration:.0f}s duration).",
                clip_path=clip_path,
                scene_index=scene_idx,
            ))

        elapsed = time.time() - pipeline_start
        log(f"\n{'='*70}")
        log(f"📊 PHASE 3 STATS SUMMARY (Fine Search & Assembly Complete)")
        log(f"   • Total Execution Time: {elapsed:.1f}s")
        log(f"   • Final Scenes Exported:{len(all_scenes)} scene(s)")
        log(f"   • Final Scene Timestamps:")
        for s in all_scenes:
            dur = s.end_seconds - s.start_seconds
            log(f"       Scene {s.scene_index}: {s.start_time} → {s.end_time} ({dur:.1f}s) | Clip: {s.clip_path or 'N/A'}")
        log(f"{'='*70}\n")

        self._save_summary_log(video_path, tracker, elapsed, all_scenes, clip_output_dir=clip_output_dir)

        return all_scenes

    def detect_scene(self, video_path: str, prompt: str, **kwargs) -> SceneMatch:
        """Backward-compatible single-scene detection. Returns first match or not-found."""
        cut_clips = kwargs.pop("cut_clip", False)
        clip_output_dir = kwargs.pop("clip_output_dir", None)
        kwargs.pop("fps", None)
        kwargs.pop("max_pixels", None)
        kwargs.pop("max_frames", None)
        kwargs.pop("two_pass_refinement", None)
        kwargs.pop("padding_sec", None)
        kwargs.pop("clip_output_path", None)
        kwargs.pop("chunk_duration_sec", None)

        scenes = self.detect_scenes(
            video_path, prompt,
            cut_clips=cut_clips,
            clip_output_dir=clip_output_dir,
        )
        if scenes:
            return scenes[0]
        return SceneMatch(
            video_path=video_path,
            prompt=prompt,
            found=False,
            start_time="00:00",
            end_time="00:00",
            start_seconds=0.0,
            end_seconds=0.0,
            explanation=f"The target scene '{prompt}' was not found in the video.",
        )

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Coarse Scan (Primary Action Recall)
    # ══════════════════════════════════════════════════════════════════════

    def _coarse_scan(
        self,
        video_path: str,
        conditions: List[str],
        duration_sec: float,
        tracker: Optional[ExecutionTracker] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[Tuple[float, float, bool]]:
        """
        Scans video with 3-min sliding windows using primary action condition (C1).
        This maximizes recall so scene setups (e.g. kneeling at 16:00) aren't filtered out early.
        """
        def log_msg(msg: str):
            if tracker:
                tracker.log(msg)
            else:
                print(msg, file=sys.stderr)

        primary_condition = conditions[0]
        scan_start = max(0.0, start_sec)
        scan_end = min(duration_sec, end_sec) if end_sec is not None else duration_sec

        windows = self._build_windows(
            scan_start, scan_end, self.COARSE_WINDOW_SEC, self.COARSE_STRIDE_SEC
        )

        log_msg(
            f"[Phase 1: Coarse Scan] {len(windows)} windows × {self.COARSE_WINDOW_SEC:.0f}s "
            f"(stride {self.COARSE_STRIDE_SEC:.0f}s, {self.COARSE_FRAMES} frames)\n"
            f"  Primary Action Prompt: \"{primary_condition}\""
        )

        if self.dual_model and self.text_llm and len(windows) > 1:
            descriptions = []
            log_msg(f"  🎥 Step 1/2: Extracting scene descriptions across {len(windows)} coarse windows (Async CPU Prefetching Active)...")

            # Free the text LLM's VRAM while only the vision model is needed; it lazily
            # reloads inside classify_descriptions_batch() below.
            self.text_llm.unload()
            if not self.model.is_loaded:
                self.model.load_model()

            from concurrent.futures import ThreadPoolExecutor

            total_gpu_stall_sec = 0.0
            initial_setup_sec = 0.0

            def _prep_window(w):
                t_ext0 = time.time()
                data = self.model.prepare_describe_inputs(
                    video_path,
                    video_start=w[0],
                    video_end=w[1],
                    max_frames=self.COARSE_FRAMES,
                    max_pixels=self.COARSE_MAX_PIXELS,
                    smart_motion_sampling=self.smart_motion_sampling,
                )
                data["_extraction_duration"] = time.time() - t_ext0
                return data

            from collections import deque
            PREFETCH_DEPTH = 2
            with ThreadPoolExecutor(max_workers=PREFETCH_DEPTH + 1) as executor:
                pending = deque(
                    executor.submit(_prep_window, w)
                    for w in windows[:PREFETCH_DEPTH]
                )
                next_idx = PREFETCH_DEPTH

                for idx, (w_start, w_end) in enumerate(windows, 1):
                    if is_cancelled and is_cancelled():
                        raise InterruptedError("Operation cancelled by user")
                    t_pop0 = time.time()
                    prepared_data = pending.popleft().result()
                    pop_wait_sec = time.time() - t_pop0
                    extract_dur = prepared_data.get("_extraction_duration", 0.0)

                    if idx == 1:
                        initial_setup_sec = pop_wait_sec
                    else:
                        total_gpu_stall_sec += pop_wait_sec

                    if next_idx < len(windows):
                        pending.append(executor.submit(_prep_window, windows[next_idx]))
                        next_idx += 1

                    t_gpu0 = time.time()
                    desc = self.model.describe_scene_from_inputs(prepared_data)
                    dt_gpu = time.time() - t_gpu0
                    descriptions.append(desc)
                    if tracker:
                        tracker.vision_inference_sec += dt_gpu

                    if "input_chunks" in prepared_data and prepared_data["input_chunks"] is not None:
                        prepared_data["input_chunks"].clear()
                    if "image_inputs" in prepared_data and prepared_data["image_inputs"] is not None:
                        prepared_data["image_inputs"].clear()
                    del prepared_data

                    stall_str = f"GPU Stall: {total_gpu_stall_sec:.2f}s" if total_gpu_stall_sec > 0.05 else "GPU Idle: 0.00s (100% Masked)"
                    log_msg(f"    [Vision Desc {idx}/{len(windows)}] [{VideoUtils.format_timestamp(w_start)}..{VideoUtils.format_timestamp(w_end)}] (GPU: {dt_gpu:.1f}s | {stall_str}):\n{desc}")

            if tracker:
                tracker.active_frame_extract_sec += (initial_setup_sec + total_gpu_stall_sec)

            log_msg(f"  ⏱️ Frame Extraction Timing Metrics: Initial Setup: {initial_setup_sec:.2f}s | GPU Idle Stall: {total_gpu_stall_sec:.2f}s (All extractions running asynchronously on CPU)")

            # Unload video vision model entirely before doing text LLM classification
            self.model.unload()

            log_msg(f"\n  🧠 Step 2/2: Batch evaluating Text LLM similarity ({len(windows)} descriptions in parallel)...")
            t_batch0 = time.time()
            classified_results = self.text_llm.classify_descriptions_batch(
                descriptions,
                [primary_condition] * len(windows),
                threshold=self.similarity_threshold,
                batch_size=self.text_llm_batch_size,
            )
            dt_batch = time.time() - t_batch0
            if tracker:
                tracker.llm_inference_sec += dt_batch

            self.text_llm.unload()

            log_msg(f"  ⚡ Parallel Text LLM batch classification complete in {dt_batch:.2f}s ({dt_batch/len(windows):.2f}s avg/window)!\n")

            results = []
            for (w_start, w_end), (detected, reason) in zip(windows, classified_results):
                tag = "YES ✓" if detected else "NO"
                clean_reason = reason.replace("\n", " ")
                log_msg(
                    f"  Window [{VideoUtils.format_timestamp(w_start)}..{VideoUtils.format_timestamp(w_end)}] "
                    f"→ {tag}\n"
                    f"    [Reason]: {clean_reason}"
                )
                results.append((w_start, w_end, detected))

            return results

        results = []
        for idx, (w_start, w_end) in enumerate(windows):
            t0 = time.time()
            detected, reason = self._eval_single_condition(
                video_path, primary_condition,
                video_start=w_start, video_end=w_end,
                max_frames=self.COARSE_FRAMES,
                max_pixels=self.COARSE_MAX_PIXELS,
                tracker=tracker,
            )
            dt = time.time() - t0
            tag = "YES ✓" if detected else "NO"
            clean_reason = reason.replace("\n", " ")
            log_msg(
                f"  Window {idx+1:2d}/{len(windows)} "
                f"[{VideoUtils.format_timestamp(w_start)}..{VideoUtils.format_timestamp(w_end)}] "
                f"→ {tag} ({dt:.1f}s)\n"
                f"    [Model Reason]: {clean_reason}"
            )
            results.append((w_start, w_end, detected))

        return results

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Medium Scan (Global Cross-Region Batching)
    # ══════════════════════════════════════════════════════════════════════

    def _medium_scan_all_regions(
        self,
        video_path: str,
        conditions: List[str],
        coarse_regions: List[Tuple[float, float]],
        tracker: Optional[ExecutionTracker] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[List[Tuple[float, float, bool]]]:
        """Subdivides all coarse hit regions into 30-sec sub-windows evaluating sub-conditions in a single global batch."""
        def log_msg(msg: str):
            if tracker:
                tracker.log(msg)
            else:
                print(msg, file=sys.stderr)

        region_windows_map: List[List[Tuple[float, float]]] = []
        all_windows: List[Tuple[int, float, float]] = []

        for r_idx, (r_start, r_end) in enumerate(coarse_regions):
            r_wins = self._build_windows(r_start, r_end, self.MEDIUM_WINDOW_SEC, self.MEDIUM_WINDOW_SEC / 2.0)
            region_windows_map.append(r_wins)
            for (w_s, w_e) in r_wins:
                all_windows.append((r_idx, w_s, w_e))

        log_msg(f"  [Phase 2: Global Medium Scan] {len(all_windows)} sub-windows across {len(coarse_regions)} region(s) × {self.MEDIUM_WINDOW_SEC:.0f}s ({self.MEDIUM_FRAMES} frames)")

        if self.dual_model and self.text_llm and len(all_windows) > 1:
            descriptions = []
            log_msg(f"  🎥 Extracting {len(all_windows)} sub-window descriptions (Async CPU Prefetching Active)...")

            # Free the text LLM's VRAM while only the vision model is needed; it lazily
            # reloads inside classify_descriptions_batch() below.
            self.text_llm.unload()
            if not self.model.is_loaded:
                self.model.load_model()

            from collections import deque
            from concurrent.futures import ThreadPoolExecutor

            def _prep_sub_window(w):
                return self.model.prepare_describe_inputs(
                    video_path,
                    video_start=w[1],
                    video_end=w[2],
                    max_frames=self.MEDIUM_FRAMES,
                    max_pixels=self.MEDIUM_MAX_PIXELS,
                    smart_motion_sampling=self.smart_motion_sampling,
                )

            # Keep PREFETCH_DEPTH sub-windows' frames being extracted on CPU threads
            # ahead of the one currently running GPU inference, so a slow extraction
            # (e.g. a window needing a big seek) has more than one window's worth of
            # GPU compute time to hide behind before it can stall the pipeline.
            PREFETCH_DEPTH = 2
            with ThreadPoolExecutor(max_workers=PREFETCH_DEPTH + 1) as executor:
                pending = deque(
                    executor.submit(_prep_sub_window, w)
                    for w in all_windows[:PREFETCH_DEPTH]
                )
                next_idx = PREFETCH_DEPTH
                for idx, item in enumerate(all_windows, 1):
                    if is_cancelled and is_cancelled():
                        raise InterruptedError("Operation cancelled by user")
                    prepared_data = pending.popleft().result()
                    if next_idx < len(all_windows):
                        pending.append(executor.submit(_prep_sub_window, all_windows[next_idx]))
                        next_idx += 1

                    t_gpu0 = time.time()
                    desc = self.model.describe_scene_from_inputs(prepared_data)
                    dt_gpu = time.time() - t_gpu0
                    descriptions.append(desc)
                    if tracker:
                        tracker.vision_inference_sec += dt_gpu
                    log_msg(f"    [Vision Desc {idx}/{len(all_windows)}] [{VideoUtils.format_timestamp(all_windows[idx-1][1])}..{VideoUtils.format_timestamp(all_windows[idx-1][2])}] (GPU: {dt_gpu:.1f}s):\n{desc}")

                    if "input_chunks" in prepared_data and prepared_data["input_chunks"] is not None:
                        prepared_data["input_chunks"].clear()
                    if "image_inputs" in prepared_data and prepared_data["image_inputs"] is not None:
                        prepared_data["image_inputs"].clear()
                    del prepared_data

            # Unload video vision model entirely before doing text LLM classification
            self.model.unload()

            log_msg(f"  🧠 Batch evaluating Text LLM similarity ({len(all_windows)} sub-windows)...")
            t_llm0 = time.time()
            classified = self.text_llm.classify_descriptions_batch(
                descriptions,
                [conditions[0]] * len(all_windows),
                threshold=self.similarity_threshold,
                batch_size=self.text_llm_batch_size,
            )
            dt_llm = time.time() - t_llm0
            if tracker:
                tracker.llm_inference_sec += dt_llm

            self.text_llm.unload()

            region_results: List[List[Tuple[float, float, bool]]] = [[] for _ in coarse_regions]
            for (r_idx, w_s, w_e), (detected, reason) in zip(all_windows, classified):
                clean_reason = reason.replace("\n", " ")
                tag = "YES ✓" if detected else "NO"
                log_msg(f"    Region {r_idx+1} Sub [{VideoUtils.format_timestamp(w_s)}..{VideoUtils.format_timestamp(w_e)}] → {tag} | Reason: {clean_reason}")
                region_results[r_idx].append((w_s, w_e, detected))

            return region_results

        # Fallback single-region scan
        region_results = []
        for r_idx, (r_start, r_end) in enumerate(coarse_regions):
            r_res = self._medium_scan(video_path, conditions, r_start, r_end, tracker=tracker)
            region_results.append(r_res)
        return region_results

    def _medium_scan(
        self, video_path: str, conditions: List[str], region_start: float, region_end: float, tracker: Optional[ExecutionTracker] = None
    ) -> List[Tuple[float, float, bool]]:
        """Subdivides a coarse hit region into 30-sec sub-windows evaluating sub-conditions."""
        def log_msg(msg: str):
            if tracker:
                tracker.log(msg)
            else:
                print(msg, file=sys.stderr)

        windows = self._build_windows(
            region_start, region_end, self.MEDIUM_WINDOW_SEC, self.MEDIUM_WINDOW_SEC
        )

        log_msg(
            f"  [Phase 2: Medium Scan] {len(windows)} sub-windows × {self.MEDIUM_WINDOW_SEC:.0f}s "
            f"({self.MEDIUM_FRAMES} frames)"
        )

        if self.dual_model and self.text_llm and len(windows) > 1:
            descriptions = []
            if not self.model.is_loaded:
                self.model.load_model()

            from collections import deque
            from concurrent.futures import ThreadPoolExecutor

            def _prep_medium_window(w):
                return self.model.prepare_describe_inputs(
                    video_path,
                    video_start=w[0],
                    video_end=w[1],
                    max_frames=self.MEDIUM_FRAMES,
                    max_pixels=self.MEDIUM_MAX_PIXELS,
                    smart_motion_sampling=self.smart_motion_sampling,
                )

            PREFETCH_DEPTH = 2
            with ThreadPoolExecutor(max_workers=PREFETCH_DEPTH + 1) as executor:
                pending = deque(
                    executor.submit(_prep_medium_window, w)
                    for w in windows[:PREFETCH_DEPTH]
                )
                next_idx = PREFETCH_DEPTH
                for idx, (w_start, w_end) in enumerate(windows, 1):
                    prepared_data = pending.popleft().result()
                    if next_idx < len(windows):
                        pending.append(executor.submit(_prep_medium_window, windows[next_idx]))
                        next_idx += 1

                    t0 = time.time()
                    desc = self.model.describe_scene_from_inputs(prepared_data)
                    dt_gpu = time.time() - t0
                    descriptions.append(desc)
                    if tracker:
                        tracker.vision_inference_sec += dt_gpu

                    if "input_chunks" in prepared_data and prepared_data["input_chunks"] is not None:
                        prepared_data["input_chunks"].clear()
                    if "image_inputs" in prepared_data and prepared_data["image_inputs"] is not None:
                        prepared_data["image_inputs"].clear()
                    del prepared_data

            # Unload video vision model entirely before doing text LLM classification
            self.model.unload()

            t0 = time.time()
            classified = self.text_llm.classify_descriptions_batch(
                descriptions,
                [conditions[0]] * len(windows),
                threshold=self.similarity_threshold,
                batch_size=self.text_llm_batch_size,
            )
            if tracker:
                tracker.llm_inference_sec += (time.time() - t0)

            self.text_llm.unload()

            results = []
            for (w_start, w_end), (detected, reason) in zip(windows, classified):
                clean_reason = reason.replace("\n", " ")
                tag = "YES ✓" if detected else "NO"
                log_msg(
                    f"    Sub [{VideoUtils.format_timestamp(w_start)}..{VideoUtils.format_timestamp(w_end)}] "
                    f"→ {tag} | Reason: {clean_reason}"
                )
                results.append((w_start, w_end, detected))

            return results

        results = []
        for idx, (w_start, w_end) in enumerate(windows):
            t0 = time.time()
            detected, tag = self._eval_conditions(
                video_path, conditions,
                video_start=w_start, video_end=w_end,
                max_frames=self.MEDIUM_FRAMES,
                max_pixels=self.MEDIUM_MAX_PIXELS,
                tracker=tracker,
            )
            dt = time.time() - t0
            log_msg(
                f"    Sub {idx+1:2d}/{len(windows)} "
                f"[{VideoUtils.format_timestamp(w_start)}..{VideoUtils.format_timestamp(w_end)}] "
                f"→ {tag}  ({dt:.1f}s)"
            )
            results.append((w_start, w_end, detected))

        return results

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Fine Boundary Search (Binary Search)
    # ══════════════════════════════════════════════════════════════════════

    def _find_start_boundary(
        self,
        video_path: str,
        conditions: List[str],
        span_start: float,
        span_end: float,
        duration_sec: float,
        tracker: Optional[ExecutionTracker] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> float:
        """Binary search for exact scene start within the first YES sub-window."""
        def log_msg(msg: str):
            if tracker:
                tracker.log(msg)
            else:
                print(msg, file=sys.stderr)

        search_end = min(span_start + self.MEDIUM_WINDOW_SEC, span_end)
        lo, hi = span_start, search_end

        log_msg(
            f"  [Phase 3: Start Boundary] "
            f"Searching [{VideoUtils.format_timestamp(lo)}..{VideoUtils.format_timestamp(hi)}]"
        )

        step = 0
        while hi - lo > self.FINE_PRECISION_SEC:
            if is_cancelled and is_cancelled():
                raise InterruptedError("Operation cancelled by user")
            mid = (lo + hi) / 2.0
            if mid - lo < self.FINE_MIN_SEGMENT_SEC:
                break

            t0 = time.time()
            detected, tag = self._eval_conditions(
                video_path, conditions,
                video_start=lo, video_end=mid,
                max_frames=self.FINE_FRAMES,
                max_pixels=self.FINE_MAX_PIXELS,
                tracker=tracker,
            )
            dt = time.time() - t0
            step += 1

            if detected:
                log_msg(
                    f"    Step {step}: [{VideoUtils.format_timestamp(lo)}..{VideoUtils.format_timestamp(mid)}] "
                    f"→ YES  (start ≤ {VideoUtils.format_timestamp(mid)})  ({dt:.1f}s)"
                )
                hi = mid  # Scene starts in first half
            else:
                log_msg(
                    f"    Step {step}: [{VideoUtils.format_timestamp(lo)}..{VideoUtils.format_timestamp(mid)}] "
                    f"→ NO   (start > {VideoUtils.format_timestamp(mid)})  ({dt:.1f}s)"
                )
                lo = mid  # Scene starts in second half

        log_msg(f"    → Start: {VideoUtils.format_timestamp(lo)}")
        return lo

    def _find_end_boundary(
        self,
        video_path: str,
        conditions: List[str],
        span_start: float,
        span_end: float,
        duration_sec: float,
        tracker: Optional[ExecutionTracker] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> float:
        """Binary search for exact scene end within the last YES sub-window."""
        def log_msg(msg: str):
            if tracker:
                tracker.log(msg)
            else:
                print(msg, file=sys.stderr)

        search_start = max(span_end - self.MEDIUM_WINDOW_SEC, span_start)
        lo, hi = search_start, span_end

        log_msg(
            f"  [Phase 3: End Boundary] "
            f"Searching [{VideoUtils.format_timestamp(lo)}..{VideoUtils.format_timestamp(hi)}]"
        )

        step = 0
        while hi - lo > self.FINE_PRECISION_SEC:
            if is_cancelled and is_cancelled():
                raise InterruptedError("Operation cancelled by user")
            mid = (lo + hi) / 2.0
            if hi - mid < self.FINE_MIN_SEGMENT_SEC:
                break

            t0 = time.time()
            detected, tag = self._eval_conditions(
                video_path, conditions,
                video_start=mid, video_end=hi,
                max_frames=self.FINE_FRAMES,
                max_pixels=self.FINE_MAX_PIXELS,
                tracker=tracker,
            )
            dt = time.time() - t0
            step += 1

            if detected:
                log_msg(
                    f"    Step {step}: [{VideoUtils.format_timestamp(mid)}..{VideoUtils.format_timestamp(hi)}] "
                    f"→ YES  (end ≥ {VideoUtils.format_timestamp(mid)})  ({dt:.1f}s)"
                )
                lo = mid  # Scene continues past midpoint
            else:
                log_msg(
                    f"    Step {step}: [{VideoUtils.format_timestamp(mid)}..{VideoUtils.format_timestamp(hi)}] "
                    f"→ NO   (end < {VideoUtils.format_timestamp(mid)})  ({dt:.1f}s)"
                )
                hi = mid  # Scene ends before midpoint

        log_msg(f"    → End: {VideoUtils.format_timestamp(hi)}")
        return hi

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_windows(
        start: float, end: float, window_size: float, stride: float
    ) -> List[Tuple[float, float]]:
        """Build temporal windows with optional overlap."""
        windows = []
        curr = start
        while curr < end:
            w_end = min(end, curr + window_size)
            windows.append((curr, w_end))
            if w_end >= end:
                break
            curr += stride
        return windows

    @staticmethod
    def _merge_hits(
        results: List[Tuple[float, float, bool]]
    ) -> List[Tuple[float, float]]:
        """Merge consecutive YES results into contiguous regions."""
        regions = []
        current_start = None
        current_end = None
        for start, end, detected in results:
            if detected:
                if current_start is None:
                    current_start = start
                current_end = end
            else:
                if current_start is not None:
                    regions.append((current_start, current_end))
                    current_start = None
                    current_end = None
        if current_start is not None:
            regions.append((current_start, current_end))
        return regions

    @staticmethod
    def _merge_adjacent_scenes(
        raw_intervals: List[Tuple[float, float]],
        max_gap_sec: float = 10.0,
        tracker: Optional[ExecutionTracker] = None,
    ) -> List[Tuple[float, float]]:
        """
        Merge scene intervals separated by gaps <= max_gap_sec (default 10s).
        If Scene A ends at 10:30 and Scene B starts at 10:38 (gap = 8s <= 10s),
        they are assembled into a single long consecutive scene from 10:00 to 11:10.
        """
        def log_msg(msg: str):
            if tracker:
                tracker.log(msg)
            else:
                print(msg, file=sys.stderr)

        if not raw_intervals:
            return []

        sorted_intervals = sorted(raw_intervals, key=lambda x: x[0])
        merged: List[Tuple[float, float]] = []

        curr_start, curr_end = sorted_intervals[0]

        for next_start, next_end in sorted_intervals[1:]:
            gap = next_start - curr_end
            if gap <= max_gap_sec:
                log_msg(
                    f"  🔗 Merging adjacent scenes (gap: {gap:.1f}s ≤ {max_gap_sec}s): "
                    f"[{VideoUtils.format_timestamp(curr_start)}..{VideoUtils.format_timestamp(curr_end)}] + "
                    f"[{VideoUtils.format_timestamp(next_start)}..{VideoUtils.format_timestamp(next_end)}]"
                )
                curr_end = max(curr_end, next_end)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = next_start, next_end

        merged.append((curr_start, curr_end))
        return merged

    @staticmethod
    def export_json(matches: List[SceneMatch], output_filepath: str):
        """Export results to JSON format."""
        data = [m.to_dict() for m in matches]
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_srt(matches: List[SceneMatch], output_filepath: str):
        """Export results to SRT subtitle format."""
        lines = []
        for i, m in enumerate(matches, 1):
            lines.append(f"{i}")
            lines.append(
                f"{m.start_time.replace('.', ',')} --> {m.end_time.replace('.', ',')}"
            )
            lines.append(f"[{m.prompt}] {m.explanation}")
            lines.append("")
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
