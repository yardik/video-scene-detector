"""
Batch Video Processor for Scene Detection.
Scans folders for videos, orchestrates scene detection across multiple files,
cuts clips to an output folder, and reports real-time progress callbacks.
"""

import os
import sys
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Callable, Tuple

from .scene_detector import VideoSceneDetector, SceneMatch
from .qwen_model import IMAGE_ONLY_MICRO_BATCH_SIZE
from .video_utils import VideoUtils

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


@dataclass
class BatchProgress:
    """Real-time batch progress state."""
    status: str                       # 'idle', 'running', 'paused', 'completed', 'error'
    current_video_idx: int = 0
    total_videos: int = 0
    current_video_path: str = ""
    current_video_name: str = ""
    current_phase: str = ""           # 'Phase 1: Coarse', 'Phase 2: Medium', 'Phase 3: Fine'
    current_step_info: str = ""
    percent_complete: float = 0.0
    scenes_found_total: int = 0
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BatchProcessor:
    """
    Orchestrates batch scene detection over a directory of video files.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        quantization: Optional[str] = "4bit",
        dual_model: bool = False,
        text_model_name: Optional[str] = None,
        similarity_threshold: float = 50.0,
        coarse_window_minutes: float = 3.0,
        coarse_frames: int = 16,
        smart_motion_sampling: bool = False,
        use_sage_attention: bool = True,
        image_only_micro_batch_size: int = IMAGE_ONLY_MICRO_BATCH_SIZE,
        text_llm_batch_size: int = 10,
    ):
        self.detector = VideoSceneDetector(
            model_name_or_path=model_name_or_path,
            quantization=quantization,
            dual_model=dual_model,
            text_model_name=text_model_name,
            similarity_threshold=similarity_threshold,
            coarse_window_minutes=coarse_window_minutes,
            coarse_frames=coarse_frames,
            smart_motion_sampling=smart_motion_sampling,
            use_sage_attention=use_sage_attention,
            image_only_micro_batch_size=image_only_micro_batch_size,
            text_llm_batch_size=text_llm_batch_size,
        )
        self._cancel_requested = False

    def cancel(self):
        """Signal the batch runner to stop immediately."""
        self._cancel_requested = True

    @staticmethod
    def find_video_files(input_path: str) -> List[str]:
        """Find all supported video files in input_path (single video file or folder)."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        # Single video file mode
        if os.path.isfile(input_path):
            ext = os.path.splitext(input_path)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                return [os.path.abspath(input_path)]
            else:
                raise ValueError(f"Unsupported video file extension: {ext}")

        # Folder batch scanning mode
        video_files = []
        for root, _, files in os.walk(input_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    video_files.append(os.path.join(root, file))

        video_files.sort()
        return video_files

    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        prompt: str,
        progress_callback: Optional[Callable[[BatchProgress, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process all videos in input_dir matching prompt, outputting cut clips to output_dir.
        Emits progress_callback(progress_state, log_line) for live UI streaming.
        """
        self._cancel_requested = False
        os.makedirs(output_dir, exist_ok=True)

        video_files = self.find_video_files(input_dir)
        total_videos = len(video_files)

        if total_videos == 0:
            msg = f"No supported video files found in: {input_dir}"
            if progress_callback:
                progress_callback(
                    BatchProgress(
                        status="completed",
                        total_videos=0,
                        error_message=msg,
                    ),
                    msg,
                )
            return []

        start_time = time.time()
        all_results = []
        total_scenes_found = 0

        log_fn = lambda msg: progress_callback and progress_callback(progress, msg)

        progress = BatchProgress(
            status="running",
            total_videos=total_videos,
            current_video_idx=0,
            percent_complete=0.0,
        )
        log_fn(f"🚀 Starting batch scan across {total_videos} video file(s) in: {input_dir}")

        for idx, video_path in enumerate(video_files, 1):
            if self._cancel_requested:
                progress.status = "stopped"
                log_fn("🛑 Batch scan stopped by user.")
                break

            video_name = os.path.basename(video_path)
            progress.current_video_idx = idx
            progress.current_video_path = video_path
            progress.current_video_name = video_name
            progress.percent_complete = round(((idx - 1) / total_videos) * 100, 1)
            progress.elapsed_seconds = round(time.time() - start_time, 1)

            log_fn(f"\n{'='*70}")
            log_fn(f"  [{idx}/{total_videos}] Processing Video: {video_name}")
            log_fn(f"{'='*70}")

            try:
                # Custom callback for scene_detector inside this video
                def video_progress_hook(phase_name: str, step_msg: str):
                    progress.current_phase = phase_name
                    progress.current_step_info = step_msg
                    progress.elapsed_seconds = round(time.time() - start_time, 1)
                    if progress_callback:
                        progress_callback(progress, f"  [{video_name}] {phase_name} {step_msg}")

                scenes = self.detector.detect_scenes(
                    video_path=video_path,
                    prompt=prompt,
                    cut_clips=True,
                    clip_output_dir=output_dir,
                    is_cancelled=lambda: self._cancel_requested,
                )

                total_scenes_found += len(scenes)
                progress.scenes_found_total = total_scenes_found

                video_result = {
                    "video_path": video_path,
                    "video_name": video_name,
                    "scenes_found": len(scenes),
                    "scenes": [s.to_dict() for s in scenes],
                }
                all_results.append(video_result)

                if scenes:
                    log_fn(f"✅ Found {len(scenes)} scene(s) in {video_name}:")
                    for s in scenes:
                        log_fn(f"   Scene {s.scene_index}: {s.start_time} -> {s.end_time} ({s.end_seconds - s.start_seconds:.1f}s)")
                        if s.clip_path:
                            log_fn(f"   Clip: {s.clip_path}")
                else:
                    log_fn(f"ℹ️ No matching scenes found in {video_name}")

            except InterruptedError:
                progress.status = "stopped"
                log_fn("🛑 Batch scan stopped by user.")
                self.detector.model.unload()
                if self.detector.text_llm:
                    self.detector.text_llm.unload()
                from .qwen_model import clear_video_capture_cache
                clear_video_capture_cache()
                break

            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")
                log_fn(f"❌ Error processing {video_name}: {e}")
                all_results.append({
                    "video_path": video_path,
                    "video_name": video_name,
                    "error": str(e),
                    "scenes_found": 0,
                    "scenes": [],
                })

        elapsed = time.time() - start_time
        progress.status = "completed" if not self._cancel_requested else "stopped"
        progress.percent_complete = 100.0
        progress.elapsed_seconds = round(elapsed, 1)

        # Export consolidated batch JSON report
        report_file = os.path.join(output_dir, "batch_detection_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({
                "input_dir": input_dir,
                "output_dir": output_dir,
                "prompt": prompt,
                "total_videos_scanned": total_videos,
                "total_scenes_found": total_scenes_found,
                "elapsed_seconds": round(elapsed, 1),
                "results": all_results,
            }, f, indent=2)

        log_fn(f"\n{'='*70}")
        log_fn(f"🎉 Batch Scan Complete!")
        log_fn(f"   Total Videos:  {total_videos}")
        log_fn(f"   Scenes Found:  {total_scenes_found}")
        log_fn(f"   Total Time:    {elapsed:.1f}s")
        log_fn(f"   Report Saved:  {report_file}")
        log_fn(f"{'='*70}\n")

        return all_results
