import os
import re
import subprocess
from typing import Dict, Any, List, Optional
import cv2

# Standard HuggingFace hub cache layout: models_cache/hub/models--{org}--{name}
MODELS_CACHE_HUB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models_cache", "hub"))

# Heuristics for classifying an installed model folder as vision vs. text
_VISION_MODEL_KEYWORDS = {"vl", "joycaption", "llava", "caption"}
_TEXT_EXCLUDE_KEYWORDS = {"bge", "e5", "gte"}  # embedding/utility models — neither vision nor a chat LLM


def _hub_cache_snapshot_dir(entry_dir: str) -> Optional[str]:
    """Resolves a `models--org--name` HF hub cache entry to its snapshot directory (the one `refs/main` points at, or any snapshot present)."""
    snapshots_dir = os.path.join(entry_dir, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    ref_main = os.path.join(entry_dir, "refs", "main")
    if os.path.exists(ref_main):
        try:
            with open(ref_main, "r", encoding="utf-8") as f:
                commit_hash = f.read().strip()
            candidate = os.path.join(snapshots_dir, commit_hash)
            if os.path.isdir(candidate):
                return candidate
        except OSError:
            pass

    subdirs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
    return os.path.join(snapshots_dir, subdirs[0]) if subdirs else None


def scan_installed_models() -> Dict[str, List[Dict[str, str]]]:
    """
    Scans both models_cache/hub/ and models/ for locally installed HuggingFace models
    and classifies each as vision or text by name heuristics.
    """
    vision_models = []
    text_models = []
    seen_ids = set()

    def _process_directory(folder_path: str, is_hub_cache: bool):
        if not os.path.isdir(folder_path):
            return
        for entry in sorted(os.listdir(folder_path)):
            entry_dir = os.path.join(folder_path, entry)
            if not os.path.isdir(entry_dir):
                continue

            if is_hub_cache:
                if not entry.startswith("models--"):
                    continue
                parts = entry.split("--", 2)
                if len(parts) < 3:
                    continue
                model_id = f"{parts[1]}/{parts[2]}"
                name_lower = parts[2].lower()
                target_dir = _hub_cache_snapshot_dir(entry_dir)
            else:
                if "--" not in entry:
                    continue
                parts = entry.split("--", 1)
                model_id = f"{parts[0]}/{parts[1]}"
                name_lower = parts[1].lower()
                target_dir = entry_dir

            if not target_dir or model_id in seen_ids:
                continue

            if any(kw in name_lower for kw in _TEXT_EXCLUDE_KEYWORDS):
                continue

            if not _local_model_weights_present(target_dir):
                continue

            seen_ids.add(model_id)
            entry_dict = {"id": model_id, "label": model_id}
            if any(kw in name_lower for kw in _VISION_MODEL_KEYWORDS):
                vision_models.append(entry_dict)
            else:
                text_models.append(entry_dict)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")

    _process_directory(MODELS_CACHE_HUB_DIR, is_hub_cache=True)
    _process_directory(models_dir, is_hub_cache=False)

    vision_models.sort(key=lambda x: x["id"])
    text_models.sort(key=lambda x: x["id"])

    return {"vision_models": vision_models, "text_models": text_models}


def pick_default_installed_model(model_type: str) -> Optional[str]:
    """
    Returns the first locally installed model id for `model_type` ("vision" or "text"),
    or None if nothing of that type is cached in models_cache/hub/ yet.
    """
    key = "vision_models" if model_type == "vision" else "text_models"
    entries = scan_installed_models()[key]
    return entries[0]["id"] if entries else None


def _local_model_weights_present(local_target: str) -> bool:
    """
    Verifies the actual weight files referenced by the model are present on disk —
    config.json/tokenizer files alone can exist from an interrupted snapshot_download
    while the (much larger, slower-to-transfer) safetensors/bin shards never landed.
    """
    import json

    for index_name, shard_key in (
        ("model.safetensors.index.json", "weight_map"),
        ("pytorch_model.bin.index.json", "weight_map"),
    ):
        index_path = os.path.join(local_target, index_name)
        if os.path.exists(index_path):
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    weight_map = json.load(f).get(shard_key, {})
            except (OSError, json.JSONDecodeError):
                return False
            shard_files = set(weight_map.values())
            if not shard_files:
                return False
            return all(os.path.exists(os.path.join(local_target, fn)) for fn in shard_files)

    for single_file in ("model.safetensors", "pytorch_model.bin"):
        if os.path.exists(os.path.join(local_target, single_file)):
            return True

    return False


def resolve_model_path(model_id_or_path: str) -> str:
    """
    Checks if model_id_or_path exists as a local folder.
    If not found at exact path, checks models/<sanitized_name> inside project directory.
    If model exists locally with all its weight files, loads directly from disk.
    If model is missing or incompletely downloaded, (re)downloads the snapshot from
    Hugging Face to models/<sanitized_name> — snapshot_download resumes from
    whatever partial/interrupted blobs are already cached there.
    """
    import sys
    if os.path.exists(model_id_or_path) and os.path.isdir(model_id_or_path):
        return model_id_or_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    sanitized = model_id_or_path.replace("/", "--").replace("\\", "--")
    local_target = os.path.join(models_dir, sanitized)

    config_file = os.path.join(local_target, "config.json")
    if os.path.exists(local_target) and os.path.exists(config_file) and _local_model_weights_present(local_target):
        print(f"📁 Loading model directly from local directory: {local_target}", file=sys.stderr)
        return local_target

    print(f"📥 Local model incomplete or not found. Downloading '{model_id_or_path}' from HuggingFace to {local_target}...", file=sys.stderr)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model_id_or_path,
            local_dir=local_target,
            local_dir_use_symlinks=False,
            token=_resolve_hf_token(),
        )
        print(f"✅ Model '{model_id_or_path}' downloaded and saved to local directory: {local_target}", file=sys.stderr)
        return local_target
    except Exception as e:
        print(f"⚠️ Warning: Could not download to local models directory: {_gated_repo_hint(model_id_or_path, e)}. Falling back to default loader.", file=sys.stderr)
        return model_id_or_path


def _resolve_hf_token() -> Optional[str]:
    """
    Explicit HF token lookup (env vars, then huggingface_hub's cached CLI login) so
    gated-model downloads don't silently depend on this process having inherited an
    env var set in some other shell/session after it was already started.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _gated_repo_hint(model_id: str, error: Exception) -> str:
    """Appends actionable guidance to gated/auth errors from the Hub."""
    msg = str(error)
    if "gated" in msg.lower() or "restricted" in msg.lower() or "authenticated" in msg.lower():
        token_state = "found" if _resolve_hf_token() else "NOT found"
        return (
            f"{msg} | This model requires accepting its license on the HuggingFace model "
            f"page (https://huggingface.co/{model_id}) with the account that owns your token, "
            f"then a valid access token (HF_TOKEN env var {token_state} in this process, "
            f"or run `huggingface-cli login`). If you just set HF_TOKEN, restart the web "
            f"server/CLI process — it only sees env vars present when it was launched."
        )
    return msg


class VideoUtils:
    """Utility class for video inspection, frame extraction, timestamp conversions, and clip cutting."""

    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """
        Extract video metadata including duration, fps, frame count, width, height.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        cap.release()

        return {
            "path": video_path,
            "duration": round(duration, 2),
            "duration_str": VideoUtils.format_timestamp(duration),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "resolution": f"{width}x{height}",
            "width": width,
            "height": height,
        }

    @staticmethod
    def filter_motion_keyframes(pil_images: list, min_frames: int = 4, difference_threshold: float = 3.0) -> list:
        """
        Calculates frame-to-frame pixel differences across candidate keyframes.
        Filters out redundant static frames where motion/visual change is minimal (< threshold),
        preserving only distinct visual keyframes.
        Always retains at least min_frames (default 4) and the first + last frames.
        """
        if len(pil_images) <= min_frames:
            return pil_images

        import numpy as np

        selected = [pil_images[0]]
        prev_arr = np.array(pil_images[0].convert("L"), dtype=np.float32)

        for i in range(1, len(pil_images) - 1):
            curr_arr = np.array(pil_images[i].convert("L"), dtype=np.float32)
            mean_diff = np.mean(np.abs(curr_arr - prev_arr))

            # Retain frame if mean pixel difference exceeds threshold
            if mean_diff >= difference_threshold:
                selected.append(pil_images[i])
                prev_arr = curr_arr

        # Always include the last frame
        selected.append(pil_images[-1])

        # Guarantee at least min_frames are returned
        if len(selected) < min_frames:
            indices = np.linspace(0, len(pil_images) - 1, min_frames, dtype=int)
            return [pil_images[idx] for idx in indices]

        return selected

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """
        Convert float seconds to HH:MM:SS string.
        """
        if seconds < 0:
            seconds = 0
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        
        if hrs > 0:
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"
        else:
            return f"{mins:02d}:{secs:02d}"

    @staticmethod
    def parse_timestamp(timestamp_str: str) -> float:
        """
        Parse time string (HH:MM:SS, MM:SS, SS, or float) into seconds.
        """
        if isinstance(timestamp_str, (int, float)):
            return float(timestamp_str)

        timestamp_str = str(timestamp_str).strip()
        
        # Match HH:MM:SS or MM:SS or SS.ms
        parts = timestamp_str.split(":")
        try:
            if len(parts) == 3:
                hrs, mins, secs = float(parts[0]), float(parts[1]), float(parts[2])
                return hrs * 3600 + mins * 60 + secs
            elif len(parts) == 2:
                mins, secs = float(parts[0]), float(parts[1])
                return mins * 60 + secs
            elif len(parts) == 1:
                return float(parts[0])
        except ValueError:
            pass
        return 0.0

    @staticmethod
    def cut_video_clip(
        video_path: str,
        output_path: str,
        start_sec: float,
        end_sec: float
    ) -> str:
        """
        Cut a clip from video between start_sec and end_sec.
        Uses bundled FFmpeg executable via imageio-ffmpeg for instant lossless extraction (0.05s).
        """
        if start_sec >= end_sec:
            raise ValueError(f"start_sec ({start_sec}) must be less than end_sec ({end_sec})")

        # Try imageio_ffmpeg bundled binary first for sub-second stream copy
        try:
            import imageio_ffmpeg
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_bin,
                "-y",
                "-ss", str(start_sec),
                "-to", str(end_sec),
                "-i", video_path,
                "-c", "copy",
                "-avoid_negative_ts", "1",
                output_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception:
            pass

        # Try system FFmpeg if available
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start_sec),
                "-to", str(end_sec),
                "-i", video_path,
                "-c", "copy",
                "-avoid_negative_ts", "1",
                output_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        except Exception:
            pass

        # OpenCV Fallback
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        while cap.isOpened() and current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()
        return output_path

    @staticmethod
    def extract_frame_at(video_path: str, timestamp_sec: float, output_image_path: str) -> Optional[str]:
        """Extract a single frame image at the specified timestamp in seconds."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        target_frame = int(timestamp_sec * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(output_image_path, frame)
            return output_image_path
        return None
