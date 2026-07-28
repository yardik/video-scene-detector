"""
FastAPI Server for Video Scene Detector Web Client.
Provides REST APIs for folder batch processing, prompt analysis, real-time SSE progress streaming,
and serving cut video clips directly to the HTML5 video player.
"""

import os
import sys
import json
import time
import asyncio
import threading
import logging
from typing import Optional

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from detector import (
    VideoSceneDetector,
    BatchProcessor,
    BatchProgress,
    PromptAnalyzer,
    VideoUtils,
    QwenVLModel,
    get_cached_qwen_model,
    TextLLMClassifier,
    get_cached_text_llm,
)
from detector.video_utils import scan_installed_models, MODELS_CACHE_HUB_DIR

logger = logging.getLogger(__name__)

app = FastAPI(title="Video Scene Detector", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
current_processor: Optional[BatchProcessor] = None
batch_thread: Optional[threading.Thread] = None
event_queue: asyncio.Queue = asyncio.Queue()
loop: Optional[asyncio.AbstractEventLoop] = None
current_output_dir: str = os.path.abspath(os.getcwd())
is_shutting_down: bool = False

class PromptRequest(BaseModel):
    prompt: str


class BatchStartRequest(BaseModel):
    input_dir: str
    output_dir: str
    prompt: str
    model_name: Optional[str] = None
    quantization: str = "4bit"
    dual_model: bool = True
    text_model_name: Optional[str] = None
    similarity_threshold: float = 50.0
    coarse_window_minutes: float = 2.0
    coarse_frames: int = 64
    smart_motion_sampling: bool = False
    use_sage_attention: bool = False


class PathCheckRequest(BaseModel):
    path: str


class DebugFrameRequest(BaseModel):
    video_path: str
    timestamp: float
    prompt: str
    model_name: Optional[str] = None
    quantization: str = "4bit"
    dual_model: bool = True
    text_model_name: Optional[str] = None
    similarity_threshold: float = 50.0
    smart_motion_sampling: bool = False
    use_sage_attention: bool = False


class PromptsUpdateRequest(BaseModel):
    captioner_prompt: Optional[str] = None
    matcher_prompt: Optional[str] = None
    reset_defaults: bool = False


class ModelDownloadRequest(BaseModel):
    model_id: str  # e.g. "Qwen/Qwen2.5-7B-Instruct"
    model_type: str = "text"  # "vision" or "text" — informational only


@app.get("/api/models")
async def get_installed_models():
    """Return lists of locally installed vision and text models."""
    return scan_installed_models()


@app.post("/api/models/download")
async def download_model(req: ModelDownloadRequest):
    """Download a HuggingFace model to models_cache/hub/ using huggingface_hub."""
    model_id = req.model_id.strip()
    if not model_id or "/" not in model_id:
        raise HTTPException(status_code=400, detail="Invalid model ID. Expected format: org/model-name")

    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Starting download of model: {model_id}")
        snapshot_download(
            repo_id=model_id,
            cache_dir=MODELS_CACHE_HUB_DIR,
            local_dir=None,
        )
        logger.info(f"Successfully downloaded model: {model_id}")
        # Return updated model lists
        return {
            "status": "success",
            "message": f"Model '{model_id}' downloaded successfully.",
            "models": scan_installed_models(),
        }
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


@app.get("/api/prompts")
async def get_system_prompts():
    """Retrieve current captioner and matcher system prompts."""
    from detector.prompts import get_captioner_prompt, get_matcher_prompt
    return {
        "captioner_prompt": get_captioner_prompt(),
        "matcher_prompt": get_matcher_prompt(),
    }


@app.post("/api/prompts")
async def update_system_prompts(req: PromptsUpdateRequest):
    """Update and persist captioner and/or matcher system prompts to disk."""
    from detector.prompts import (
        save_captioner_prompt, save_matcher_prompt,
        get_captioner_prompt, get_matcher_prompt,
        DEFAULT_CAPTIONER_PROMPT, DEFAULT_MATCHER_PROMPT
    )
    if req.reset_defaults:
        save_captioner_prompt(DEFAULT_CAPTIONER_PROMPT)
        save_matcher_prompt(DEFAULT_MATCHER_PROMPT)
        return {
            "status": "success",
            "message": "Reset all prompts to default templates.",
            "captioner_prompt": DEFAULT_CAPTIONER_PROMPT,
            "matcher_prompt": DEFAULT_MATCHER_PROMPT,
        }

    if req.captioner_prompt is not None:
        save_captioner_prompt(req.captioner_prompt)
    if req.matcher_prompt is not None:
        save_matcher_prompt(req.matcher_prompt)

    return {
        "status": "success",
        "message": "System prompts updated and saved to prompts/ directory.",
        "captioner_prompt": get_captioner_prompt(),
        "matcher_prompt": get_matcher_prompt(),
    }


@app.on_event("startup")
async def startup_event():
    global loop, is_shutting_down
    loop = asyncio.get_running_loop()
    is_shutting_down = False


@app.on_event("shutdown")
def shutdown_event():
    global is_shutting_down, current_processor
    is_shutting_down = True
    if current_processor:
        try:
            current_processor.cancel()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass



@app.get("/api/stream-video")
async def stream_video(path: str):
    """Stream any workstation video file for HTML5 video player playback."""
    clean_path = os.path.abspath(path.strip().strip('"').strip("'"))
    if not os.path.exists(clean_path) or not os.path.isfile(clean_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {clean_path}")
    return FileResponse(clean_path, media_type="video/mp4")


@app.post("/api/debug-frame")
async def debug_frame(req: DebugFrameRequest):
    """Analyze a single video frame/timecode against prompt and return full model reasoning."""
    clean_path = os.path.abspath(req.video_path.strip().strip('"').strip("'"))
    if not os.path.exists(clean_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {clean_path}")

    quant_setting = None if req.quantization == "none" else req.quantization

    try:
        model = get_cached_qwen_model(
            model_name_or_path=req.model_name,
            quantization=quant_setting,
        )

        t_total_start = time.time()
        if req.dual_model:
            v_start = max(0.0, req.timestamp - 1.0)
            v_end = req.timestamp + 1.0
            
            t_v0 = time.time()
            description = model.describe_scene(
                video_path=clean_path,
                video_start=v_start,
                video_end=v_end,
                max_frames=16,
            )
            t_vision_dur = round(time.time() - t_v0, 2)

            text_llm = get_cached_text_llm(
                model_name_or_path=req.text_model_name,
                quantization=quant_setting,
            )
            
            t_t0 = time.time()
            is_match, reason = text_llm.classify_description(
                description, req.prompt, threshold=req.similarity_threshold
            )
            t_text_dur = round(time.time() - t_t0, 2)
            t_total_dur = round(time.time() - t_total_start, 2)

            return {
                "found": is_match,
                "timestamp": req.timestamp,
                "prompt": req.prompt,
                "description": description,
                "reason": reason,
                "dual_model": True,
                "vision_time_sec": t_vision_dur,
                "text_time_sec": t_text_dur,
                "total_time_sec": t_total_dur,
                "raw_response": f"Qwen Description: {description}\n\nText LLM Reasoning: {reason}",
            }
        else:
            t_v0 = time.time()
            result = model.analyze_frame_at_timestamp(
                video_path=clean_path,
                target_prompt=req.prompt,
                timestamp_sec=req.timestamp,
            )
            t_total_dur = round(time.time() - t_total_start, 2)
            result["dual_model"] = False
            result["vision_time_sec"] = t_total_dur
            result["text_time_sec"] = 0.0
            result["total_time_sec"] = t_total_dur
            return result
    except Exception as e:
        logger.error(f"Debug frame error: {e}")
        raise HTTPException(status_code=500, detail=str(e))




# ══════════════════════════════════════════════════════════════════════
# REST API Endpoints
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/analyze-prompt")
async def analyze_prompt(req: PromptRequest):
    """Analyze prompt quality and return improvement suggestions."""
    analyzer = PromptAnalyzer()
    analysis = analyzer.analyze(req.prompt)
    return {
        "prompt": analysis.original_prompt,
        "score": round(analysis.quality_score, 2),
        "issues": analysis.issues,
        "suggestions": analysis.suggestions,
    }


@app.post("/api/check-path")
async def check_path(req: PathCheckRequest):
    """Validate input path (single video file or folder) and return list of videos."""
    path = os.path.abspath(req.path.strip())
    exists = os.path.exists(path)
    is_file = os.path.isfile(path) if exists else False
    is_dir = os.path.isdir(path) if exists else False

    videos = []
    if exists:
        try:
            videos = BatchProcessor.find_video_files(path)
            videos = [os.path.basename(v) for v in videos]
        except Exception:
            pass

    return {
        "path": path,
        "exists": exists,
        "is_file": is_file,
        "is_dir": is_dir,
        "video_count": len(videos),
        "video_names": videos[:10],  # preview first 10
    }


@app.post("/api/start-batch")
async def start_batch(req: BatchStartRequest):
    """Start scene detection across a single video file or an input directory."""
    global current_processor, batch_thread, current_output_dir

    if current_processor is not None and batch_thread is not None and batch_thread.is_alive():
        raise HTTPException(status_code=400, detail="A detection job is already running.")

    input_dir = os.path.abspath(req.input_dir.strip())
    raw_out = req.output_dir.strip() if req.output_dir else ""
    if not raw_out:
        output_dir = os.path.abspath(os.path.join(os.getcwd(), "extracted_clips"))
    else:
        output_dir = os.path.abspath(raw_out)
    current_output_dir = output_dir

    if not os.path.exists(input_dir):
        raise HTTPException(status_code=404, detail=f"Input path does not exist: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    quant_setting = None if req.quantization == "none" else req.quantization

    current_processor = BatchProcessor(
        model_name_or_path=req.model_name,
        quantization=quant_setting,
        dual_model=req.dual_model,
        text_model_name=req.text_model_name,
        similarity_threshold=req.similarity_threshold,
        coarse_window_minutes=req.coarse_window_minutes,
        coarse_frames=req.coarse_frames,
        smart_motion_sampling=req.smart_motion_sampling,
        use_sage_attention=req.use_sage_attention,
    )

    def progress_callback(progress: BatchProgress, log_line: str):
        vram_stats = {}
        if torch.cuda.is_available():
            vram_stats = {
                "active_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 1),
            }

        payload = {
            "progress": progress.to_dict(),
            "log": log_line,
            "vram": vram_stats,
        }
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(event_queue.put(payload), loop)

    def run_job():
        try:
            current_processor.process_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                prompt=req.prompt,
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.error(f"Batch processing error: {e}")

    batch_thread = threading.Thread(target=run_job, daemon=True)
    batch_thread.start()

    return {"status": "started", "input_dir": input_dir, "output_dir": output_dir}


@app.post("/api/stop-batch")
async def stop_batch():
    """Cancel running batch job immediately."""
    global current_processor
    if current_processor:
        current_processor.cancel()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        payload = {
            "progress": {
                "status": "stopped",
                "total_videos": 0,
                "current_video_idx": 0,
                "current_video_name": "Batch Stopped",
                "current_phase": "Stopped",
                "percent_complete": 0.0,
                "scenes_found_total": 0,
                "elapsed_seconds": 0.0,
            },
            "log": "🛑 Batch processing stopped by user.",
            "vram": {},
        }
        await event_queue.put(payload)
        return {"status": "stopping"}
    return {"status": "not_running"}


@app.get("/api/progress-stream")
async def progress_stream(request: Request):
    """Server-Sent Events endpoint streaming live logs & progress updates."""
    async def event_generator():
        while not is_shutting_down:
            if await request.is_disconnected():
                break
            try:
                data = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                yield f"data: {json.dumps(data)}\n\n"
            except asyncio.TimeoutError:
                # Keep-alive heartbeat
                yield ": heartbeat\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/clips")
async def list_clips(dir_path: Optional[str] = None):
    """List generated clip MP4 files in the output directory."""
    target_dir = os.path.abspath(dir_path.strip()) if dir_path else current_output_dir

    if not os.path.exists(target_dir):
        return {"clips": [], "output_dir": target_dir}

    clips = []
    for file in os.listdir(target_dir):
        if file.lower().endswith(".mp4"):
            full_path = os.path.join(target_dir, file)
            size_mb = round(os.path.getsize(full_path) / (1024 * 1024), 2)
            clips.append({
                "filename": file,
                "path": full_path,
                "size_mb": size_mb,
            })

    clips.sort(key=lambda x: x["filename"])
    return {"clips": clips, "output_dir": target_dir}


@app.get("/api/stream-clip")
async def stream_clip(path: str):
    """Stream cut video clip file directly for browser video player."""
    clean_path = os.path.abspath(path.strip())
    if not os.path.exists(clean_path):
        raise HTTPException(status_code=404, detail="Clip file not found")
    return FileResponse(clean_path, media_type="video/mp4")


# ══════════════════════════════════════════════════════════════════════
# Serve Static Front-End Client
# ══════════════════════════════════════════════════════════════════════

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Video Scene Detector Web GUI</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_graceful_shutdown=1)


