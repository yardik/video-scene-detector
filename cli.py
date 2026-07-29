#!/usr/bin/env python3
"""CLI and Web Server entrypoint for Video Scene Detector."""

import argparse
import json
import sys
import os

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from detector import VideoSceneDetector, BatchProcessor
from detector.qwen_model import IMAGE_ONLY_MICRO_BATCH_SIZE


def main():
    parser = argparse.ArgumentParser(
        description="Detect scenes in videos using AI vision models."
    )
    parser.add_argument(
        "--video", default=None, help="Path to a single video file"
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Path to an input directory containing video files for batch processing",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="Natural language description of the scene to find",
    )
    parser.add_argument(
        "--cut-clip", action="store_true",
        help="Cut and save video clips for each detected scene",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path for results",
    )
    parser.add_argument(
        "--model", default=None,
        help="HuggingFace model ID or local safetensors path. If omitted, uses the first "
             "vision model already installed in models_cache/hub/ (suggested: Qwen/Qwen2.5-VL-7B-Instruct).",
    )
    parser.add_argument(
        "--quantization", default="4bit", choices=["4bit", "none", "bfloat16"],
        help="Quantization mode for GPU VRAM optimization (default: 4bit)",
    )
    parser.add_argument(
        "--dual-model", action="store_true", default=True,
        help="Enable Dual Model Scene Classification (vision description -> text LLM reasoning, default: Enabled)",
    )
    parser.add_argument(
        "--single-model", action="store_false", dest="dual_model",
        help="Disable Dual Model Scene Classification (use single VLM model)",
    )
    parser.add_argument(
        "--text-model", default=None,
        help="Text LLM model for dual-model classification. If omitted, uses the first "
             "text model already installed in models_cache/hub/ (suggested: Qwen/Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--similarity-threshold", "--threshold", type=float, default=50.0,
        help="Similarity match probability percentage threshold for dual-model mode (default: 50.0)",
    )
    parser.add_argument(
        "--coarse-window-min", type=float, default=2.0,
        help="Coarse scan window length in minutes (range: 1.0 to 5.0, default: 2.0)",
    )
    parser.add_argument(
        "--coarse-frames", type=int, default=16,
        help="Number of frames sampled across each coarse window (range: 8 to 64, default: 16)",
    )
    parser.add_argument(
        "--smart-motion", action="store_true",
        help="Enable Smart Motion Keyframe Sampling (filters out duplicate static frames before VLM inference)",
    )
    parser.add_argument(
        "--sage-attn", dest="sage_attn", action="store_true", default=True,
        help="Enable SageAttention 2 INT8 Acceleration (NHD contiguous layout) (default: on)",
    )
    parser.add_argument(
        "--no-sage-attn", dest="sage_attn", action="store_false",
        help="Disable SageAttention 2 and use native PyTorch SDPA instead",
    )
    parser.add_argument(
        "--image-only-micro-batch", type=int, default=IMAGE_ONLY_MICRO_BATCH_SIZE,
        help="Frames processed concurrently per generate() call for single-image "
             f"(non video-native) vision models — lower to reduce VRAM use at the cost of "
             f"wall-clock time (default: {IMAGE_ONLY_MICRO_BATCH_SIZE}). No effect on "
             "video-native models.",
    )
    parser.add_argument(
        "--text-llm-batch", type=int, default=10,
        help="Number of scene descriptions processed concurrently during Text LLM classification "
             "in micro-batches (1 to 16, default: 10). Lower to reduce peak memory usage.",
    )
    parser.add_argument(
        "--start-time", "--start", default=None,
        help="Optional start timestamp (e.g. 00:01:30 or 90.0) to restrict detection window",
    )
    parser.add_argument(
        "--end-time", "--end", default=None,
        help="Optional end timestamp (e.g. 00:05:00 or 300.0) to restrict detection window",
    )
    parser.add_argument(
        "--clip-dir", default=None,
        help="Directory to save cut clips (default: current directory)",
    )
    parser.add_argument(
        "--web", "--gui", action="store_true",
        help="Launch local Web Client Studio server",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to run Web Client Studio server (default: 8000)",
    )

    args = parser.parse_args()

    # Launch Web Server if --web or no arguments provided
    if args.web or (not args.video and not args.input_dir):
        print(f"\n🌐 Launching Video Scene Detector Web Client Studio on http://localhost:{args.port} ...\n")
        import uvicorn
        uvicorn.run("web.app:app", host="0.0.0.0", port=args.port, reload=False, timeout_graceful_shutdown=1)
        return

    if not args.prompt:
        parser.error("--prompt is required when running CLI detection")

    quant_setting = None if args.quantization == "none" else args.quantization

    default_export_dir = os.path.join(os.getcwd(), "extracted_clips")

    # Folder Batch Processing Mode
    if args.input_dir:
        clip_out = args.clip_dir or default_export_dir
        print(f"\n📁 Batch Folder Mode: Input={args.input_dir} | Output={clip_out}")
        
        processor = BatchProcessor(
            model_name_or_path=args.model,
            quantization=quant_setting,
            dual_model=args.dual_model,
            text_model_name=args.text_model,
            similarity_threshold=args.similarity_threshold,
            coarse_window_minutes=args.coarse_window_min,
            coarse_frames=args.coarse_frames,
            use_sage_attention=args.sage_attn,
            image_only_micro_batch_size=args.image_only_micro_batch,
            text_llm_batch_size=args.text_llm_batch,
        )
        
        processor.process_batch(
            input_dir=args.input_dir,
            output_dir=clip_out,
            prompt=args.prompt,
            progress_callback=lambda prog, log: print(log),
        )
        return

    # Single Video Mode
    if args.video:
        detector = VideoSceneDetector(
            model_name_or_path=args.model,
            quantization=quant_setting,
            dual_model=args.dual_model,
            text_model_name=args.text_model,
            similarity_threshold=args.similarity_threshold,
            coarse_window_minutes=args.coarse_window_min,
            coarse_frames=args.coarse_frames,
            use_sage_attention=args.sage_attn,
            image_only_micro_batch_size=args.image_only_micro_batch,
            text_llm_batch_size=args.text_llm_batch,
        )

        scenes = detector.detect_scenes(
            video_path=args.video,
            prompt=args.prompt,
            cut_clips=args.cut_clip,
            clip_output_dir=args.clip_dir or default_export_dir,
            start_time=args.start_time,
            end_time=args.end_time,
        )

        # Print results summary
        if scenes:
            print(f"\nFound {len(scenes)} scene(s):\n")
            for s in scenes:
                duration = s.end_seconds - s.start_seconds
                print(f"  Scene {s.scene_index}:")
                print(f"    Time:     {s.start_time} -> {s.end_time} ({duration:.1f}s)")
                print(f"    Seconds:  {s.start_seconds}s -> {s.end_seconds}s")
                if s.clip_path:
                    print(f"    Clip:     {s.clip_path}")
                print()
        else:
            print(f"\nNo scenes matching \"{args.prompt}\" were found.\n")

        # Export JSON
        if args.output:
            if scenes:
                results = [s.to_dict() for s in scenes]
            else:
                results = [{
                    "video_path": args.video,
                    "prompt": args.prompt,
                    "found": False,
                    "explanation": "No matching scenes found.",
                }]
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results exported to: {args.output}")

        print("Done!")


if __name__ == "__main__":
    main()
