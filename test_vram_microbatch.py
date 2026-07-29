"""
VRAM Micro-Batch Fragmentation Test
====================================
Reproduces and measures the VRAM explosion when processing 16 frames
as 2 micro-batches of 8 vs. a single batch of 16.

Usage:
    python test_vram_microbatch.py

Requires the model to be available in models_cache/ (same setup as the main app).
"""

import os
import sys
import gc
import time

# ── Must set allocator config BEFORE importing torch ──
# Toggle this to test the fix vs. the old behavior:
USE_FIX = False

if USE_FIX:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print("🔧 ALLOCATOR CONFIG: expandable_segments:True (FIX APPLIED)")
else:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    print("🔧 ALLOCATOR CONFIG: max_split_size_mb:128 (OLD BEHAVIOR)")

import torch
from PIL import Image

# ── Add project root to path so detector module imports work ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def fmt_mb(bytes_val: int) -> str:
    mb = bytes_val / (1024 ** 2)
    gb = bytes_val / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.2f} GB ({mb:.0f} MB)"
    return f"{mb:.0f} MB"


def vram_snapshot(tag: str = "") -> dict:
    """Print and return current VRAM stats."""
    if not torch.cuda.is_available():
        print(f"  [VRAM {tag}] CUDA not available")
        return {}
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    print(
        f"  [VRAM {tag}] "
        f"Active Tensors: {fmt_mb(allocated)} | "
        f"Reserved Pool: {fmt_mb(reserved)} | "
        f"Peak Active: {fmt_mb(peak)}"
    )
    return {"allocated": allocated, "reserved": reserved, "peak": peak}


def full_vram_reset():
    """Aggressively free all cached VRAM and reset peak stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def create_synthetic_frames(n: int, size: tuple = (384, 384)) -> list:
    """Create n synthetic PIL images (different colors so they're not deduplicated)."""
    frames = []
    for i in range(n):
        r = (i * 37) % 256
        g = (i * 73 + 100) % 256
        b = (i * 131 + 50) % 256
        img = Image.new("RGB", size, (r, g, b))
        frames.append(img)
    return frames


def run_single_batch(model, processor, frames, chat_prompt, tag="Single Batch"):
    """Run all frames in a single batch. Returns peak VRAM."""
    print(f"\n{'='*70}")
    print(f"  TEST: {tag} ({len(frames)} frames)")
    print(f"{'='*70}")

    full_vram_reset()
    vram_snapshot("Before inference")

    inputs = processor(
        text=[chat_prompt] * len(frames),
        images=frames,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    vram_snapshot("After tokenization (on GPU)")

    t0 = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    elapsed = time.time() - t0

    stats = vram_snapshot(f"After generate() [{elapsed:.2f}s]")

    # Decode outputs
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answers = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    print(f"  Sample output: {answers[0][:60]}...")

    del inputs, generated_ids, generated_ids_trimmed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    vram_snapshot("After cleanup")
    return stats


def run_micro_batched(model, processor, frames, chat_prompt, batch_size, tag="Micro-Batched"):
    """Run frames in micro-batches. Returns peak VRAM."""
    n_chunks = (len(frames) + batch_size - 1) // batch_size
    chunks = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]

    print(f"\n{'='*70}")
    print(f"  TEST: {tag} ({len(frames)} frames in {n_chunks} chunks of {batch_size})")
    print(f"{'='*70}")

    full_vram_reset()
    vram_snapshot("Before inference")

    all_answers = []
    for chunk_idx, chunk_images in enumerate(chunks, 1):
        print(f"\n  --- Chunk {chunk_idx}/{n_chunks} ({len(chunk_images)} frames) ---")

        inputs = processor(
            text=[chat_prompt] * len(chunk_images),
            images=chunk_images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        vram_snapshot(f"Chunk {chunk_idx} after tokenization")

        t0 = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        elapsed = time.time() - t0
        vram_snapshot(f"Chunk {chunk_idx} after generate() [{elapsed:.2f}s]")

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        all_answers.extend(
            processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        )

        # Cleanup between chunks (matches the app's micro-batch cleanup)
        del inputs, generated_ids, generated_ids_trimmed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        vram_snapshot(f"Chunk {chunk_idx} after cleanup")

    stats = vram_snapshot("Final (all chunks done)")
    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    print(f"  Sample output: {all_answers[0][:60]}...")
    return {"peak": peak, **stats}


def main():
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"🐍 PyTorch: {torch.__version__}")
    print(f"📦 CUDA Allocator Config: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")

    # ── Load model using the app's own loader ──
    print("\n📦 Loading model (this may take a moment)...")
    from detector.qwen_model import get_cached_qwen_model
    qwen = get_cached_qwen_model(quantization="4bit")

    if qwen.is_video_native:
        print("⚠️  Model is video-native — micro-batching path only applies to image-only models.")
        print("    This test simulates the image-only code path anyway for VRAM measurement.")

    model = qwen.model
    processor = qwen.processor

    # Build chat prompt (same as the app builds for image-only detection)
    image_token = getattr(processor, "image_token", "<image>")
    prompt_text = (
        f"{image_token}\nWatch this video frame carefully. Question: Does this show the following: "
        f"\"a person walking\"? Answer ONLY YES or NO, followed by a brief reason."
    )
    msgs = [{"role": "user", "content": prompt_text}]
    try:
        chat_prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        chat_prompt = f"USER: {image_token}\n{prompt_text}\nASSISTANT:"

    # Ensure left-padding for batched generation
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Create synthetic test frames ──
    TOTAL_FRAMES = 16
    frames = create_synthetic_frames(TOTAL_FRAMES)
    print(f"\n🎞️  Created {TOTAL_FRAMES} synthetic {frames[0].size[0]}x{frames[0].size[1]} frames")

    # ── Warm-up pass (first generate() has one-time overhead) ──
    print("\n🔥 Running warm-up pass (2 frames)...")
    warmup_frames = create_synthetic_frames(2)
    warmup_inputs = processor(
        text=[chat_prompt] * 2, images=warmup_frames, padding=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        _ = model.generate(**warmup_inputs, max_new_tokens=8, do_sample=False)
    del warmup_inputs, warmup_frames, _
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("  Warm-up done.\n")

    # ══════════════════════════════════════════════════════════════════
    # TEST A: Single batch of 16 frames
    # ══════════════════════════════════════════════════════════════════
    stats_single = run_single_batch(model, processor, frames, chat_prompt, "Single Batch of 16")
    peak_single = stats_single.get("peak", 0)

    # ══════════════════════════════════════════════════════════════════
    # TEST B: Two micro-batches of 8 frames
    # ══════════════════════════════════════════════════════════════════
    stats_micro = run_micro_batched(model, processor, frames, chat_prompt, 8, "2× Micro-Batches of 8")
    peak_micro = stats_micro.get("peak", 0)

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Allocator Config:     {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'default')}")
    print(f"  Single Batch (16):    Peak VRAM = {fmt_mb(peak_single)}")
    print(f"  Micro-Batch (2×8):    Peak VRAM = {fmt_mb(peak_micro)}")

    if peak_single > 0:
        ratio = peak_micro / peak_single
        diff_gb = (peak_micro - peak_single) / (1024**3)
        print(f"  Ratio:                {ratio:.2f}× ({'+' if diff_gb > 0 else ''}{diff_gb:.2f} GB)")

        if ratio > 1.5:
            print(f"\n  ❌ FRAGMENTATION DETECTED: Micro-batching uses {ratio:.1f}× more VRAM!")
            print(f"     The allocator is not reusing memory between generate() calls.")
        elif ratio > 1.1:
            print(f"\n  ⚠️  Moderate overhead ({ratio:.1f}×) — some fragmentation but manageable.")
        else:
            print(f"\n  ✅ VRAM usage is comparable ({ratio:.1f}×) — fix is working!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
