import os
import sys
import tempfile

import gradio as gr
from detector import VideoSceneDetector, VideoUtils

# Reconfigure stdout for UTF-8 compatibility on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Global cached detector instance
detector_instance = None
current_model_path = None


def get_detector(model_path: str, uncensored: bool):
    global detector_instance, current_model_path
    if (
        detector_instance is None
        or current_model_path != model_path
        or detector_instance.model.uncensored != uncensored
    ):
        detector_instance = VideoSceneDetector(
            model_name_or_path=model_path,
            uncensored=uncensored,
        )
        current_model_path = model_path
    return detector_instance


def resolve_video_path(path_text: str, upload_path: str) -> str:
    """Returns local workstation path if provided, otherwise uploaded file path."""
    if path_text and path_text.strip():
        clean_path = path_text.strip().strip('"').strip("'")
        if os.path.exists(clean_path):
            return clean_path
    if upload_path and os.path.exists(upload_path):
        return upload_path
    return ""


def process_video_info(path_text, upload_path):
    video_path = resolve_video_path(path_text, upload_path)
    if not video_path:
        return "Enter a valid video file path on your workstation or upload a file."
    try:
        info = VideoUtils.get_video_info(video_path)
        return (
            f"**Resolved Path**: `{video_path}`\n\n"
            f"**Duration**: {info['duration_str']} ({info['duration']}s)\n\n"
            f"**FPS**: {info['fps']} | **Resolution**: {info['resolution']} | **Frames**: {info['total_frames']}"
        )
    except Exception as e:
        return f"Error reading video metadata: {e}"


def run_scene_detection(
    path_text, upload_path, prompt, model_path, fps_sample, max_frames_val, two_pass_toggle, uncensored_mode, cut_clip_option
):
    video_path = resolve_video_path(path_text, upload_path)
    if not video_path:
        return "[Error] Please specify a valid video file path on your workstation or upload a video.", "", None, None, None
    if not prompt or not prompt.strip():
        return "[Error] Please provide a scene description prompt.", "", None, None, None

    try:
        detector = get_detector(model_path, uncensored_mode)

        match = detector.detect_scene(
            video_path=video_path,
            prompt=prompt.strip(),
            fps=float(fps_sample),
            max_frames=int(max_frames_val),
            two_pass_refinement=two_pass_toggle,
            cut_clip=cut_clip_option,
        )

        status_markdown = (
            f"### Target Scene Detection Result\n\n"
            f"| Metric | Value |\n"
            f"| --- | --- |\n"
            f"| **Video Path** | `{match.video_path}` |\n"
            f"| **Found Scene** | {'YES' if match.found else 'NO'} |\n"
            f"| **Start Timestamp** | `{match.start_time}` ({match.start_seconds}s) |\n"
            f"| **End Timestamp** | `{match.end_time}` ({match.end_seconds}s) |\n"
            f"| **Scene Duration** | {round(match.end_seconds - match.start_seconds, 2)} seconds |\n\n"
            f"**Explanation / Visual Description:**\n>{match.explanation}\n"
        )

        # Temp files for download
        temp_dir = tempfile.mkdtemp()
        json_file = os.path.join(temp_dir, "scene_result.json")
        srt_file = os.path.join(temp_dir, "scene_result.srt")

        VideoSceneDetector.export_json(match, json_file)
        VideoSceneDetector.export_srt(match, srt_file)

        return (
            status_markdown,
            match.explanation,
            match.clip_path if match.clip_path else None,
            json_file,
            srt_file,
        )

    except Exception as e:
        error_msg = f"[Error] Analysis failed: {str(e)}"
        return error_msg, "", None, None, None


def build_ui():
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    .header-box {
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2b55 100%);
        border-radius: 12px;
        color: #ffffff;
    }
    """

    with gr.Blocks(title="Qwen2-VL Video Scene Detector", css=custom_css) as app:
        gr.HTML(
            """
            <div class="header-box">
                <h1>Qwen2-VL Video Scene Detector</h1>
                <p>Locate exact scene timestamps in videos using natural language prompts powered by Qwen2-VL with uncensored prompt support.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_path_input = gr.Textbox(
                    label="Workstation Video File Path",
                    placeholder="e.g. D:/Videos/my_movie.mp4 or C:\\Users\\Name\\Videos\\clip.mp4",
                    info="Specify local file path directly from your computer (no upload required!).",
                )

                with gr.Accordion("Or Upload / Preview Video File", open=False):
                    video_upload_input = gr.Video(label="Upload Video", sources=["upload"])

                video_info_box = gr.Markdown("Enter a file path or upload a video to inspect metadata.")

                prompt_input = gr.Textbox(
                    label="Target Scene Description (Natural Language)",
                    placeholder="Enter target scene description prompt...",
                    lines=3,
                )

                with gr.Accordion("Advanced Settings", open=False):
                    model_selector = gr.Dropdown(
                        label="Model ID / Local Model Path",
                        choices=[
                            "Qwen/Qwen2-VL-2B-Instruct",
                            "Qwen/Qwen2-VL-7B-Instruct",
                            "Qwen/Qwen2.5-VL-3B-Instruct",
                            "Qwen/Qwen2.5-VL-7B-Instruct",
                        ],
                        value="Qwen/Qwen2-VL-2B-Instruct",
                        allow_custom_value=True,
                    )
                    two_pass_checkbox = gr.Checkbox(
                        label="Enable 2-Pass Hierarchical Refinement (Coarse-to-Fine Zoom)",
                        value=True,
                        info="Pass 1 scans full video; Pass 2 zooms into target scene for sub-second precision.",
                    )
                    fps_slider = gr.Slider(
                        label="Frame Extraction Sampling (FPS)",
                        minimum=0.2,
                        maximum=4.0,
                        step=0.2,
                        value=1.0,
                    )
                    max_frames_slider = gr.Slider(
                        label="Explicit Frame Count Target (~30k Token Target)",
                        minimum=32,
                        maximum=256,
                        step=16,
                        value=224,
                        info="Default 224 frames targets ~28,000 to 30,000 tokens (utilizing ~90% of the 32,768 context limit).",
                    )
                    uncensored_toggle = gr.Checkbox(
                        label="Enable Uncensored Permissive System Prompt",
                        value=True,
                        info="Ensures model executes visual grounding without refusal on sensitive or unrestricted prompts.",
                    )
                    cut_clip_toggle = gr.Checkbox(
                        label="Automatically extract video clip of detected scene",
                        value=True,
                    )

                detect_btn = gr.Button("Detect Target Scene", variant="primary", size="lg")

            with gr.Column(scale=1):
                results_markdown = gr.Markdown("### Detection Results will appear here...")

                clip_player = gr.Video(label="Detected Scene Preview Clip", interactive=False)

                with gr.Row():
                    json_download = gr.File(label="Download Result JSON")
                    srt_download = gr.File(label="Download Result SRT Subtitles")

        # Event handlers
        video_path_input.change(
            fn=process_video_info,
            inputs=[video_path_input, video_upload_input],
            outputs=[video_info_box],
        )
        video_upload_input.change(
            fn=process_video_info,
            inputs=[video_path_input, video_upload_input],
            outputs=[video_info_box],
        )

        detect_btn.click(
            fn=run_scene_detection,
            inputs=[
                video_path_input,
                video_upload_input,
                prompt_input,
                model_selector,
                fps_slider,
                max_frames_slider,
                two_pass_checkbox,
                uncensored_toggle,
                cut_clip_toggle,
            ],
            outputs=[
                results_markdown,
                gr.State(),
                clip_player,
                json_download,
                srt_download,
            ],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
