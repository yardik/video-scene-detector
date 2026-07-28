"""Prompt Management System for Video Scene Detector.

Persists and reloads editable System Prompts for:
1. Captioner (Vision Model)
2. Matcher (Text LLM Classifier)

Prompts are stored as plain text files in the project's 'prompts/' directory.
"""

import os
import sys

PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prompts"))
os.makedirs(PROMPTS_DIR, exist_ok=True)

CAPTIONER_PROMPT_FILE = os.path.join(PROMPTS_DIR, "captioner_prompt.txt")
MATCHER_PROMPT_FILE = os.path.join(PROMPTS_DIR, "matcher_prompt.txt")

DEFAULT_CAPTIONER_PROMPT = (
    "Observe this video clip carefully and describe exclusively the people present, their exact spatial positions, body postures, and physical actions.\n\n"
    "Structure your response strictly into these two sections:\n"
    "- PEOPLE & SPATIAL POSITIONS: Describe every person present, their exact placement in the frame (e.g. left, right, center, foreground, background), spatial positioning relative to each other (e.g. sitting side-by-side, standing behind, facing each other), body posture (e.g. seated upright, leaning forward, standing, arms crossed), facial expression, hair/attire, and gaze direction.\n"
    "- ACTIONS & MOVEMENTS: Describe step-by-step the physical movements, hand gestures, body posture changes, and direct interactions between the people across the clip.\n\n"
    "Be objective and specific. Focus exclusively on the people, their spatial placement, and their actions."
)

DEFAULT_MATCHER_PROMPT = (
    "You are an expert video scene matching and similarity evaluation assistant.\n"
    "You will be given an exhaustive visual description of a video clip and a target scene prompt.\n\n"
    "Your task is to conduct a dual-layer evaluation:\n"
    "1. PHRASING & TERMINOLOGY SIMILARITY: Analyze exact or near-synonymous key phrases, action verbs, subject names, object labels, visible text, and distinct wording shared between the target prompt and the video description. Matching specific phrasing or key terminology strongly indicates a higher quality match.\n"
    "2. VISUAL & SEMANTIC ALIGNMENT: Check if the described physical setting, subjects, attire, and step-by-step actions match the underlying context requested in the target prompt.\n\n"
    "Based on both phrasing similarities and visual alignment, assign a MATCH PROBABILITY percentage from 0% to 100%.\n\n"
    "Respond strictly in the following format:\n"
    "PROBABILITY: <integer score from 0 to 100>%\n"
    "PHRASING MATCHES: <list key phrases, matching vocabulary, or terms shared between prompt and description>\n"
    "REASONING: <detailed evaluation of visual similarities, phrasing alignment, and any discrepancies>"
)


def get_captioner_prompt() -> str:
    """Reads the captioner prompt from prompts/captioner_prompt.txt.
    Creates default file if it doesn't exist.
    """
    if not os.path.exists(CAPTIONER_PROMPT_FILE):
        save_captioner_prompt(DEFAULT_CAPTIONER_PROMPT)
        return DEFAULT_CAPTIONER_PROMPT

    try:
        with open(CAPTIONER_PROMPT_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return content if content else DEFAULT_CAPTIONER_PROMPT
    except Exception as e:
        print(f"⚠️ Error reading captioner prompt file: {e}", file=sys.stderr)
        return DEFAULT_CAPTIONER_PROMPT


def get_matcher_prompt() -> str:
    """Reads the matcher system prompt from prompts/matcher_prompt.txt.
    Creates default file if it doesn't exist.
    """
    if not os.path.exists(MATCHER_PROMPT_FILE):
        save_matcher_prompt(DEFAULT_MATCHER_PROMPT)
        return DEFAULT_MATCHER_PROMPT

    try:
        with open(MATCHER_PROMPT_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return content if content else DEFAULT_MATCHER_PROMPT
    except Exception as e:
        print(f"⚠️ Error reading matcher prompt file: {e}", file=sys.stderr)
        return DEFAULT_MATCHER_PROMPT


def save_captioner_prompt(prompt_text: str) -> bool:
    """Saves updated captioner prompt text to prompts/captioner_prompt.txt."""
    try:
        os.makedirs(PROMPTS_DIR, exist_ok=True)
        with open(CAPTIONER_PROMPT_FILE, "w", encoding="utf-8") as f:
            f.write(prompt_text.strip())
        print(f"📝 Captioner prompt saved to {CAPTIONER_PROMPT_FILE}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"❌ Error saving captioner prompt: {e}", file=sys.stderr)
        return False


def save_matcher_prompt(prompt_text: str) -> bool:
    """Saves updated matcher prompt text to prompts/matcher_prompt.txt."""
    try:
        os.makedirs(PROMPTS_DIR, exist_ok=True)
        with open(MATCHER_PROMPT_FILE, "w", encoding="utf-8") as f:
            f.write(prompt_text.strip())
        print(f"📝 Matcher prompt saved to {MATCHER_PROMPT_FILE}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"❌ Error saving matcher prompt: {e}", file=sys.stderr)
        return False
