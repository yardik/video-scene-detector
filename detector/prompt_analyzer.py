"""
Prompt Quality Analyzer for Video Scene Detection.

Scores user prompts for visual specificity and potential ambiguity, and
suggests concrete improvements.
"""

import sys
from dataclasses import dataclass, field
from typing import List


@dataclass
class PromptAnalysis:
    """Result of prompt quality analysis."""
    original_prompt: str
    quality_score: float              # 0.0 (poor) to 1.0 (excellent)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class PromptAnalyzer:
    """
    Analyzes scene detection prompts for visual specificity and ambiguity,
    and suggests concrete improvements.
    """

    # Visual anchor categories
    BODY_WORDS = [
        'face', 'head', 'hand', 'hands', 'arm', 'arms', 'leg', 'legs',
        'back', 'chest', 'body', 'knee', 'knees', 'mouth', 'eyes',
    ]
    POSITION_WORDS = [
        'standing', 'sitting', 'kneeling', 'lying', 'bending', 'leaning',
        'on all fours', 'doggy', 'missionary', 'cowgirl', 'reverse',
        'on top', 'underneath', 'beside', 'on her back', 'on his back',
    ]
    SPATIAL_WORDS = [
        'behind', 'front', 'facing', 'toward', 'away', 'above', 'below',
        'left', 'right', 'next to', 'between', 'on top of', 'under',
    ]
    CAMERA_WORDS = [
        'camera', 'angle', 'view', 'close-up', 'closeup', 'wide shot',
        'pov', 'overhead', 'side view', 'profile',
    ]

    def analyze(self, prompt: str) -> PromptAnalysis:
        """Analyze prompt quality."""
        issues = []
        suggestions = []
        score = 1.0

        prompt = prompt.strip()
        words = prompt.split()

        # ── Length check ──
        if len(words) < 4:
            score -= 0.3
            issues.append("Prompt is very short and likely too generic for accurate detection.")
            suggestions.append(
                "Expand with visual details: body positions, spatial relationships, "
                "camera angle, and distinguishing actions."
            )
        elif len(words) < 8:
            score -= 0.1

        # ── Visual specificity checks ──
        prompt_lower = prompt.lower()

        has_body = any(w in prompt_lower for w in self.BODY_WORDS)
        has_position = any(w in prompt_lower for w in self.POSITION_WORDS)
        has_spatial = any(w in prompt_lower for w in self.SPATIAL_WORDS)
        has_camera = any(w in prompt_lower for w in self.CAMERA_WORDS)

        specificity_count = sum([has_body, has_position, has_spatial, has_camera])

        if specificity_count == 0:
            score -= 0.25
            issues.append("No visual anchor words detected (body parts, positions, spatial terms, camera angles).")
            suggestions.append(
                "Add concrete visual descriptions like: 'facing the camera', "
                "'on all fours', 'standing behind', 'close-up of face'."
            )
        elif specificity_count == 1:
            score -= 0.1
            missing = []
            if not has_position:
                missing.append("body positions (standing, kneeling, lying)")
            if not has_spatial:
                missing.append("spatial relationships (behind, facing, next to)")
            if not has_camera:
                missing.append("camera perspective (facing camera, side view)")
            if missing:
                suggestions.append(f"Consider adding: {', '.join(missing[:2])}.")

        # ── Ambiguity checks ──
        vague_terms = ['something', 'somehow', 'kind of', 'sort of', 'maybe', 'possibly']
        if any(term in prompt_lower for term in vague_terms):
            score -= 0.15
            issues.append("Prompt contains vague/uncertain language.")
            suggestions.append("Use definitive visual descriptions instead of uncertain terms.")

        score = max(0.0, min(1.0, score))

        return PromptAnalysis(
            original_prompt=prompt,
            quality_score=score,
            issues=issues,
            suggestions=suggestions,
        )

    def print_analysis(self, analysis: PromptAnalysis) -> None:
        """Print a formatted prompt analysis report to stderr."""
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  Prompt Quality Analysis", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"  Prompt: \"{analysis.original_prompt}\"", file=sys.stderr)

        # Quality score bar
        bar_len = 20
        filled = int(analysis.quality_score * bar_len)
        bar = '#' * filled + '-' * (bar_len - filled)
        label = "Excellent" if analysis.quality_score >= 0.8 else \
                "Good" if analysis.quality_score >= 0.6 else \
                "Fair" if analysis.quality_score >= 0.4 else "Poor"
        print(f"  Quality: [{bar}] {analysis.quality_score:.0%} ({label})", file=sys.stderr)

        if analysis.issues:
            print(f"\n  Issues:", file=sys.stderr)
            for issue in analysis.issues:
                print(f"    - {issue}", file=sys.stderr)

        if analysis.suggestions:
            print(f"\n  Suggestions:", file=sys.stderr)
            for suggestion in analysis.suggestions:
                print(f"    - {suggestion}", file=sys.stderr)

        print(f"{'='*60}\n", file=sys.stderr)
