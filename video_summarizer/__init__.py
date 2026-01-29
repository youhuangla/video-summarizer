"""Video Summarizer - Generate timestamped Markdown summaries from videos.

This package provides tools to analyze video content and generate
chapter-based summaries or transcripts.

Features:
- Full mode: VLM analysis with chapter summaries
- Fallback mode: Audio transcription only (when VLM unavailable)

External dependencies:
- decord: https://github.com/dmlc/decord (Apache-2.0)
- whisper: Local API or direct library
- openai: https://github.com/openai/openai-python (Apache-2.0)
"""

__version__ = "0.1.0"

from video_summarizer.config import SummarizerConfig
from video_summarizer.pipeline import VideoSummarizerPipeline, VideoSummaryResult
from video_summarizer.fallback_pipeline import FallbackPipeline, FallbackResult, check_vlm_available
from video_summarizer.output.transcript_exporter import TranscriptExporter

__all__ = [
    "SummarizerConfig",
    "VideoSummarizerPipeline",
    "VideoSummaryResult",
    "FallbackPipeline",
    "FallbackResult",
    "check_vlm_available",
    "TranscriptExporter",
]
