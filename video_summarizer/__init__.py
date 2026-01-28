"""Video Summarizer - Generate timestamped Markdown summaries from videos.

This package provides tools to analyze video content using Kimi VLM
and generate chapter-based summaries in Markdown format.

Inspired by Video-Browser (https://github.com/chrisx599/Video-Browser)
paper's pyramidal perception architecture.

External dependencies:
- decord: https://github.com/dmlc/decord (Apache-2.0)
- whisper: Local HTTP API at http://127.0.0.1:18181/v1/audio/
- openai: https://github.com/openai/openai-python (Apache-2.0)
"""

__version__ = "0.1.0"

from video_summarizer.config import SummarizerConfig
from video_summarizer.pipeline import VideoSummarizerPipeline, VideoSummaryResult

__all__ = ["SummarizerConfig", "VideoSummarizerPipeline", "VideoSummaryResult"]
