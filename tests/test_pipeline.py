"""Tests for main pipeline."""

import pytest
from pathlib import Path
from video_summarizer.pipeline import VideoSummarizerPipeline, VideoSummaryResult
from video_summarizer.config import SummarizerConfig


def test_pipeline_init():
    """Test pipeline initialization."""
    config = SummarizerConfig(api_key="test")
    pipeline = VideoSummarizerPipeline(config)
    assert pipeline is not None
    assert pipeline.config == config


def test_fmt_time():
    """Test time formatting helper."""
    config = SummarizerConfig(api_key="test-key")
    pipeline = VideoSummarizerPipeline(config)
    assert pipeline._fmt_time(125.5) == "02:05"
    assert pipeline._fmt_time(60) == "01:00"
    assert pipeline._fmt_time(0) == "00:00"


def test_merge_short_chapters():
    """Test merging short chapters."""
    config = SummarizerConfig(api_key="test-key")
    pipeline = VideoSummarizerPipeline(config)
    
    chapters = [
        {"start_time": 0, "end_time": 100, "summary": "Chapter 1"},
        {"start_time": 100, "end_time": 110, "summary": "Short"},  # Too short
        {"start_time": 110, "end_time": 200, "summary": "Chapter 2"},
    ]
    
    merged = pipeline._merge_short_chapters(chapters)
    assert len(merged) == 2
    assert "Short" in merged[0]["summary"]  # Merged into first


def test_video_summary_result():
    """Test VideoSummaryResult dataclass."""
    result = VideoSummaryResult(
        video_path=Path("test.mp4"),
        duration=120.0,
        overall_summary="测试摘要",
        chapters=[],
        processing_time=5.0,
        output_path=Path("output.md")
    )
    assert result.duration == 120.0
    assert result.processing_time == 5.0
