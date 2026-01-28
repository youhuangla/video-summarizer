"""Tests for metadata extractor."""

import pytest
from pathlib import Path
from video_summarizer.extractors.metadata import MetadataExtractor, VideoMetadata


def test_video_metadata_creation():
    """Test VideoMetadata dataclass."""
    meta = VideoMetadata(
        duration=120.5,
        fps=30.0,
        resolution=(1920, 1080),
        total_frames=3615
    )
    assert meta.duration == 120.5
    assert meta.fps == 30.0
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.total_frames == 3615


def test_metadata_extractor_init():
    """Test MetadataExtractor initialization."""
    extractor = MetadataExtractor()
    assert extractor is not None


def test_extract_nonexistent_file():
    """Test extracting metadata from non-existent file raises error."""
    extractor = MetadataExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract("/nonexistent/video.mp4")


def test_video_metadata_properties():
    """Test VideoMetadata convenience properties."""
    meta = VideoMetadata(
        duration=60.0,
        fps=25.0,
        resolution=(1280, 720),
        total_frames=1500
    )
    assert meta.resolution == (1280, 720)
    assert meta.width == 1280
    assert meta.height == 720
