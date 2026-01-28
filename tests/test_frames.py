"""Tests for frame extractor."""

import pytest
from pathlib import Path
from video_summarizer.extractors.frames import FrameExtractor, FrameInfo


def test_frame_info_creation():
    """Test FrameInfo creation."""
    frame = FrameInfo(
        path=Path("/tmp/frame_001.jpg"),
        timestamp=10.5,
        frame_number=315
    )
    assert frame.timestamp == 10.5
    assert frame.frame_number == 315


def test_frame_extractor_init(tmp_path):
    """Test FrameExtractor initialization."""
    extractor = FrameExtractor(cache_dir=str(tmp_path / "frames"))
    assert extractor.cache_dir.exists()


def test_compute_video_id(tmp_path):
    """Test video ID computation."""
    extractor = FrameExtractor(cache_dir=str(tmp_path))
    
    test_file = tmp_path / "test.mp4"
    test_file.write_bytes(b"test content for hashing")
    
    video_id = extractor._compute_video_id(test_file)
    assert len(video_id) == 12
    # Same file should produce same ID
    assert video_id == extractor._compute_video_id(test_file)


def test_frame_info_base64(tmp_path):
    """Test FrameInfo to_base64 conversion."""
    # Create a dummy image file
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"fake image data")
    
    frame = FrameInfo(path=img_path, timestamp=5.0, frame_number=150)
    b64 = frame.to_base64()
    
    assert isinstance(b64, str)
    # base64 encoded "fake image data"
    assert b64 == "ZmFrZSBpbWFnZSBkYXRh"
