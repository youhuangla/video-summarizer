"""Tests for audio extractor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from video_summarizer.extractors.audio import AudioExtractor, TranscriptSegment


def test_transcript_segment_creation():
    """Test TranscriptSegment creation."""
    seg = TranscriptSegment(start=10.5, end=15.0, text="Hello world")
    assert seg.start == 10.5
    assert seg.end == 15.0
    assert seg.text == "Hello world"


def test_audio_extractor_init():
    """Test AudioExtractor initialization."""
    extractor = AudioExtractor(api_base="http://localhost:8000")
    assert extractor.api_base == "http://localhost:8000"


def test_get_transcript_text():
    """Test getting transcript text within time range."""
    extractor = AudioExtractor()
    segments = [
        TranscriptSegment(0, 5, "First sentence."),
        TranscriptSegment(5, 10, "Second sentence."),
        TranscriptSegment(10, 15, "Third sentence."),
    ]
    
    # Get all text
    text = extractor.get_transcript_text(segments)
    assert "First" in text and "Second" in text and "Third" in text
    
    # Get partial text (first two segments only)
    text = extractor.get_transcript_text(segments, start_time=0, end_time=9)
    assert "First" in text
    assert "Second" in text
    assert "Third" not in text


def test_compute_video_id(tmp_path):
    """Test video ID computation."""
    extractor = AudioExtractor(cache_dir=str(tmp_path))
    
    # Create a test file
    test_file = tmp_path / "test.mp4"
    test_file.write_bytes(b"test content for hashing")
    
    video_id = extractor._compute_video_id(test_file)
    assert len(video_id) == 12
    assert video_id == extractor._compute_video_id(test_file)  # Consistent


def test_cache_save_load(tmp_path):
    """Test saving and loading from cache."""
    extractor = AudioExtractor(cache_dir=str(tmp_path))
    
    segments = [
        TranscriptSegment(0, 5, "Hello"),
        TranscriptSegment(5, 10, "World"),
    ]
    
    cache_file = tmp_path / "test_cache.json"
    extractor._save_to_cache(cache_file, segments)
    
    loaded = extractor._load_from_cache(cache_file)
    assert len(loaded) == 2
    assert loaded[0].text == "Hello"
    assert loaded[1].start == 5
