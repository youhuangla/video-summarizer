"""Tests for configuration module."""

import os
import pytest
from video_summarizer.config import SummarizerConfig


def test_default_config():
    """Test default configuration values."""
    config = SummarizerConfig()
    assert config.segment_duration == 600
    assert config.sparse_frame_count == 8  # Reduced for local VLM
    assert config.model == "GLM46V-Flash"  # Local VLM model
    assert config.dense_fps == 0.1  # 1 frame per 10 seconds
    assert config.image_width == 640  # Resized for VRAM saving
    assert config.image_height == 480


def test_config_from_env():
    """Test loading API key from environment variable."""
    os.environ["OPENAI_API_KEY"] = "test-key-123"
    os.environ["MODEL_NAME"] = "test-model"
    config = SummarizerConfig()
    assert config.api_key == "test-key-123"
    assert config.model == "test-model"
    del os.environ["OPENAI_API_KEY"]
    del os.environ["MODEL_NAME"]


def test_config_custom_values():
    """Test custom configuration values."""
    config = SummarizerConfig(
        api_key="custom-key",
        model="custom-model",
        segment_duration=300,
        sparse_frame_count=10,
        max_chapters=5
    )
    assert config.api_key == "custom-key"
    assert config.model == "custom-model"
    assert config.segment_duration == 300
    assert config.sparse_frame_count == 10
    assert config.max_chapters == 5


def test_directories_created(tmp_path):
    """Test that output and cache directories are created."""
    output_dir = tmp_path / "test_output"
    cache_dir = tmp_path / "test_cache"
    
    config = SummarizerConfig(
        output_dir=str(output_dir),
        cache_dir=str(cache_dir)
    )
    
    assert output_dir.exists()
    assert cache_dir.exists()
