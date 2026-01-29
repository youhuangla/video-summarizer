"""Tests for VLM API client."""

import pytest
from pathlib import Path
from video_summarizer.utils.vlm_client import VLMClient


def test_vlm_client_init():
    """Test VLM client initialization."""
    client = VLMClient(
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        model="test-model"
    )
    assert client.model == "test-model"
    assert client.api_key == "test-key"


def test_vlm_client_empty_key_defaults():
    """Test that empty API key defaults to EMPTY."""
    client = VLMClient(api_key="")
    assert client.api_key == "EMPTY"


def test_vlm_client_custom_config():
    """Test custom configuration."""
    client = VLMClient(
        api_key="custom-key",
        base_url="http://localhost:8000/v1",
        model="qwen-vl"
    )
    assert client.model == "qwen-vl"
    assert client.base_url == "http://localhost:8000/v1"


def test_encode_image(tmp_path):
    """Test image encoding."""
    client = VLMClient(api_key="test")
    
    # Create a dummy image
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"fake image data")
    
    encoded = client._encode_image(img_path)
    assert isinstance(encoded, str)
    assert encoded == "ZmFrZSBpbWFnZSBkYXRh"
