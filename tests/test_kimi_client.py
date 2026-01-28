"""Tests for Kimi API client."""

import pytest
from pathlib import Path
from video_summarizer.utils.kimi_client import KimiClient


def test_kimi_client_init():
    """Test OpenAI client initialization."""
    client = KimiClient(
        api_key="test-key",
        base_url="http://localhost:1234/v1",
        model="test-model"
    )
    assert client.model == "test-model"
    assert client.client is not None


def test_kimi_client_empty_key_defaults():
    """Test that empty API key defaults to EMPTY."""
    client = KimiClient(api_key="")
    assert client.api_key == "EMPTY"


def test_kimi_client_custom_config():
    """Test custom configuration."""
    client = KimiClient(
        api_key="custom-key",
        base_url="http://localhost:8000/v1",
        model="qwen-vl"
    )
    assert client.model == "qwen-vl"


def test_encode_image(tmp_path):
    """Test image encoding."""
    client = KimiClient(api_key="test")
    
    # Create a dummy image
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"fake image data")
    
    encoded = client._encode_image(img_path)
    assert isinstance(encoded, str)
    assert encoded == "ZmFrZSBpbWFnZSBkYXRh"
