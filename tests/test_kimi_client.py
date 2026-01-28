"""Tests for Kimi API client."""

import pytest
from pathlib import Path
from video_summarizer.utils.kimi_client import KimiClient


def test_kimi_client_init():
    """Test KimiClient initialization."""
    client = KimiClient(
        api_key="test-key",
        base_url="https://test.com",
        model="test-model"
    )
    assert client.model == "test-model"
    assert client.client is not None


def test_kimi_client_no_api_key():
    """Test that empty API key raises error."""
    with pytest.raises(ValueError, match="API key is required"):
        KimiClient(api_key="")


def test_kimi_client_custom_config():
    """Test custom configuration."""
    client = KimiClient(
        api_key="custom-key",
        base_url="https://custom.moonshot.cn/v1",
        model="kimi-custom"
    )
    assert client.model == "kimi-custom"


def test_encode_image(tmp_path):
    """Test image encoding."""
    client = KimiClient(api_key="test")
    
    # Create a dummy image
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"fake image data")
    
    encoded = client._encode_image(img_path)
    assert isinstance(encoded, str)
    assert encoded == "ZmFrZSBpbWFnZSBkYXRh"
