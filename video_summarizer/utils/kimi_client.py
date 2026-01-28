"""Kimi VLM API client.

Provides interface to Kimi Vision-Language Model for analyzing
video frames and generating summaries.
"""

import json
import base64
from pathlib import Path
from typing import List, Union, Optional

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class KimiClient:
    """Client for Kimi VLM API (OpenAI-compatible)."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str = None
    ):
        """Initialize OpenAI-compatible client.
        
        Args:
            api_key: API key (use "EMPTY" for local APIs)
            base_url: API base URL
            model: Model name (if None, will be auto-detected)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required. Install: uv pip install openai")
        
        if not api_key:
            api_key = "EMPTY"
        
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        
        # Auto-detect model if not specified
        if self.model is None:
            try:
                models = self.client.models.list()
                if models.data:
                    self.model = models.data[0].id
                    print(f"Auto-detected model: {self.model}")
            except Exception:
                self.model = "default"
    
    def analyze_images(
        self,
        images: List[Union[str, Path]],
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3
    ) -> str:
        """Analyze images with Kimi VLM.
        
        Args:
            images: List of image paths
            prompt: Text prompt for analysis
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            
        Returns:
            Model response text
        """
        content = [{"type": "text", "text": prompt}]
        
        # Add images (limit to first 8)
        for img_path in images[:8]:
            img_base64 = self._encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
            })
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.5
    ) -> str:
        """Generate text without images.
        
        Args:
            prompt: Text prompt
            max_tokens: Maximum response tokens
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
