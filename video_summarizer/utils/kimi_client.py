"""Kimi VLM API client.

Provides interface to Kimi Vision-Language Model for analyzing
video frames and generating summaries.
"""

import json
import base64
import requests
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
        if not api_key:
            api_key = "EMPTY"
        
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Auto-detect model if not specified
        self.model = model
        if self.model is None:
            try:
                resp = requests.get(f"{self.base_url}/models", headers=self.headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('data'):
                        self.model = data['data'][0]['id']
                        print(f"Auto-detected model: {self.model}")
            except Exception:
                pass
        if self.model is None:
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
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data,
            timeout=300
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    
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
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data,
            timeout=300
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
