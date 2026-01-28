"""Configuration for Video Summarizer.

This module defines configuration parameters for the video summarization pipeline.
Inspired by Video-Browser (https://github.com/chrisx599/Video-Browser) paper's
pyramidal perception architecture.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class SummarizerConfig:
    """Configuration for video summarization.
    
    Uses pyramidal perception approach:
    - Stage 1: Sparse uniform sampling for chapter boundary detection
    - Stage 2: Dense sampling within chapters for detailed summary
    """
    
    # API Configuration (Local OpenAI-compatible API)
    api_key: str = "EMPTY"  # Local APIs often don't need real key
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "Qwen3vl-30B-instruct-XH-NL"
    
    # Video Processing
    segment_duration: int = 600  # Split videos longer than 10 minutes
    overlap_duration: int = 30   # Overlap between segments
    sparse_frame_count: int = 20  # Stage 1: frames for chapter detection
    dense_fps: float = 0.5        # Stage 2: frames per second in chapters
    min_chapter_duration: int = 30  # Minimum chapter length in seconds
    max_chapters: int = 8
    
    # Output
    output_dir: str = "./output"
    cache_dir: str = "./cache"
    enable_cache: bool = True
    
    def __post_init__(self):
        # Allow overriding via environment variables
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            self.api_key = env_key
        env_url = os.getenv("OPENAI_BASE_URL")
        if env_url:
            self.base_url = env_url
        env_model = os.getenv("MODEL_NAME")
        if env_model:
            self.model = env_model
        
        # Create directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
